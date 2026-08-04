"""Stage B — the search the leaf change makes possible.

The shipped search was 1-ply with a ROLLOUT leaf: one candidate action,
then a whole simulated game to score it. Two things followed from that,
and both are fixed here.

COST. A rollout leaf costs ~160ms (the 4x4 search spent ~2.6s to buy 16
evaluations). A value leaf costs one node: measured IN SITU, ~1.35ms.
So a leaf is ~120x cheaper, not the ~500x a bench of the primitives in
isolation suggests -- profiling the working search put 79% of its time
inside ``cg.api.search_step``, where the engine rebuilds its JSON reply
into dataclasses. That is the engine's own cost, it is paid per node by
any search, and Rule #0 says we do not go around it. 1.35ms/node is the
honest floor and every budget number here is derived from it.

What we DID remove was ours: the first working version called
``_to_dict`` on every live node to read four fields, and _to_dict is
``json.loads(json.dumps(dataclasses.asdict(...)))``. That was 45% of
search time, for four attribute reads.

DEPTH, AND WHY IT IS FREE OF THE OPPONENT MODEL. The engine re-offers
MAIN after every action that does not end the turn, so "one ply" was
one micro-action -- one attach, one item. Measured over real games, OUR
OWN turn runs 5.2 decisions deep (median 4, p90 11) with a branching
factor of only 4.6. So the whole remainder of our turn is searchable,
and searching it requires NO model of the opponent's policy: every node
between the root and the leaf is OUR decision. The opponent enters only
at the horizon, where the value head prices the position it inherits.

That is the structural answer to the 29/Jul finding. The rollout
search's entire effect lived on how well our rollout policy imitated
the real opponent (+14.7pp against our own model, -2.5pp against a
behaviour clone). This search asks the opponent nothing.

Search shape: BEAM over our own turn, per determinization.

    level 0   the candidate actions (top-K by prior score)
    level d   expand every live node's legal options, evaluate the ones
              that ended the turn / ended the game / hit the depth cap,
              keep the best ``beam`` live nodes and go on
    value     of a candidate = the best leaf reachable under it (we
              control every node in between, so max is correct), then
              averaged over determinizations

Levels are expanded breadth-first ON PURPOSE: it lets every leaf of a
level be encoded once and scored in ONE batched matmul, which is ~10x
cheaper per leaf than scoring them one at a time (0.022ms vs 0.263ms).

None-safety is the same contract as RuntimeSearchAgent: the prior
answers first, every failure path returns the prior's answer, and
``search_end()`` is in a ``finally``.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Final, Sequence

import numpy as np

from cg import api

from .encoding import ENCODING_DIM
from .search_core import DRAW_VALUE, RESULT_DRAW

logger = logging.getLogger(__name__)

# Hard ceiling on nodes expanded for one determinization, whatever the
# tier asks for. A pathological position with maximum branching at every
# level must not be able to turn one decision into an unbounded spend --
# the budget guard reacts to MEASURED cost, which is one decision too
# late if a single decision is the one that blows up.
MAX_NODES_PER_DETERMINIZATION: Final[int] = 4000


@dataclass
class ValueSearchStats:
    """What the search did. Aggregated across games by the harness."""

    decisions: int = 0
    searched: int = 0
    changed: int = 0
    nodes: int = 0                # search_step calls
    leaves: int = 0               # value-head evaluations
    leaf_batches: int = 0
    depth_reached: int = 0        # max beam level reached, any decision
    node_caps: int = 0            # determinizations that hit the node cap
    exceptions: int = 0
    search_time_s: float = 0.0
    fallback_reasons: Counter = field(default_factory=Counter)
    tier_uses: Counter = field(default_factory=Counter)

    @property
    def mean_search_ms(self) -> float:
        return (1000.0 * self.search_time_s / self.searched
                if self.searched else 0.0)

    @property
    def nodes_per_s(self) -> float:
        return self.nodes / self.search_time_s if self.search_time_s else 0.0

    def summary(self) -> str:
        share = self.searched / self.decisions if self.decisions else 0.0
        return (f"decisions {self.decisions}, searched {self.searched} "
                f"({share:.0%}), changed {self.changed}, "
                f"{self.mean_search_ms:.0f}ms/search, "
                f"nodes {self.nodes} ({self.nodes_per_s:.0f}/s), "
                f"leaves {self.leaves} in {self.leaf_batches} batches, "
                f"max depth {self.depth_reached}, node caps {self.node_caps}, "
                f"tiers {dict(self.tier_uses)}, "
                f"exceptions {self.exceptions}, "
                f"fallbacks {dict(self.fallback_reasons.most_common())}")


@dataclass(frozen=True)
class ValueTier:
    """One rung of the degradation ladder for a value-leaf search.

    Duck-types budget.SearchTier: BudgetGuard only ever reads ``name``,
    ``min_bank_s`` and ``rollouts``, so exposing a projected NODE count
    as ``rollouts`` makes the guard's EWMA a seconds-per-node model with
    no change to the guard itself.
    """

    name: str
    n_candidates: int
    n_determinizations: int
    depth: int
    beam: int
    min_bank_s: float

    @property
    def rollouts(self) -> int:
        """Projected node count -- the guard's cost unit.

        Nodes per determinization are bounded by expanding at most
        ``beam`` live nodes per level, each with the measured mean
        branching factor, for ``depth`` levels. Using the MEAN (not the
        max) here is safe because the guard multiplies by its own safety
        factor and recalibrates on what it actually measures.
        """
        per_level = self.n_candidates + self.beam * _MEAN_BRANCHING
        return max(1, int(self.n_determinizations * per_level
                          * max(1, self.depth - 1)))


_MEAN_BRANCHING: Final[float] = 4.6   # measured over real games

# Seconds per NODE assumed before anything has been measured. The
# guard's default prior is one second, which is right for a rollout and
# absurd for a node -- it would refuse every tier forever. Measured node
# cost is ~0.3ms; 5ms is a 15x pessimistic start, which is the correct
# direction for a guard whose failure mode is disqualification.
NODE_PRIOR_S: Final[float] = 0.005


# Every named tier, for pinning while the budget CURVE is measured.
# Only some of these end up on the shipped ladder.
#
# The depth numbers are bounded by the thing being searched: our own
# turn runs 5.2 decisions deep on average (median 4, p90 11), so depth 6
# already covers the large majority of turns end to end and depth 8 is
# mostly buying the tail.
ALL_TIERS: Final[dict[str, ValueTier]] = {
    t.name: t for t in (
        ValueTier("d1x2", 3, 2, 1, 1, 160.0),
        ValueTier("d2x4", 4, 4, 2, 3, 250.0),
        ValueTier("d3x6", 5, 6, 3, 4, 380.0),
        ValueTier("d4x8", 6, 8, 4, 6, 500.0),
        ValueTier("d6x8", 6, 8, 6, 6, 520.0),
        ValueTier("d6x12", 6, 12, 6, 8, 540.0),
        ValueTier("d8x12", 8, 12, 8, 8, 560.0),
        # DEPTH-ISOLATING rungs. The tiers above vary depth, candidates,
        # determinizations and beam together, so "d2x4 beat d6x12" cannot
        # be read as a statement about depth -- it is four changes at
        # once. These hold candidates=5, determinizations=6, beam=4 fixed
        # and move ONLY depth, which is the comparison that answers
        # whether searching further into our own turn helps at all.
        ValueTier("iso-d1", 5, 6, 1, 4, 0.0),
        ValueTier("iso-d2", 5, 6, 2, 4, 0.0),
        ValueTier("iso-d3", 5, 6, 3, 4, 0.0),
        ValueTier("iso-d4", 5, 6, 4, 4, 0.0),
    )
}

# Richest -> cheapest. Thresholds mirror budget.DEFAULT_LADDER: the
# guard's projection does the fine-grained work, the rungs only need to
# be coarsely ordered.
DEFAULT_VALUE_LADDER: Final[tuple[ValueTier, ...]] = (
    ALL_TIERS["d6x8"],
    ALL_TIERS["d4x8"],
    ALL_TIERS["d3x6"],
    ALL_TIERS["d2x4"],
    ALL_TIERS["d1x2"],
)


class ValueLeafEvaluator:
    """Encodes observations and scores them with the value head, batched.

    ``seat`` is ours. The head always answers for the player TO MOVE, so
    a node where the opponent is on turn is negated -- that is the whole
    convention, and it is why value_collect records both seats.
    """

    __slots__ = ("_net", "_encoder", "stats")

    def __init__(self, net, encoder, stats: ValueSearchStats) -> None:
        self._net = net
        self._encoder = encoder
        self.stats = stats

    def values(self, observations: Sequence, seat: int) -> np.ndarray:
        """[N] values in [-1, 1] from OUR seat's point of view."""
        if not observations:
            return np.zeros(0, np.float32)
        rows = np.empty((len(observations), ENCODING_DIM), np.float32)
        signs = np.empty(len(observations), np.float32)
        for i, obs in enumerate(observations):
            rows[i] = self._encoder.encode(obs)
            current = obs.current
            acting = current.yourIndex if current is not None else seat
            signs[i] = 1.0 if acting == seat else -1.0
        out = self._net.value_batch(rows) * signs
        self.stats.leaves += len(observations)
        self.stats.leaf_batches += 1
        return out


def _terminal_value(current, seat: int) -> float | None:
    """Engine-decided value for ``seat``, or None if not terminal."""
    if current is None or current.result == -1:
        return None
    if current.result == seat:
        return 1.0
    return DRAW_VALUE * 2.0 - 1.0 if current.result == RESULT_DRAW else -1.0


def _legal_single_options(observation) -> list[int] | None:
    """Option indices we may branch on at this node, or None if we cannot.

    Read straight off the Observation DATACLASS. The obvious spelling is
    ``_to_dict(observation)`` and then four dict lookups, but _to_dict is
    ``json.loads(json.dumps(dataclasses.asdict(...)))`` — it serialises
    the entire game state to rebuild it as dicts. Profiling the first
    working version put 45% of all search time in that one call. Reading
    the four fields we need costs nothing.

    Only single-select decisions are branched: multi-select and open
    ``looking`` zones are the same states the shared eligibility filter
    refuses at the root, and letting the interior of the search visit
    what the root refuses would mean the search explores states the
    measurements never covered.
    """
    select = observation.select
    if select is None or select.deck is not None:
        return None
    options = select.option or []
    if not options or select.maxCount != 1:
        return None
    current = observation.current
    if current is not None and current.looking is not None:
        return None
    return list(range(len(options)))


def beam_search_turn(root, seat: int, candidates: list[int], depth: int,
                     beam: int, leaf: ValueLeafEvaluator,
                     stats: ValueSearchStats,
                     max_nodes: int = MAX_NODES_PER_DETERMINIZATION,
                     ) -> dict[int, float]:
    """Value of each candidate under ONE determinization.

    Every node between the root and a leaf is ours, so the value of a
    candidate is the MAX over its reachable leaves. Beam search
    approximates that max from below, which is the safe direction: it
    can only understate a candidate, never invent a line that is not
    there.
    """
    # level 0: apply each candidate, giving one live node per candidate
    live: list[tuple[int, object]] = []          # (candidate, node)
    best: dict[int, float] = {}
    pending: list[tuple[int, object]] = []       # (candidate, observation)
    nodes_used = 0

    for cand in candidates:
        if nodes_used >= max_nodes:
            break
        node = api.search_step(root.searchId, [cand])
        nodes_used += 1
        stats.nodes += 1
        current = node.observation.current
        terminal = _terminal_value(current, seat)
        if terminal is not None:
            best[cand] = max(best.get(cand, -2.0), terminal)
            continue
        acting = current.yourIndex if current is not None else seat
        if acting != seat or depth <= 1:
            # turn passed (or no depth left): this is a leaf
            pending.append((cand, node.observation))
            continue
        live.append((cand, node))

    if pending:
        values = leaf.values([obs for _, obs in pending], seat)
        for (cand, _), value in zip(pending, values):
            best[cand] = max(best.get(cand, -2.0), float(value))
        pending = []

    level = 1
    while live and level < depth and nodes_used < max_nodes:
        stats.depth_reached = max(stats.depth_reached, level)
        next_live: list[tuple[int, object, float]] = []
        for cand, node in live:
            options = _legal_single_options(node.observation)
            if options is None:
                # cannot branch here -- price the node itself and stop
                pending.append((cand, node.observation))
                continue
            for opt in options:
                if nodes_used >= max_nodes:
                    stats.node_caps += 1
                    break
                child = api.search_step(node.searchId, [opt])
                nodes_used += 1
                stats.nodes += 1
                current = child.observation.current
                terminal = _terminal_value(current, seat)
                if terminal is not None:
                    best[cand] = max(best.get(cand, -2.0), terminal)
                    continue
                acting = current.yourIndex if current is not None else seat
                if acting != seat or level + 1 >= depth:
                    pending.append((cand, child.observation))
                else:
                    next_live.append((cand, child, 0.0))

        # One batched forward for the whole level: leaves that end the
        # line AND the live nodes, whose values are what the beam is
        # pruned on.
        if pending:
            values = leaf.values([obs for _, obs in pending], seat)
            for (cand, _), value in zip(pending, values):
                best[cand] = max(best.get(cand, -2.0), float(value))
            pending = []
        if next_live:
            values = leaf.values([n.observation for _, n, _ in next_live],
                                 seat)
            scored = [(c, n, float(v)) for (c, n, _), v in
                      zip(next_live, values)]
            # A global beam would let one candidate's good line crowd out
            # every other candidate's, and then the comparison the search
            # exists to make is between one searched candidate and a set
            # of unsearched ones. So the beam is PER CANDIDATE.
            per_cand: dict[int, list] = {}
            for cand, node, value in scored:
                per_cand.setdefault(cand, []).append((value, node))
            live = []
            for cand, entries in per_cand.items():
                entries.sort(key=lambda e: -e[0])
                for value, node in entries[:max(1, beam)]:
                    live.append((cand, node))
            # NOTE: a live node's value prunes the beam and NOTHING else.
            # It is tempting to fold it into best[cand] as well -- the
            # position is real and the head was trained on mid-turn
            # positions -- but we cannot STOP there: the engine keeps
            # re-offering MAIN until something ends the turn. Counting it
            # would credit a candidate with a position it cannot choose to
            # hold, and would bias toward whichever candidate happened to
            # get explored deeper, since later nodes in a turn have had
            # more played and tend to score higher. Only horizons count:
            # terminal, turn passed, depth cap, or nothing left to branch.
        else:
            live = []
        level += 1

    if pending:
        values = leaf.values([obs for _, obs in pending], seat)
        for (cand, _), value in zip(pending, values):
            best[cand] = max(best.get(cand, -2.0), float(value))

    return {c: v for c, v in best.items() if v > -2.0}


__all__ = ["ALL_TIERS", "DEFAULT_VALUE_LADDER",
           "MAX_NODES_PER_DETERMINIZATION", "NODE_PRIOR_S",
           "ValueLeafEvaluator", "ValueSearchStats", "ValueTier",
           "beam_search_turn"]
