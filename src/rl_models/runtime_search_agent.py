"""RuntimeSearchAgent — shallow determinized search, submissible.

The offline SearchAgent cannot ship: it is handed the opponent's
decklist, and it spends whatever time it likes. This agent is the same
search with those two holes closed.

    opponent's deck   OpponentDeckEstimator labels the archetype from
                      the cards revealed so far and supplies that
                      archetype's consensus list as a HYPOTHESIS. Until
                      it is confident there is no hypothesis, and the
                      agent is exactly its prior.

    time              BudgetGuard reads ``remainingOverageTime`` out of
                      the observation and degrades the search — fewer
                      candidates, fewer determinizations, then none —
                      as the 600s bank drains. Busting the bank is
                      disqualification, so the guard is allowed to be
                      wrong only in the direction of searching less.

The resulting risk profile is deliberately lopsided. Every path that is
not "confident estimate AND affordable AND eligible AND the
determinizer closed" ends at the prior — which is CrustleAgent v3, the
current ship. The floor of this agent is the thing it replaces; the
search can only add, and its worst case is having wasted some bank.

The prior is called BEFORE any search on every decision, so a legal
answer exists before anything that could fail has run. ``search_end()``
is in a ``finally``; exceptions are counted, never propagated.

Offline evaluation can pass ``opponent_deck_override`` to pin the
presumed deck (bypassing the estimator) — that is how the "what does
the search itself buy, with estimation held perfect" question gets
asked separately from "how good is the estimator".
"""

from __future__ import annotations

import copy
import logging
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Sequence

from cg import api

from ..agent_heuristics.crustle_agent import CrustleAgent
from ..agent_heuristics.heuristic_agent import HeuristicAgent
from ..agent_heuristics.random_agent import read_deck_csv
from ..ingestion.build_effect_model import EffectIndex
from ..ingestion.card_index import CardIndex
from .budget import BudgetGuard, SearchTier
from .determinize import runtime_determinize
from .opponent_estimator import OpponentDeckEstimator
from .search_core import (eligibility_reason, rank_candidates,
                          rollout_to_terminal)

logger = logging.getLogger(__name__)

DEFAULT_ROLLOUT_CAP: Final[int] = 600

# How the opponent is played inside a rollout.
#   generic  always HeuristicAgent — we model their DECK, not their pilot
#   match    pick the pilot from the ESTIMATED archetype where we have a
#            specialised one (only Crustle today). Modelling a mill/stall
#            opponent as a generic pilot misprices exactly the races that
#            decide those matchups.
OPPONENT_PILOT_GENERIC: Final[str] = "generic"
OPPONENT_PILOT_MATCH: Final[str] = "match"

# estimated archetype -> the specialised pilot that flies it
_CRUSTLE_ARCHETYPES: Final[frozenset[str]] = frozenset({
    "Crustle mill (ours)", "Crustle stall (other)",
    "Crustle + Mega Kangaskhan stall",
})

# How much better than the PRIOR'S OWN CHOICE a candidate's measured win
# rate must be before the search is allowed to override it, as a fraction
# of a win (0.25 == one extra win in four determinizations).
#
# 0.0 reproduces the configuration that lost the N=600 mirror A/B, and
# the reason to suspect this knob is the optimizer's curse: at 4x4 each
# candidate's value is the mean of FOUR Bernoulli rollouts, so taking an
# argmax over four such estimates mostly selects whichever candidate got
# lucky, and the winner's estimate is biased upward by the selection
# itself. Overriding a well-tuned prior on four samples is a good way to
# convert a strong policy into a noisy one -- the search changed the
# prior's answer on 387 of 1726 decisions and finished 5pp WORSE.
#
# A margin makes the search prove its case before it is believed, which
# is the right default when the thing it is arguing against is the
# current ship.
DEFAULT_OVERRIDE_MARGIN: Final[float] = 0.0

# ADAPTIVE ALLOCATION. A decision is DOMINATED when the prior's best
# option beats the runner-up by more than this fraction of the option
# score range; searching one is paying full price to be told what the
# prior already knew. Censused over 40 real games, 42.1% of our
# selections are dominated and only 23.6% are contested (~13/game), so
# skipping the dominated ones is where the budget comes from.
#
# None disables the filter — every eligible decision is searched, which
# is what the first (over-budget) measurements did.
DEFAULT_CONTESTED_MARGIN: Final[float] = 0.10


@dataclass
class RuntimeSearchStats:
    """What actually happened — every number the checkpoint report needs."""

    decisions: int = 0
    searched: int = 0            # searches that produced a comparison
    changed: int = 0             # searches that overrode the prior
    rollouts: int = 0
    rollout_caps: int = 0
    exceptions: int = 0
    search_time_s: float = 0.0
    fallback_reasons: Counter = field(default_factory=Counter)
    tier_uses: Counter = field(default_factory=Counter)

    @property
    def mean_search_ms(self) -> float:
        return (1000.0 * self.search_time_s / self.searched
                if self.searched else 0.0)

    def summary(self) -> str:
        share = self.searched / self.decisions if self.decisions else 0.0
        return (f"decisions {self.decisions}, searched {self.searched} "
                f"({share:.0%}), changed {self.changed}, "
                f"{self.mean_search_ms:.0f}ms/search, "
                f"rollouts {self.rollouts} (cap {self.rollout_caps}), "
                f"tiers {dict(self.tier_uses)}, "
                f"exceptions {self.exceptions}, "
                f"fallbacks {dict(self.fallback_reasons.most_common())}")


class RuntimeSearchAgent:
    """Kaggle agent contract: ``agent(obs_dict) -> list[int]``."""

    def __init__(
        self,
        deck_path: str | Path | None = None,
        index: CardIndex | None = None,
        effects: EffectIndex | None = None,
        variant: str = "v3",
        estimator: OpponentDeckEstimator | None = None,
        guard: BudgetGuard | None = None,
        seed: int = 0,
        rollout_max_selections: int = DEFAULT_ROLLOUT_CAP,
        stats: RuntimeSearchStats | None = None,
        opponent_deck_override: Sequence[int] | None = None,
        own_deck_ids: Sequence[int] | None = None,
        enable_search: bool = True,
        opponent_pilot: str = OPPONENT_PILOT_GENERIC,
        override_margin: float = DEFAULT_OVERRIDE_MARGIN,
        contested_margin: float | None = None,
    ) -> None:
        self._index = index if index is not None else CardIndex()
        self._effects = effects if effects is not None else EffectIndex()
        self._deck_path = str(deck_path) if deck_path else None
        self._variant = variant
        self._prior = CrustleAgent(seed=seed, deck_path=self._deck_path,
                                   index=self._index, effects=self._effects,
                                   variant=variant)
        self._own_deck = ([int(c) for c in own_deck_ids] if own_deck_ids
                          else self._read_own_deck())
        self._enable_search = enable_search
        self._opponent_pilot = opponent_pilot
        self._override_margin = max(0.0, override_margin)
        self._contested_margin = contested_margin
        self.estimator = (estimator if estimator is not None
                          else OpponentDeckEstimator(index=self._index))
        self.guard = guard if guard is not None else BudgetGuard()
        self._cap = rollout_max_selections
        self._rng = random.Random(seed)
        self._seed = seed
        self.stats = stats if stats is not None else RuntimeSearchStats()
        self._override = (tuple(int(c) for c in opponent_deck_override)
                          if opponent_deck_override else None)
        self.last_candidate_values: dict[int, float] | None = None

    # ------------------------------------------------------------------ #
    # Deck
    # ------------------------------------------------------------------ #

    def _read_own_deck(self) -> list[int]:
        """Our 60. read_deck_csv already falls back to the Kaggle path."""
        try:
            return [int(c) for c in read_deck_csv(self._deck_path)]
        except Exception:  # noqa: BLE001 — the prior still answers selections
            logger.warning("could not read own deck from %r",
                           self._deck_path, exc_info=True)
            return []

    # ------------------------------------------------------------------ #
    # Contract
    # ------------------------------------------------------------------ #

    def __call__(self, obs_dict: dict) -> list[int]:
        self.last_candidate_values = None
        if not isinstance(obs_dict, dict) or obs_dict.get("select") is None:
            # initial selection == a new episode: forget the last game
            self.estimator.reset()
            self.guard.reset()
            return list(self._own_deck)

        self.stats.decisions += 1
        answer = self._prior(copy.deepcopy(obs_dict))
        if not self._enable_search:
            # the FLOOR of this agent, made explicit and testable: with
            # search disabled it is byte-for-byte its prior.
            self.stats.fallback_reasons["search-disabled"] += 1
            return answer
        scores = getattr(self._prior, "last_scores", None)
        estimate = self.estimator.observe(obs_dict)

        reason = eligibility_reason(obs_dict, answer, scores)
        if reason is not None:
            self.stats.fallback_reasons[reason] += 1
            return answer

        opp_deck = (list(self._override) if self._override is not None
                    else list(estimate.deck_ids))
        if not opp_deck or (self._override is None and not estimate.usable):
            self.stats.fallback_reasons[
                f"estimator:{estimate.reason}"] += 1
            return answer

        # Adaptive allocation, BEFORE the budget is touched: a dominated
        # decision must cost nothing at all, otherwise skipping it frees
        # no bank for the contested ones.
        if not _is_contested(list(scores), self._contested_margin):
            self.stats.fallback_reasons["dominated"] += 1
            return answer

        tier = self.guard.choose(obs_dict)
        if tier is None:
            self.stats.fallback_reasons["budget"] += 1
            return answer

        rollouts_done = 0
        t0 = time.perf_counter()
        try:
            best, rollouts_done = self._search(obs_dict, list(scores),
                                               answer[0], opp_deck, tier,
                                               estimate.archetype)
        except Exception:  # noqa: BLE001 — counted, prior answer still legal
            self.stats.exceptions += 1
            logger.debug("search failed; falling back to prior", exc_info=True)
            best = None
        finally:
            elapsed = time.perf_counter() - t0
            self.stats.search_time_s += elapsed
            # Time spent is time spent, even when the search bailed out.
            # rollouts_done is passed EXACTLY (0 included): a search that
            # ran no rollouts must charge its time to the bank without
            # feeding the cost model, or a cheap early bail would teach
            # the guard that rollouts are cheaper than they are.
            self.guard.record(tier, elapsed, rollouts_done)

        if best is None:
            return answer
        self.stats.searched += 1
        self.stats.tier_uses[tier.name] += 1
        if best != answer[0]:
            self.stats.changed += 1
        return [best]

    # ------------------------------------------------------------------ #
    # Search: 1-ply, paired determinizations, leaf = terminal rollout
    # ------------------------------------------------------------------ #

    def _search(self, obs_dict: dict, scores: list[float], prior_choice: int,
                opp_deck: list[int], tier: SearchTier,
                archetype: str) -> tuple[int | None, int]:
        """Returns (chosen option or None, rollouts actually run)."""
        seat = obs_dict["current"]["yourIndex"]
        candidates = rank_candidates(scores, prior_choice,
                                     tier.n_candidates)
        if len(set(candidates)) < 2:
            self.stats.fallback_reasons["single-candidate"] += 1
            return None, 0

        obs_cls = api.to_observation_class(copy.deepcopy(obs_dict))
        values = {i: 0.0 for i in candidates}
        rollouts = 0
        samples = 0
        for _ in range(tier.n_determinizations):
            det = runtime_determinize(obs_dict, seat, self._own_deck,
                                      opp_deck, self._rng)
            if det is None:
                self.stats.fallback_reasons["determinize"] += 1
                break
            root = api.search_begin(obs_cls, *det, [])
            try:
                for i in candidates:
                    branch = api.search_step(root.searchId, [i])
                    value, hit_cap = rollout_to_terminal(
                        branch, seat, self._rollout_agent(),
                        self._opponent_rollout_agent(archetype), self._cap)
                    values[i] += value
                    rollouts += 1
                    self.stats.rollouts += 1
                    if hit_cap:
                        self.stats.rollout_caps += 1
            finally:
                api.search_end()
            samples += 1

        if samples == 0:
            return None, rollouts
        self.last_candidate_values = {i: values[i] / samples
                                      for i in candidates}
        # Ties resolve to the prior's ranking: the search only overrides
        # on evidence, never on a coin flip.
        best = max(candidates, key=lambda i: (values[i], scores[i]))
        if best == prior_choice or self._override_margin <= 0.0:
            return best, rollouts
        # And with a margin, "evidence" means MORE than one lucky
        # rollout: the burden of proof is on the search, because what it
        # is arguing against is the current ship.
        gain = (values[best] - values.get(prior_choice, 0.0)) / samples
        if gain < self._override_margin:
            self.stats.fallback_reasons["below-margin"] += 1
            return prior_choice, rollouts
        return best, rollouts

    # ------------------------------------------------------------------ #
    # Rollout policies
    # ------------------------------------------------------------------ #

    def _rollout_agent(self) -> CrustleAgent:
        """Our side of a rollout: the same pilot that is deciding."""
        return CrustleAgent(seed=self._rng.randrange(1 << 30),
                            deck_path=self._deck_path, index=self._index,
                            effects=self._effects, variant=self._variant)

    def _opponent_rollout_agent(self, archetype: str):
        """Their side of a rollout.

        Default models the DECK but not the PILOT: a generic heuristic,
        because we have no read on how they play. That default turned
        out to be the expensive assumption — see STRATEGY_JOURNAL
        [29/Jul]. In the mirror it prices a stall/mill race as if the
        opponent were a generic pilot, which is exactly the judgement
        the matchup turns on, and the search lost 44.8% [40.9, 48.8] at
        N=600 with an otherwise perfect deck read.

        ``opponent_pilot="match"`` uses the specialised pilot when the
        estimated archetype has one.
        """
        seed = self._rng.randrange(1 << 30)
        if (self._opponent_pilot == OPPONENT_PILOT_MATCH
                and archetype in _CRUSTLE_ARCHETYPES):
            return CrustleAgent(seed=seed, deck_path=self._deck_path,
                                index=self._index, effects=self._effects,
                                variant=self._variant)
        return HeuristicAgent(seed=seed, index=self._index,
                              effects=self._effects)


def _is_contested(scores: list[float], margin: float | None) -> bool:
    """Is the prior's top choice close enough to be worth searching?

    ``margin`` is a fraction of the option score RANGE, not an absolute
    score gap, so it means the same thing across decisions whose scores
    live on different scales (the bands run 20-80 depending on what is
    legal). None disables the filter.
    """
    if margin is None:
        return True
    if len(scores) < 2:
        return False
    ordered = sorted(scores, reverse=True)
    spread = ordered[0] - ordered[-1]
    if spread <= 0.0:
        return True          # every option scores alike: genuinely open
    return (ordered[0] - ordered[1]) / spread <= margin


__all__ = ["DEFAULT_CONTESTED_MARGIN", "DEFAULT_ROLLOUT_CAP",
           "OPPONENT_PILOT_GENERIC", "OPPONENT_PILOT_MATCH",
           "RuntimeSearchAgent", "RuntimeSearchStats"]
