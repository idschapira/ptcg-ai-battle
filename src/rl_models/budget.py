"""Time-bank guard — the component that must never fail.

The competition gives each agent a bank, not a per-move deadline:
``actTimeout=0`` and ``observation.remainingOverageTime`` starts at 600
seconds per agent per episode. Spend it all and the agent is not merely
slow, it is DISQUALIFIED with TIMEOUT status and forfeits the game. That
is not hypothetical: in the 1032-replay corpus, the top-10 team busted
the bank four times and lost all four by forfeit.

So the guard is built around one asymmetry: running out of bank costs a
whole game, while searching less costs a fraction of one decision. Every
ambiguity resolves toward spending less.

Three mechanisms, in increasing order of how much they save us:

1.  A RESERVE that is simply never spent. The guard refuses to search
    once the bank drops to it, so the agent always finishes the episode
    on the prior, however badly everything else was calibrated.

2.  A LADDER of (candidates x determinizations) settings. As the bank
    falls the search gets cheaper, and below the last rung it stops.

3.  A MEASURED cost model. The guard times its own rollouts and keeps an
    EWMA of seconds-per-rollout, then refuses any setting whose
    PROJECTED cost would break the reserve. This is what makes the guard
    portable: nothing here is calibrated to the machine it was tuned on.
    Kaggle's 2 vCPU box being ~3x slower than the dev machine needs no
    constant — a slower box measures slower rollouts and degrades
    sooner, automatically.

The bank is tracked two ways and the guard believes whichever is more
pessimistic: ``remainingOverageTime`` as reported by the environment,
and our own cumulative measured spend subtracted from the 600s start. A
missing or malformed report therefore degrades to self-accounting rather
than to blind optimism.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final

# Competition constants, read off the environment specification carried
# in every replay (specification.observation.remainingOverageTime).
BANK_START_S: Final[float] = 600.0

# Never spend the last of the bank. 150s is 25% of it: enough to finish
# any plausible episode on the prior (our prior costs ~2.7s of MEDIAN
# episode wall on the real ladder, and 6.5s at its observed worst).
DEFAULT_RESERVE_S: Final[float] = 150.0

# Seconds per rollout assumed before we have measured any. Deliberately
# pessimistic: the first search must not be the one that overspends.
DEFAULT_ROLLOUT_PRIOR_S: Final[float] = 1.0

# Weight of the newest measurement in the EWMA of rollout cost.
EWMA_ALPHA: Final[float] = 0.2

# How much worse than the running mean a single search is allowed to be
# before the projection is considered broken. Projections are multiplied
# by this, so the guard budgets for the bad case, not the average.
SAFETY_FACTOR: Final[float] = 2.0


@dataclass(frozen=True)
class SearchTier:
    """One rung of the degradation ladder."""

    name: str
    n_candidates: int
    n_determinizations: int
    min_bank_s: float

    @property
    def rollouts(self) -> int:
        return self.n_candidates * self.n_determinizations


# Ordered richest -> cheapest.
#
# RECALIBRATED after the first field measurement. The original first
# rung sat at 400s and the worst Grimmsnarl episode drove the bank to
# 413.3s — it cleared the step by 13 seconds, which is not a margin, it
# is a coincidence. The whole ladder therefore never engaged in the one
# matchup that was over budget, so the guard contributed nothing exactly
# where it was needed.
#
# Now the richest tier gives up at 500s: past 100s of spend on a single
# episode we are already outside the profile of every agent in the
# corpus except the one that got itself disqualified, and that is the
# point at which "search less" should start rather than finish. The
# thresholds stay coarse on purpose — the measured projection below does
# the fine-grained work, and a ladder pretending to be precise about an
# unmeasured machine is a guess wearing a number.
DEFAULT_LADDER: Final[tuple[SearchTier, ...]] = (
    SearchTier("4x4", 4, 4, 500.0),
    SearchTier("3x2", 3, 2, 380.0),
    SearchTier("2x1", 2, 1, 250.0),
)


@dataclass
class BudgetStats:
    """Aggregable counters — the evidence that the guard did its job."""

    decisions: int = 0
    searches: int = 0
    skipped_reserve: int = 0        # refused: bank at/below the reserve
    skipped_projection: int = 0     # refused: no tier fits the projection
    tier_uses: dict[str, int] = field(default_factory=dict)
    measured_search_s: float = 0.0
    min_bank_seen_s: float = BANK_START_S
    rollouts_timed: int = 0

    def note_tier(self, name: str) -> None:
        self.tier_uses[name] = self.tier_uses.get(name, 0) + 1

    def summary(self) -> str:
        share = self.searches / self.decisions if self.decisions else 0.0
        return (f"decisions {self.decisions}, searches {self.searches} "
                f"({share:.0%}), tiers {self.tier_uses}, "
                f"skipped(reserve) {self.skipped_reserve}, "
                f"skipped(projection) {self.skipped_projection}, "
                f"search time {self.measured_search_s:.1f}s, "
                f"min bank {self.min_bank_seen_s:.1f}s")


class BudgetGuard:
    """Decides how much search this decision may buy, if any.

    Usage per decision:
        tier = guard.choose(obs_dict)      # None -> play the prior
        ...run the search...
        guard.record(tier, elapsed_s)      # feeds the cost model
    """

    def __init__(self, ladder: tuple[SearchTier, ...] = DEFAULT_LADDER,
                 reserve_s: float = DEFAULT_RESERVE_S,
                 bank_start_s: float = BANK_START_S,
                 rollout_prior_s: float = DEFAULT_ROLLOUT_PRIOR_S,
                 safety_factor: float = SAFETY_FACTOR,
                 stats: BudgetStats | None = None) -> None:
        self._ladder = tuple(sorted(ladder, key=lambda t: -t.rollouts))
        self._reserve_s = max(0.0, reserve_s)
        self._bank_start_s = bank_start_s
        self._rollout_s = max(1e-6, rollout_prior_s)
        self._safety = max(1.0, safety_factor)
        self._spent_s = 0.0
        self.stats = stats if stats is not None else BudgetStats()

    # ------------------------------------------------------------------ #
    # Bank accounting
    # ------------------------------------------------------------------ #

    @property
    def rollout_cost_s(self) -> float:
        """Current EWMA estimate of one rollout's cost, in seconds."""
        return self._rollout_s

    @property
    def spent_s(self) -> float:
        """Search time we have measured ourselves this episode."""
        return self._spent_s

    def reset(self) -> None:
        """New episode: forget the spend, keep the learned cost model."""
        self._spent_s = 0.0

    def bank_s(self, obs_dict: object) -> float:
        """Remaining bank, believing whichever source is more pessimistic.

        None-safe: a missing or nonsensical ``remainingOverageTime``
        leaves only self-accounting, which is the conservative side.
        """
        self_accounted = self._bank_start_s - self._spent_s
        reported = None
        if isinstance(obs_dict, dict):
            value = obs_dict.get("remainingOverageTime")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                reported = float(value)
                # isfinite first: NaN fails every range comparison, so a
                # bare `< 0 or > start` test would let it through and
                # poison every arithmetic downstream of it.
                if (not math.isfinite(reported) or reported < 0.0
                        or reported > self._bank_start_s):
                    reported = None
        bank = self_accounted if reported is None else min(reported,
                                                           self_accounted)
        self.stats.min_bank_seen_s = min(self.stats.min_bank_seen_s, bank)
        return bank

    # ------------------------------------------------------------------ #
    # Decision
    # ------------------------------------------------------------------ #

    def choose(self, obs_dict: object) -> SearchTier | None:
        """Richest tier this decision can afford, or None for prior-only."""
        self.stats.decisions += 1
        bank = self.bank_s(obs_dict)
        spendable = bank - self._reserve_s
        if spendable <= 0.0:
            self.stats.skipped_reserve += 1
            return None
        for tier in self._ladder:
            if bank < tier.min_bank_s:
                continue
            projected = tier.rollouts * self._rollout_s * self._safety
            if projected <= spendable:
                self.stats.searches += 1
                self.stats.note_tier(tier.name)
                return tier
        self.stats.skipped_projection += 1
        return None

    def record(self, tier: SearchTier | None, elapsed_s: float,
               rollouts: int | None = None) -> None:
        """Fold a completed search's real cost into the model and the spend.

        Always call this, including when the search bailed out early —
        time spent on an abandoned search is still time spent.
        """
        if elapsed_s <= 0.0:
            return
        self._spent_s += elapsed_s
        self.stats.measured_search_s += elapsed_s
        n = rollouts if rollouts is not None else (
            tier.rollouts if tier is not None else 0)
        if n > 0:
            per_rollout = elapsed_s / n
            self._rollout_s = ((1.0 - EWMA_ALPHA) * self._rollout_s
                               + EWMA_ALPHA * per_rollout)
            self.stats.rollouts_timed += n

    def note_untimed_spend(self, elapsed_s: float) -> None:
        """Account time that was not a search (prior, bookkeeping)."""
        if elapsed_s > 0.0:
            self._spent_s += elapsed_s


__all__ = ["BANK_START_S", "DEFAULT_LADDER", "DEFAULT_RESERVE_S",
           "BudgetGuard", "BudgetStats", "SearchTier"]
