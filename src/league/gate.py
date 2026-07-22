"""Noise-proof promotion: when is a challenger genome ACTUALLY better?

Phase 2 measured the damage: with 24 games/cell the same theta scored
anywhere in 0.651-0.767 (stdev 0.036), and taking the best of 8 genomes
inflated the score by +0.044 with zero learning. Two champions promoted
by that loop were bit-identical to the default theta. The loop was
selecting noise.

This module is the fix. Three ideas, in order of importance:

1. TEST THE POOLED WINRATE, NOT THE COMPOSITE SCORE.
   `Fitness.score` is 0.5*radar + 0.3*mean + 0.2*WORST-CELL. That last
   term is a MINIMUM over ~11 independently noisy cells, and a min-of-N
   is a variance amplifier: it systematically finds whichever cell got
   unlucky, so it moves a lot even when nothing changed. It is a fine
   thing to REPORT and to rank on, but a terrible thing to promote on.
   The gate therefore runs on total wins over total decided games —
   one proportion, with the whole cohort's sample size behind it.

2. COMPARE THE TWO PROPORTIONS PROPERLY, NOT BY EYEBALLING OVERLAP.
   "Their Wilson intervals don't overlap" is a needlessly conservative
   test (it is roughly a 99% test, not 95%). The correct Wilson-family
   answer is Newcombe's score interval for the DIFFERENCE of two
   proportions, built from the two individual Wilson intervals. We
   promote only when that interval's lower bound is above zero.

3. SPEND THE GAMES WHERE THE DECISION IS.
   Screening many genomes and testing one are different problems.
   Successive halving (see `evolve.py`) finds the finalist cheaply; only
   that finalist is measured against the incumbent at the sample size
   the gate actually needs. Uniform N over the whole population would
   cost ~13x for no extra confidence in the decision that matters.

The honest consequence, quantified by `minimum_detectable_effect`: at
any budget we can afford per generation, only fairly LARGE effects are
detectable. That is a real limit, not a flaw in the test — and it is
much better to know the floor than to promote below it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Final, Sequence

from ..environment_wrapper.ab_test import wilson_interval

#: One-sided alpha for promotion. A challenger must clear the bar in the
#: RIGHT direction; there is no reason to spend power on two tails.
DEFAULT_ALPHA: Final[float] = 0.05
DEFAULT_POWER: Final[float] = 0.80

#: Typical pooled winrate of a candidate against the cohort (~73-80% in
#: Phase 1/2). Power depends on p(1-p), so the planning numbers are
#: quoted at this operating point.
OPERATING_WINRATE: Final[float] = 0.75


def _z(prob: float) -> float:
    return NormalDist().inv_cdf(prob)


@dataclass(frozen=True)
class Proportion:
    """wins / decided, with its Wilson interval."""

    wins: int
    decided: int

    @property
    def rate(self) -> float:
        return self.wins / self.decided if self.decided else 0.5

    def interval(self, confidence: float = 0.95) -> tuple[float, float]:
        return wilson_interval(self.wins, self.decided,
                               _z(1 - (1 - confidence) / 2))

    def __str__(self) -> str:
        low, high = self.interval()
        return (f"{self.rate:.1%} [{low:.1%}-{high:.1%}] "
                f"({self.wins}/{self.decided})")


def newcombe_difference(a: Proportion, b: Proportion,
                        confidence: float = 0.95) -> tuple[float, float]:
    """CI for (a.rate - b.rate) — Newcombe's score (Wilson) method.

    Built from the two individual Wilson intervals, which is what makes
    it well behaved at extreme rates and small samples, where the naive
    normal-approximation interval famously is not."""
    if a.decided <= 0 or b.decided <= 0:
        return (-1.0, 1.0)
    p1, p2 = a.rate, b.rate
    l1, u1 = a.interval(confidence)
    l2, u2 = b.interval(confidence)
    delta = p1 - p2
    lower = delta - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    upper = delta + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return (max(-1.0, lower), min(1.0, upper))


@dataclass(frozen=True)
class GateResult:
    """The promotion decision, with everything needed to audit it."""

    challenger: Proportion
    incumbent: Proportion
    diff_low: float
    diff_high: float
    promoted: bool
    reason: str
    min_games: int

    @property
    def delta(self) -> float:
        return self.challenger.rate - self.incumbent.rate

    def format(self) -> str:
        verdict = "PROMOTE" if self.promoted else "hold"
        return (f"[{verdict}] challenger {self.challenger} vs incumbent "
                f"{self.incumbent} | delta {self.delta:+.1%} "
                f"CI [{self.diff_low:+.1%}, {self.diff_high:+.1%}] "
                f"— {self.reason}")

    def to_dict(self) -> dict:
        return {"challenger_wins": self.challenger.wins,
                "challenger_decided": self.challenger.decided,
                "challenger_rate": self.challenger.rate,
                "incumbent_wins": self.incumbent.wins,
                "incumbent_decided": self.incumbent.decided,
                "incumbent_rate": self.incumbent.rate,
                "delta": self.delta, "diff_low": self.diff_low,
                "diff_high": self.diff_high, "promoted": self.promoted,
                "reason": self.reason, "min_games": self.min_games}


def promotion_gate(challenger: Proportion, incumbent: Proportion,
                   confidence: float = 0.95,
                   min_games: int = 0) -> GateResult:
    """Promote only on a Wilson-significant improvement.

    Two ways to fail, and both matter:
    - too few decided games to have any power -> hold, and say so
      (an underpowered 'pass' is how a loop starts eating noise again);
    - the difference interval includes zero -> hold.
    """
    low, high = newcombe_difference(challenger, incumbent, confidence)
    if challenger.decided < min_games or incumbent.decided < min_games:
        return GateResult(challenger, incumbent, low, high, False,
                          f"underpowered: need >= {min_games} decided games "
                          f"per arm", min_games)
    if low > 0.0:
        return GateResult(challenger, incumbent, low, high, True,
                          "difference CI excludes zero", min_games)
    return GateResult(challenger, incumbent, low, high, False,
                      "difference CI includes zero", min_games)


# ---------------------------------------------------------------------- #
# Power planning — what the budget actually buys
# ---------------------------------------------------------------------- #

def minimum_detectable_effect(games_per_arm: int,
                              rate: float = OPERATING_WINRATE,
                              alpha: float = DEFAULT_ALPHA,
                              power: float = DEFAULT_POWER) -> float:
    """Smallest winrate gain detectable at this budget (absolute, e.g.
    0.046 = 4.6 percentage points)."""
    if games_per_arm <= 0:
        return 1.0
    z_sum = _z(1 - alpha) + _z(power)
    variance = 2.0 * rate * (1.0 - rate) / games_per_arm
    return z_sum * math.sqrt(variance)


def games_for_effect(effect: float, rate: float = OPERATING_WINRATE,
                     alpha: float = DEFAULT_ALPHA,
                     power: float = DEFAULT_POWER) -> int:
    """Decided games PER ARM needed to detect `effect` (absolute)."""
    if effect <= 0:
        return 10 ** 9
    z_sum = _z(1 - alpha) + _z(power)
    return math.ceil((z_sum ** 2) * 2.0 * rate * (1.0 - rate) / (effect ** 2))


def power_table(cells: int, per_cell: Sequence[int] = (24, 50, 100, 200, 400),
                rate: float = OPERATING_WINRATE) -> list[tuple[int, int, float]]:
    """(games/cell, total decided games, minimum detectable effect)."""
    rows = []
    for games in per_cell:
        total = games * cells
        rows.append((games, total, minimum_detectable_effect(total, rate)))
    return rows


def format_power_table(cells: int,
                       per_cell: Sequence[int] = (24, 50, 100, 200, 400),
                       rate: float = OPERATING_WINRATE) -> str:
    lines = [f"power at p={rate:.0%}, alpha={DEFAULT_ALPHA} one-sided, "
             f"power={DEFAULT_POWER:.0%}, {cells} cells:",
             "  games/cell   total games   min detectable gain"]
    for games, total, effect in power_table(cells, per_cell, rate):
        lines.append(f"  {games:>10d}   {total:>11d}   {effect:>17.1%}")
    return "\n".join(lines)


def pooled(cells: Sequence) -> Proportion:
    """Total wins / total decided across a Fitness's cells.

    This is the gate's statistic: one proportion carrying the entire
    cohort's sample size, instead of a composite whose worst-cell term
    is a min-of-N variance amplifier."""
    wins = sum(int(getattr(c, "wins", 0)) for c in cells)
    decided = sum(int(getattr(c, "wins", 0)) + int(getattr(c, "losses", 0))
                  for c in cells)
    return Proportion(wins=wins, decided=decided)


__all__ = ["Proportion", "GateResult", "newcombe_difference", "promotion_gate",
           "minimum_detectable_effect", "games_for_effect", "power_table",
           "format_power_table", "pooled", "DEFAULT_ALPHA", "DEFAULT_POWER",
           "OPERATING_WINRATE"]
