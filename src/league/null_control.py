"""Null control: does the promotion gate promote NOTHING?

Phase 2's loop failed exactly here. It promoted two "champions" that
were bit-identical to the default theta, because it compared noisy
composite scores and took the best of eight. The fix is only worth
trusting if it survives the same test that exposed the bug.

The protocol: take ONE theta, measure it TWICE against the same pool as
two independent arms, and hand both to the gate. The true difference is
exactly zero by construction, so every promotion is a false positive.
Repeat and report the rate.

Two details make this an honest test rather than a rigged one:

- The sample size must be large enough that the gate COULD promote. Run
  it below `min_gate_games` and every trial is held for being
  underpowered, which proves nothing. The runner checks this and refuses
  to report a pass it did not earn.
- The arms must be independently measured. Evaluating once and comparing
  a number to itself would trivially pass; the whole point is that the
  simulator is not seedable, so two measurements of one genome genuinely
  differ (that is the noise we are defending against).

Run from the repo root:
    python -m src.league.null_control --trials 8 --games 100
    python -m src.league.null_control --deck grimmsnarl --trials 6
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

from ..ingestion.build_card_model import REPO_ROOT
from ..ingestion.build_effect_model import EffectIndex
from ..ingestion.card_index import CardIndex
from .fitness import DEFAULT_FIELD, Entry, evaluate, load_decks
from .gate import (Proportion, minimum_detectable_effect, pooled,
                   promotion_gate)
from .modules import module_for_deck
from .theta import Theta

DEFAULT_OUT: Final[Path] = REPO_ROOT / "data" / "league" / "null_control.json"


@dataclass(frozen=True)
class Trial:
    arm_a: Proportion
    arm_b: Proportion
    promoted: bool
    diff_low: float
    diff_high: float
    score_a: float
    score_b: float

    @property
    def delta(self) -> float:
        return self.arm_a.rate - self.arm_b.rate


def run_null_control(deck: str, trials: int, games: int, min_gate_games: int,
                     field_decks: Sequence[str] = DEFAULT_FIELD,
                     theta: Theta | None = None,
                     progress: bool = True) -> dict:
    """Measure one theta against itself, `trials` times, through the gate."""
    index, effects = CardIndex(), EffectIndex()
    field = [name for name in field_decks if name != deck]
    pool = [Entry(name=name, deck=name) for name in field]
    decks = load_decks(sorted({deck} | set(field)), index)
    theta = theta or module_for_deck(deck).default_theta()

    def measure(seed: int):
        candidate = Entry(name=deck, deck=deck, theta=theta, is_candidate=True)
        return evaluate(candidate, pool + [candidate], decks, index, effects,
                        games, seed)

    results: list[Trial] = []
    for trial in range(trials):
        # two INDEPENDENT measurements of the identical genome
        a = measure(seed=10_000 * (trial + 1))
        b = measure(seed=10_000 * (trial + 1) + 5_000)
        prop_a, prop_b = pooled(a.cells), pooled(b.cells)
        gate = promotion_gate(prop_a, prop_b, min_games=min_gate_games)
        results.append(Trial(prop_a, prop_b, gate.promoted, gate.diff_low,
                             gate.diff_high, a.score(), b.score()))
        if progress:
            flag = "FALSE POSITIVE" if gate.promoted else "held"
            print(f"  trial {trial}: A {prop_a} vs B {prop_b} "
                  f"delta {prop_a.rate - prop_b.rate:+.1%} -> {flag}",
                  flush=True)

    decided = results[0].arm_a.decided if results else 0
    promotions = sum(1 for r in results if r.promoted)
    rates = [r.arm_a.rate for r in results] + [r.arm_b.rate for r in results]
    scores = [r.score_a for r in results] + [r.score_b for r in results]
    deltas = [abs(r.delta) for r in results]
    powered = decided >= min_gate_games

    return {
        "deck": deck, "trials": trials, "games_per_cell": games,
        "cells": len(pool) + 0, "decided_per_arm": decided,
        "min_gate_games": min_gate_games,
        "powered": powered,
        "promotions": promotions,
        "false_positive_rate": promotions / trials if trials else 0.0,
        "winrate_range": [min(rates), max(rates)] if rates else None,
        "winrate_stdev": statistics.pstdev(rates) if len(rates) > 1 else 0.0,
        "score_range": [min(scores), max(scores)] if scores else None,
        "score_stdev": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
        "max_abs_delta": max(deltas) if deltas else 0.0,
        "minimum_detectable_effect": minimum_detectable_effect(decided)
        if decided else None,
        "trials_detail": [
            {"a_rate": r.arm_a.rate, "b_rate": r.arm_b.rate,
             "delta": r.delta, "diff_low": r.diff_low,
             "diff_high": r.diff_high, "promoted": r.promoted}
            for r in results],
    }


def format_report(payload: dict) -> str:
    lines = [
        f"NULL CONTROL — {payload['deck']}: identical theta, "
        f"{payload['trials']} trials, {payload['games_per_cell']} games/cell",
        f"  decided games per arm: {payload['decided_per_arm']} "
        f"(gate floor {payload['min_gate_games']})",
    ]
    if not payload["powered"]:
        lines.append("  !! NOT POWERED: every trial was held for being "
                     "underpowered, so this run proves NOTHING about the "
                     "gate's discrimination. Raise --games.")
    rate_lo, rate_hi = payload["winrate_range"] or (0, 0)
    score_lo, score_hi = payload["score_range"] or (0, 0)
    lines += [
        f"  pooled winrate of the SAME genome: {rate_lo:.1%} .. {rate_hi:.1%} "
        f"(stdev {payload['winrate_stdev']:.3f})",
        f"  composite score of the SAME genome: {score_lo:.3f} .. "
        f"{score_hi:.3f} (stdev {payload['score_stdev']:.3f})",
        f"  largest spurious delta seen: {payload['max_abs_delta']:.1%}",
        f"  minimum detectable effect at this N: "
        f"{payload['minimum_detectable_effect']:.1%}",
        f"  FALSE POSITIVES: {payload['promotions']}/{payload['trials']} "
        f"({payload['false_positive_rate']:.1%})",
    ]
    verdict = ("PASS — the gate promoted nothing"
               if payload["promotions"] == 0 and payload["powered"]
               else "FAIL — the gate promoted a genome identical to itself"
               if payload["promotions"] else "INCONCLUSIVE")
    lines.append(f"  VERDICT: {verdict}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", default="abomasnow")
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--games", type=int, default=100,
                        help="games/cell per arm (must clear the gate floor)")
    parser.add_argument("--min-gate-games", type=int, default=600)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    t0 = time.perf_counter()
    payload = run_null_control(args.deck, args.trials, args.games,
                               args.min_gate_games)
    payload["wall_seconds"] = time.perf_counter() - t0
    print()
    print(format_report(payload))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out} ({payload['wall_seconds']:.0f}s)")


if __name__ == "__main__":
    main()
