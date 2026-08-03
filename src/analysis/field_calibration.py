"""The calibration table: internal field cells against the REAL targets.

Every round of opponent-pilot work asks the same question — does the
internal field now sit where the ladder says it sits? — and every round
it has been answered by hand. This is that table as one command.

The real targets are the ladder's own numbers for the SHIP, per cell,
and they are FIXED here on purpose: a target that moves with the
experiment is not a target. They come from the submission the cells were
mined against (54917180); the per-cell sample is small and printed with
its own count so nobody reads 12.5% as if it were 600 games.

Discipline (unchanged, and it is the point):
  * sharded over PROCESSES, never threads — cg.api keeps one global
    agent_ptr and search_end() frees everything allocated against it;
  * pooled Wilson 95% intervals, never a bare winrate, because a ~150
    game A/B on this engine is noise;
  * both columns re-measured in the SAME run whenever --baseline-arm is
    given, so before/after is same-machine, same-day.

Run from the repo root:
    python -m src.analysis.field_calibration --games 600 \
        --opp-arm heuristic-tempo-scaled-aggro \
        --baseline-arm heuristic-tempo-scaled
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path
from typing import Final

from ..environment_wrapper.ab_test import wilson_interval
from ..ingestion.build_card_model import REPO_ROOT

OUT_DIR: Final[Path] = REPO_ROOT / "data" / "processed" / "calibration"

# (cell, opposing decklist, REAL winrate for us, games behind that real
# number). Fixed targets — see the module docstring.
CELLS: Final[tuple[tuple[str, str, float, int], ...]] = (
    ("Alakazam",     "data/decks/meta_alakazam.csv",          0.356, 59),
    ("Mega Lucario", "data/decks/meta_mega_lucario.csv",      0.500, 12),
    ("Archaludon",   "data/decks/meta_archaludon.csv",        0.906, 32),
    ("Kangaskhan",   "data/decks/meta_crustle_kangaskhan.csv", 0.750, 12),
    ("Spidops",      "data/decks/meta_spidops.csv",           0.125, 8),
    ("Starmie",      "data/decks/meta_starmie.csv",           0.625, 8),
)

OUR_SIDE: Final[str] = "deck.csv@crustle-v3"


def _run_shard(payload: tuple[str, str, str, int, int]) -> dict:
    """Worker: play ``games`` of one cell. Fresh engine per process."""
    our_text, opp_deck_text, opp_arm, games, seed = payload
    from ..deckbuilding.gauntlet import run_pair
    from ..deckbuilding.legality import read_deck_ids
    from ..environment_wrapper.ab_test import ArmMetrics, ArmSpec, arm_factory
    from ..ingestion.build_effect_model import EffectIndex
    from ..ingestion.card_index import CardIndex

    index, effects = CardIndex(), EffectIndex()
    our_deck_text, our_arm = our_text.split("@", 1)
    our_deck = read_deck_ids(REPO_ROOT / our_deck_text)
    opp_deck = read_deck_ids(REPO_ROOT / opp_deck_text)
    spec_a, spec_b = ArmSpec.parse(our_arm), ArmSpec.parse(opp_arm)
    metrics_a, metrics_b = ArmMetrics(), ArmMetrics()
    pair = run_pair(arm_factory(spec_a, index, effects, metrics_a, our_deck),
                    arm_factory(spec_b, index, effects, metrics_b, opp_deck),
                    our_deck, opp_deck, games, seed)
    return {"a_wins": pair.a_wins, "b_wins": pair.b_wins,
            "draws": pair.draws, "errors": list(pair.errors)}


def run_cell(opp_deck: str, opp_arm: str, games: int, workers: int,
             seed: int) -> dict:
    per_shard = max(1, games // workers)
    payloads = [(OUR_SIDE, opp_deck, opp_arm, per_shard, seed + 100_000 * i)
                for i in range(workers)]
    with mp.Pool(processes=workers) as pool:
        shards = pool.map(_run_shard, payloads)
    a_wins = sum(s["a_wins"] for s in shards)
    b_wins = sum(s["b_wins"] for s in shards)
    draws = sum(s["draws"] for s in shards)
    errors = [e for s in shards for e in s["errors"]]
    decided = a_wins + b_wins
    lo, hi = wilson_interval(a_wins, decided)
    return {"a_wins": a_wins, "b_wins": b_wins, "draws": draws,
            "decided": decided, "winrate": a_wins / decided if decided else 0.0,
            "ci": [lo, hi], "exceptions": len(errors),
            "error_samples": errors[:3]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opp-arm", type=str, required=True,
                        help="pilot flying the OPPONENT deck in every cell")
    parser.add_argument("--baseline-arm", type=str, default=None,
                        help="second pilot, measured in the same run")
    parser.add_argument("--cells", nargs="+", default=None,
                        help="restrict to these cells by name; needed for "
                             "archetype-specific pilots (a BC clone of one "
                             "archetype flying another deck measures "
                             "nothing)")
    parser.add_argument("--games", type=int, default=600)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    arms = [args.opp_arm] if args.baseline_arm is None else [
        args.baseline_arm, args.opp_arm]
    cells = CELLS
    if args.cells:
        wanted = {c.casefold() for c in args.cells}
        cells = tuple(c for c in CELLS if c[0].casefold() in wanted)
        if not cells:
            raise SystemExit(f"nenhuma celula casa com {args.cells}; "
                             f"disponiveis: {[c[0] for c in CELLS]}")
    results: dict[str, dict[str, dict]] = {arm: {} for arm in arms}
    t0 = time.perf_counter()
    for name, deck, _real, _n in cells:
        for arm in arms:
            results[arm][name] = run_cell(deck, arm, args.games,
                                          args.workers, args.seed)
            row = results[arm][name]
            print(f"  {name:14s} {arm:38s} {row['winrate']:6.1%} "
                  f"[{row['ci'][0]:.1%}, {row['ci'][1]:.1%}]  "
                  f"n={row['decided']}  exc={row['exceptions']}")

    base_arm = args.baseline_arm
    print("\n" + "=" * 96)
    print("CALIBRACAO — celulas internas contra os alvos REAIS (fixos)")
    print("=" * 96)
    header = f"  {'celula':14s}{'real':>9s}"
    if base_arm:
        header += f"{'antes':>9s}"
    header += f"{'depois':>9s}"
    if base_arm:
        header += f"{'erro antes':>12s}"
    header += f"{'erro depois':>13s}{'melhora':>10s}"
    print(header)
    print("  " + "-" * 92)

    before_errors, after_errors = [], []
    for name, _deck, real, real_n in cells:
        after = results[arms[-1]][name]["winrate"]
        after_err = after - real
        after_errors.append(abs(after_err))
        line = f"  {name:14s}{real:8.1%} "
        if base_arm:
            before = results[base_arm][name]["winrate"]
            before_err = before - real
            before_errors.append(abs(before_err))
            line += f"{before:8.1%} "
        line += f"{after:8.1%} "
        if base_arm:
            line += f"{before_err * 100:+11.1f}"
        line += f"{after_err * 100:+12.1f}"
        if base_arm:
            line += f"{(abs(before_err) - abs(after_err)) * 100:+9.1f}"
        print(line + f"   (real n={real_n})")

    print("  " + "-" * 92)
    if before_errors:
        print(f"  erro absoluto medio: antes "
              f"{sum(before_errors) / len(before_errors) * 100:.1f}pp  ->  "
              f"depois {sum(after_errors) / len(after_errors) * 100:.1f}pp")
    else:
        print(f"  erro absoluto medio: "
              f"{sum(after_errors) / len(after_errors) * 100:.1f}pp")
    total_exc = sum(r["exceptions"] for arm in arms
                    for r in results[arm].values())
    print(f"  exceptions no total: {total_exc} (tem de ser 0)")
    print(f"  wall {(time.perf_counter() - t0) / 60:.1f} min")

    out = args.out or (OUT_DIR / f"{arms[-1]}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"games_per_cell": args.games, "arms": arms,
               "cells": {name: {"real": real, "real_games": real_n,
                                **{arm: results[arm][name] for arm in arms}}
                         for name, _d, real, real_n in cells}}
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
