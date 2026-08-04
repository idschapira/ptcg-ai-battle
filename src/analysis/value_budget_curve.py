"""Stage B — the budget curve: what depth fits in the 600s bank?

The resource that binds is a BANK PER EPISODE, not a deadline per move:
``remainingOverageTime`` starts at 600s per agent per episode and
exhausting it is TIMEOUT, which is disqualification and a loss, not
slowness. So the only cost number that means anything is the wall time
of a whole EPISODE, projected onto a box assumed ~3x slower than this
one.

This measures each rung of the value-search ladder separately (the
shipped arm lets the guard pick; pinning a rung is how the CURVE gets
built) and reports, per rung:

    episode wall     median / p95 / max on this box
    projected 3x     max episode x3 against the 600s bank
    nodes, depth     what the search actually explored
    searched share   how much of the bank the adaptive filter spends

The verdict line answers Gate B: does a materially deeper search than
the shipped 1-ply fit inside 50% of the bank?

Run from the repo root:
    python -m src.analysis.value_budget_curve --games 40
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path
from typing import Final

from ..ingestion.build_card_model import REPO_ROOT

OUT_DIR: Final[Path] = REPO_ROOT / "data" / "processed" / "value_search"

CORRECTED_FIELD: Final[str] = "heuristic-tempo-scaled-routing-gust-conserve,8"

# Cells chosen for COST, not for strength: Grimmsnarl is the longest
# matchup we have (median rollout depth 106 selections vs 58 in the
# mirror), so it is where an episode gets expensive, and Alakazam is the
# biggest share of the real ladder.
DEFAULT_CELLS: Final[tuple[tuple[str, str], ...]] = (
    ("grimmsnarl", "data/decks/meta_grimmsnarl.csv"),
    ("alakazam", "data/decks/meta_alakazam.csv"),
    ("mirror", "deck.csv"),
)

DEFAULT_ARMS: Final[tuple[str, ...]] = (
    "value-search-fixed,d1x2",
    "value-search-fixed,d2x4",
    "value-search-fixed,d3x6",
    "value-search-fixed,d4x8",
    "value-search",           # guard picks the rung (the shipped shape)
)

KAGGLE_SLOWDOWN: Final[float] = 3.0
BANK_S: Final[float] = 600.0


def _run_shard(payload: tuple[str, str, str, int, int]) -> dict:
    our_arm, opp_deck_text, opp_arm, games, seed = payload
    from ..deckbuilding.gauntlet import run_pair
    from ..deckbuilding.legality import read_deck_ids
    from ..environment_wrapper.ab_test import ArmMetrics, ArmSpec, arm_factory
    from ..ingestion.build_effect_model import EffectIndex
    from ..ingestion.card_index import CardIndex

    index, effects = CardIndex(), EffectIndex()
    our_deck = read_deck_ids(REPO_ROOT / "deck.csv")
    opp_deck = read_deck_ids(REPO_ROOT / opp_deck_text)
    ma, mb = ArmMetrics(), ArmMetrics()
    pair = run_pair(
        arm_factory(ArmSpec.parse(our_arm), index, effects, ma, our_deck),
        arm_factory(ArmSpec.parse(opp_arm), index, effects, mb, opp_deck),
        our_deck, opp_deck, games, seed)
    search = ma.search
    budget = ma.budget
    return {
        "a_wins": pair.a_wins, "b_wins": pair.b_wins, "draws": pair.draws,
        "errors": list(pair.errors),
        "episode_wall_s": list(ma.episode_wall_s),
        "decisions": getattr(search, "decisions", 0),
        "searched": getattr(search, "searched", 0),
        "changed": getattr(search, "changed", 0),
        "nodes": getattr(search, "nodes", 0),
        "leaves": getattr(search, "leaves", 0),
        "leaf_batches": getattr(search, "leaf_batches", 0),
        "depth_reached": getattr(search, "depth_reached", 0),
        "node_caps": getattr(search, "node_caps", 0),
        "exceptions": getattr(search, "exceptions", 0),
        "search_time_s": getattr(search, "search_time_s", 0.0),
        "tier_uses": dict(getattr(search, "tier_uses", {}) or {}),
        "fallbacks": dict(getattr(search, "fallback_reasons", {}) or {}),
        "skipped_reserve": getattr(budget, "skipped_reserve", 0),
        "skipped_projection": getattr(budget, "skipped_projection", 0),
        "min_bank_seen_s": getattr(budget, "min_bank_seen_s", BANK_S),
    }


def measure(arm: str, cell_deck: str, opp_arm: str, games: int,
            workers: int, seed: int) -> dict:
    per = max(1, games // workers)
    payloads = [(arm, cell_deck, opp_arm, per, seed + 100_000 * i)
                for i in range(workers)]
    with mp.Pool(processes=workers) as pool:
        shards = pool.map(_run_shard, payloads)

    walls = sorted(w for s in shards for w in s["episode_wall_s"])
    n = len(walls)
    median = walls[n // 2] if n else 0.0
    p95 = walls[min(n - 1, int(n * 0.95))] if n else 0.0
    worst = walls[-1] if n else 0.0
    agg = {k: sum(s[k] for s in shards) for k in
           ("decisions", "searched", "changed", "nodes", "leaves",
            "leaf_batches", "node_caps", "exceptions", "skipped_reserve",
            "skipped_projection")}
    agg["search_time_s"] = sum(s["search_time_s"] for s in shards)
    agg["depth_reached"] = max(s["depth_reached"] for s in shards)
    tiers: dict[str, int] = {}
    for s in shards:
        for k, v in s["tier_uses"].items():
            tiers[k] = tiers.get(k, 0) + v
    wins = sum(s["a_wins"] for s in shards)
    losses = sum(s["b_wins"] for s in shards)
    errors = [e for s in shards for e in s["errors"]]
    return {
        "arm": arm, "episodes": n, "median_wall_s": median, "p95_wall_s": p95,
        "max_wall_s": worst,
        "projected_max_s": worst * KAGGLE_SLOWDOWN,
        "bank_share": worst * KAGGLE_SLOWDOWN / BANK_S,
        "winrate": wins / max(wins + losses, 1),
        "wins": wins, "losses": losses,
        "exceptions_games": len(errors), "tiers": tiers, **agg,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS))
    parser.add_argument("--cells", nargs="+", default=None)
    parser.add_argument("--opponent", type=str, default=CORRECTED_FIELD)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cells = DEFAULT_CELLS
    if args.cells:
        wanted = {c.casefold() for c in args.cells}
        cells = tuple(c for c in DEFAULT_CELLS if c[0].casefold() in wanted)

    t0 = time.perf_counter()
    rows = []
    for cell_name, deck in cells:
        print(f"\n=== cell {cell_name} vs {args.opponent} "
              f"({args.games} games/arm) ===")
        print(f"  {'arm':26s}{'wall med':>10s}{'p95':>8s}{'max':>8s}"
              f"{'3x max':>9s}{'bank':>7s}{'srch%':>7s}{'nodes/dec':>10s}"
              f"{'depth':>6s}{'chg%':>6s}{'exc':>5s}")
        print("  " + "-" * 104)
        for arm in args.arms:
            m = measure(arm, deck, args.opponent, args.games, args.workers,
                        args.seed)
            m["cell"] = cell_name
            rows.append(m)
            srch = m["searched"] / max(m["decisions"], 1)
            nodes_dec = m["nodes"] / max(m["searched"], 1)
            chg = m["changed"] / max(m["searched"], 1)
            print(f"  {arm:26s}{m['median_wall_s']:10.2f}"
                  f"{m['p95_wall_s']:8.2f}{m['max_wall_s']:8.2f}"
                  f"{m['projected_max_s']:9.1f}{m['bank_share']:7.0%}"
                  f"{srch:7.0%}{nodes_dec:10.0f}{m['depth_reached']:6d}"
                  f"{chg:6.0%}{m['exceptions'] + m['exceptions_games']:5d}")

    print("\n" + "=" * 104)
    print("GATE B — does a materially deeper search fit the bank?")
    print("=" * 104)
    worst_by_arm: dict[str, float] = {}
    for r in rows:
        worst_by_arm[r["arm"]] = max(worst_by_arm.get(r["arm"], 0.0),
                                     r["bank_share"])
    for arm, share in worst_by_arm.items():
        depth = max((r["depth_reached"] for r in rows if r["arm"] == arm),
                    default=0)
        nodes = max((r["nodes"] / max(r["searched"], 1)
                     for r in rows if r["arm"] == arm), default=0)
        verdict = "FITS" if share <= 0.50 else "OVER BUDGET"
        print(f"  {arm:26s} worst bank share {share:5.0%}   "
              f"beam depth {depth}   {nodes:6.0f} nodes/decision   {verdict}")
    total_exc = sum(r["exceptions"] + r["exceptions_games"] for r in rows)
    total_nodes = sum(r["nodes"] for r in rows)
    total_time = sum(r["search_time_s"] for r in rows)
    print(f"\n  throughput: {total_nodes / max(total_time, 1e-9):,.0f} "
          f"nodes/s  ({total_nodes:,} nodes in {total_time:.1f}s of search)")
    print(f"  exceptions across every arm and cell: {total_exc} (must be 0)")
    print(f"  wall {(time.perf_counter() - t0) / 60:.1f} min")

    out = args.out or (OUT_DIR / "budget_curve.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"opponent": args.opponent, "games": args.games,
                   "rows": rows}, fh, indent=1)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
