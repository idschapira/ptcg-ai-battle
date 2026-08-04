"""Stage C — is the value search actually STRONGER than the ship?

Two questions, deliberately asked separately, because the last round of
search work passed the first and failed the second.

  1. THE FIELD A/B. Candidate vs CrustleAgent v3, same deck, against the
     corrected internal field, weighted by the share each archetype had
     on OUR ladder. Newcombe interval on the DIFFERENCE (two overlapping
     Wilson intervals do not answer whether the difference contains
     zero), broken out by seat because this engine cannot pair seeds.

  2. THE HONESTY TEST. The same comparison against opponents that are
     NOT our model of anything: behaviour clones of real ladder humans
     (BC-Majkel and BC-Yushin on Alakazam, BC-Luca and BC-Dries on
     Grimmsnarl, BC-Spidops). On 29/Jul the rollout search measured
     +14.7pp against the heuristic it used to model its opponent, +6.3pp
     against a parametric agent that inherits from that heuristic, and
     -2.5pp against a behaviour clone. The effect was monotone in how
     closely the opponent resembled our own model, which means it was
     never a property of the search.

     The value search has no rollout policy, so there is nothing for it
     to be secretly right about -- but that is an argument, not a
     measurement, and the argument is exactly the kind this project has
     been wrong about before. Gate C is read on THIS number.

Run from the repo root:
    python -m src.analysis.value_search_ab --a "deck.csv@value-search" \
        --games 600
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Final

from ..environment_wrapper.ab_test import newcombe_difference
from ..ingestion.build_card_model import REPO_ROOT
from .pilot_ab import CORRECTED_FIELD, measure

OUT_DIR: Final[Path] = REPO_ROOT / "data" / "processed" / "value_search"
STATS: Final[str] = "models/feature_stats.npz"

# (cell, decklist, opponent arm, share of OUR real ladder).
#
# Shares come from field_coverage.json — the 215 decided games of
# submission 54917180 — renormalised over the cells we can actually
# play ("unknown", 7.0%, has no decklist).
CORRECTED: Final[tuple[tuple[str, str, str, float], ...]] = (
    ("Alakazam",     "data/decks/meta_alakazam.csv",           CORRECTED_FIELD, 0.2744),
    ("Grimmsnarl",   "data/decks/meta_grimmsnarl.csv",         CORRECTED_FIELD, 0.1674),
    ("Mega Lucario", "data/decks/meta_mega_lucario.csv",       CORRECTED_FIELD, 0.1488),
    ("Archaludon",   "data/decks/meta_archaludon.csv",         CORRECTED_FIELD, 0.1488),
    ("Kangaskhan",   "data/decks/meta_crustle_kangaskhan.csv", CORRECTED_FIELD, 0.0558),
    ("Starmie",      "data/decks/meta_starmie.csv",            CORRECTED_FIELD, 0.0372),
    ("Spidops",      "data/decks/meta_spidops.csv",            CORRECTED_FIELD, 0.0372),
    ("mirror",       "deck.csv",                               "crustle-v3",    0.0186),
)

# The clones. Shares are the SHARE OF THE ARCHETYPE those cells stand
# for, renormalised inside this field, so the weighted aggregate answers
# "what would this be worth on a ladder made of these opponents".
CLONES: Final[tuple[tuple[str, str, str, float], ...]] = (
    ("Alakazam/Majkel",  "data/decks/meta_alakazam.csv",
     f"network,models/bc_majkel.npz,{STATS}", 0.1372),
    ("Alakazam/Yushin",  "data/decks/meta_alakazam.csv",
     f"network,models/bc_yushin.npz,{STATS}", 0.1372),
    ("Grimmsnarl/Luca",  "data/decks/meta_grimmsnarl.csv",
     f"network,models/bc_luca.npz,{STATS}", 0.0837),
    ("Grimmsnarl/Dries", "data/decks/meta_grimmsnarl.csv",
     f"network,models/bc_dries.npz,{STATS}", 0.0837),
    ("Spidops/BC",       "data/decks/meta_spidops.csv",
     f"network,models/bc_spidops_v2.npz,{STATS}", 0.0372),
)

FIELDS: Final[dict[str, tuple]] = {"corrected": CORRECTED, "clones": CLONES}


def run_field(name: str, field: tuple, arm_a: str, arm_b: str, games: int,
              workers: int, seed: int) -> dict:
    print(f"\n{'=' * 104}")
    print(f"{name.upper()} FIELD — A={arm_a}  vs  B={arm_b}  "
          f"(N={games}/arm/cell)")
    print("=" * 104)
    print(f"  {'cell':18s}{'weight':>8s}{'A':>9s}{'B':>9s}{'delta':>9s}"
          f"{'IC95 (Newcombe)':>22s}{'seat0':>9s}{'seat1':>9s}{'exc':>6s}")
    print("  " + "-" * 100)
    rows = []
    for cell, deck, opp_arm, share in field:
        a = measure(arm_a, deck, opp_arm, games, workers, seed)
        b = measure(arm_b, deck, opp_arm, games, workers, seed + 7)
        diff, lo, hi = newcombe_difference(b["wins"], b["decided"],
                                           a["wins"], a["decided"])
        verdict = "A>B" if lo > 0 else "B>A" if hi < 0 else "="
        seats = []
        for s in (0, 1):
            da, db = a["seat_decided"][s], b["seat_decided"][s]
            seats.append((a["seat_wins"][s] / da) - (b["seat_wins"][s] / db)
                         if da and db else 0.0)
        rows.append({"cell": cell, "share": share, "a": a, "b": b,
                     "diff": diff, "ci": [lo, hi], "verdict": verdict,
                     "seats": seats})
        print(f"  {cell:18s}{share:8.1%}{a['winrate']:9.1%}{b['winrate']:9.1%}"
              f"{diff * 100:+9.1f}   [{lo * 100:+6.1f}, {hi * 100:+6.1f}]"
              f"{verdict:>4s}{seats[0] * 100:+9.1f}{seats[1] * 100:+9.1f}"
              f"{a['exceptions'] + b['exceptions']:6d}")

    total_w = sum(r["share"] for r in rows)
    weighted = sum(r["share"] * r["diff"] for r in rows) / total_w
    # Pooled difference across the whole field, so the aggregate carries
    # an interval instead of a point estimate. Cells are pooled with
    # their real weights by construction of the run (equal N per cell),
    # so this is the UNWEIGHTED pooled effect and is reported as such.
    a_w = sum(r["a"]["wins"] for r in rows)
    a_n = sum(r["a"]["decided"] for r in rows)
    b_w = sum(r["b"]["wins"] for r in rows)
    b_n = sum(r["b"]["decided"] for r in rows)
    pooled, plo, phi = newcombe_difference(b_w, b_n, a_w, a_n)
    exceptions = sum(r["a"]["exceptions"] + r["b"]["exceptions"]
                     for r in rows)
    print("  " + "-" * 100)
    print(f"  WEIGHTED by real ladder share: {weighted * 100:+.2f}pp "
          f"(weights sum {total_w:.1%})")
    print(f"  POOLED (equal N per cell):     {pooled * 100:+.2f}pp   "
          f"IC95 [{plo * 100:+.2f}, {phi * 100:+.2f}]   "
          f"{'SIGNIFICANT' if plo > 0 or phi < 0 else 'contains zero'}")
    print(f"  exceptions: {exceptions} (must be 0)")
    return {"field": name, "weighted_delta": weighted,
            "pooled_delta": pooled, "pooled_ci": [plo, phi],
            "exceptions": exceptions, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=str, default="deck.csv@value-search")
    parser.add_argument("--b", type=str, default="deck.csv@crustle-v3")
    parser.add_argument("--games", type=int, default=600)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--fields", nargs="+", default=["clones", "corrected"],
                        choices=sorted(FIELDS))
    parser.add_argument("--cells", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    t0 = time.perf_counter()
    results = []
    for name in args.fields:
        field = FIELDS[name]
        if args.cells:
            wanted = {c.casefold() for c in args.cells}
            field = tuple(c for c in field if c[0].casefold() in wanted)
        if field:
            results.append(run_field(name, field, args.a, args.b, args.games,
                                     args.workers, args.seed))

    print("\n" + "=" * 104)
    print("GATE C — a positive, significant gain against opponents we do "
          "NOT model")
    print("=" * 104)
    for r in results:
        lo, hi = r["pooled_ci"]
        sig = "excludes zero" if (lo > 0 or hi < 0) else "CONTAINS ZERO"
        print(f"  {r['field']:10s} weighted {r['weighted_delta'] * 100:+6.2f}pp"
              f"   pooled {r['pooled_delta'] * 100:+6.2f}pp "
              f"[{lo * 100:+.2f}, {hi * 100:+.2f}]  {sig}")
    clone = next((r for r in results if r["field"] == "clones"), None)
    if clone is not None:
        lo, _hi = clone["pooled_ci"]
        passed = lo > 0 and clone["exceptions"] == 0
        print(f"\n  GATE C: {'PASS' if passed else 'FAIL'} — the clone field "
              f"is the one that decides, and its interval "
              f"{'excludes' if lo > 0 else 'does not exclude'} zero "
              f"from below.")
    print(f"  wall {(time.perf_counter() - t0) / 60:.1f} min")

    out = args.out or (OUT_DIR / "gate_c.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"a": args.a, "b": args.b, "games": args.games,
                   "results": results}, fh, indent=1)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
