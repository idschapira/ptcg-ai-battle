"""Pilot A/B against the field, as a RELATIVE instrument.

Absolute calibration of the internal field is closed (02/Ago): five
rounds of opponent work moved the Alakazam cell from +47.2pp to +40.5pp
and a matched behaviour clone did no better, so the internal field does
not predict ladder winrate in absolute terms. What it does do — and the
whole reason it still earns its keep — is compare two of OUR pilots
under the SAME opponent, where the bias is common to both arms and
cancels in the difference. That is what Gate C and the gauntlet always
were; this module makes it the explicit contract.

Two disciplines the earlier ad-hoc A/Bs kept getting wrong:

  WEIGHTING. A mean over cells weights Starmie (8 real games, 3.7% of
  the field) like Alakazam (59 games, 27.4%). The aggregate here is
  weighted by the share each archetype actually had on OUR ladder —
  data/processed/field_coverage.json, not the top-100 radar, because
  those are different populations (the radar has 6 teams playing
  Alakazam; we met 54 distinct ones).

  THE DIFFERENCE, NOT TWO INTERVALS. Two overlapping Wilson intervals
  do not answer whether the difference contains zero. Newcombe's
  hybrid-score interval does, and it is what every verdict here uses.

Seats are split out because this engine cannot pair seeds: if an effect
lives in one seat only it is a seating artifact, not a pilot gain.

Run from the repo root:
    python -m src.analysis.pilot_ab --a crustle-v4 --b crustle-v3 \
        --games 600
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path
from typing import Final

from ..environment_wrapper.ab_test import newcombe_difference, wilson_interval
from ..ingestion.build_card_model import REPO_ROOT

OUT_DIR: Final[Path] = REPO_ROOT / "data" / "processed" / "pilot_ab"

# The corrected internal field: the opponent pilot that four rounds of
# calibration produced (mean absolute error 28.8pp, the best we have).
CORRECTED_FIELD: Final[str] = "heuristic-tempo-scaled-routing-gust-conserve,8"

# (cell, decklist, share of OUR real ladder). Shares from
# field_coverage.json — the 215 decided games of submission 54917180.
# "unknown" (7.0%) has no decklist and is left out of the weighting, so
# the weights below are renormalised over what we can actually play.
FIELD: Final[tuple[tuple[str, str, float], ...]] = (
    ("Alakazam",     "data/decks/meta_alakazam.csv",           0.2744),
    ("Grimmsnarl",   "data/decks/meta_grimmsnarl.csv",         0.1674),
    ("Mega Lucario", "data/decks/meta_mega_lucario.csv",       0.1488),
    ("Archaludon",   "data/decks/meta_archaludon.csv",         0.1488),
    ("Kangaskhan",   "data/decks/meta_crustle_kangaskhan.csv", 0.0558),
    ("Starmie",      "data/decks/meta_starmie.csv",            0.0372),
    ("Spidops",      "data/decks/meta_spidops.csv",            0.0372),
    ("mirror",       "deck.csv",                               0.0186),
)


def _run_shard(payload: tuple[str, str, str, int, int]) -> dict:
    """Worker: ``games`` of one (our pilot, cell) pair. Fresh engine."""
    our_side, opp_deck_text, opp_arm, games, seed = payload
    our_deck_text, our_arm = (our_side.split("@", 1) if "@" in our_side
                              else ("deck.csv", our_side))
    from ..deckbuilding.gauntlet import run_pair
    from ..deckbuilding.legality import read_deck_ids
    from ..environment_wrapper.ab_test import ArmMetrics, ArmSpec, arm_factory
    from ..ingestion.build_effect_model import EffectIndex
    from ..ingestion.card_index import CardIndex

    index, effects = CardIndex(), EffectIndex()
    our_deck = read_deck_ids(REPO_ROOT / our_deck_text)
    opp_deck = read_deck_ids(REPO_ROOT / opp_deck_text)
    spec_a, spec_b = ArmSpec.parse(our_arm), ArmSpec.parse(opp_arm)
    metrics_a, metrics_b = ArmMetrics(), ArmMetrics()
    pair = run_pair(arm_factory(spec_a, index, effects, metrics_a, our_deck),
                    arm_factory(spec_b, index, effects, metrics_b, opp_deck),
                    our_deck, opp_deck, games, seed)
    return {"a_wins": pair.a_wins, "b_wins": pair.b_wins,
            "draws": pair.draws, "errors": list(pair.errors),
            "a_wins_seat": list(pair.a_wins_by_seat),
            "decided_seat": list(pair.decided_by_seat)}


def measure(our_arm: str, opp_deck: str, opp_arm: str, games: int,
            workers: int, seed: int) -> dict:
    per_shard = max(1, games // workers)
    payloads = [(our_arm, opp_deck, opp_arm, per_shard, seed + 100_000 * i)
                for i in range(workers)]
    with mp.Pool(processes=workers) as pool:
        shards = pool.map(_run_shard, payloads)
    wins = sum(s["a_wins"] for s in shards)
    losses = sum(s["b_wins"] for s in shards)
    errors = [e for s in shards for e in s["errors"]]
    seat_w = [sum(s["a_wins_seat"][i] for s in shards) for i in (0, 1)]
    seat_n = [sum(s["decided_seat"][i] for s in shards) for i in (0, 1)]
    decided = wins + losses
    return {"wins": wins, "decided": decided,
            "winrate": wins / decided if decided else 0.0,
            "ci": list(wilson_interval(wins, decided)),
            "exceptions": len(errors), "seat_wins": seat_w,
            "seat_decided": seat_n}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=str, required=True,
                        help="candidate side: <arm> or <deck.csv>@<arm>")
    parser.add_argument("--b", type=str, default="crustle-v3",
                        help="incumbent side (the ship): same grammar")
    parser.add_argument("--opponent", type=str, default=CORRECTED_FIELD)
    parser.add_argument("--cells", nargs="+", default=None)
    parser.add_argument("--games", type=int, default=600)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    field = FIELD
    if args.cells:
        wanted = {c.casefold() for c in args.cells}
        field = tuple(c for c in FIELD if c[0].casefold() in wanted)

    t0 = time.perf_counter()
    rows = []
    for name, deck, share in field:
        a = measure(args.a, deck, args.opponent, args.games, args.workers,
                    args.seed)
        b = measure(args.b, deck, args.opponent, args.games, args.workers,
                    args.seed + 7)
        diff, lo, hi = newcombe_difference(b["wins"], b["decided"],
                                           a["wins"], a["decided"])
        verdict = ("A>B" if lo > 0 else "B>A" if hi < 0 else "=")
        rows.append({"cell": name, "share": share, "a": a, "b": b,
                     "diff": diff, "ci": [lo, hi], "verdict": verdict})
        print(f"  {name:14s} A {a['winrate']:6.1%}  B {b['winrate']:6.1%}  "
              f"delta {diff * 100:+5.1f}pp  IC95 [{lo * 100:+5.1f}, "
              f"{hi * 100:+5.1f}]  {verdict}  exc={a['exceptions'] + b['exceptions']}")

    print("\n" + "=" * 100)
    print(f"A/B RELATIVO — A={args.a}  vs  B={args.b}")
    print(f"oponente em todas as celulas: {args.opponent}")
    print("=" * 100)
    print(f"  {'celula':14s}{'peso':>7s}{'A':>9s}{'B':>9s}{'delta':>9s}"
          f"{'IC95 da diferenca':>22s}{'assento 0':>12s}{'assento 1':>12s}")
    print("  " + "-" * 96)
    total_w = sum(r["share"] for r in rows)
    weighted = 0.0
    for r in rows:
        a, b = r["a"], r["b"]
        seats = []
        for s in (0, 1):
            da, db = a["seat_decided"][s], b["seat_decided"][s]
            if da and db:
                seats.append((a["seat_wins"][s] / da) - (b["seat_wins"][s] / db))
            else:
                seats.append(0.0)
        weighted += r["share"] * r["diff"]
        print(f"  {r['cell']:14s}{r['share']:7.1%}{a['winrate']:9.1%}"
              f"{b['winrate']:9.1%}{r['diff'] * 100:+9.1f}"
              f"   [{r['ci'][0] * 100:+6.1f}, {r['ci'][1] * 100:+6.1f}]"
              f"{seats[0] * 100:+12.1f}{seats[1] * 100:+12.1f}")
    print("  " + "-" * 96)
    print(f"  AGREGADO ponderado pelo campo REAL: "
          f"{weighted / total_w * 100:+.2f}pp   (pesos somam {total_w:.1%}; "
          f"'unknown' 7,0% fica de fora por nao ter decklist)")
    exceptions = sum(r["a"]["exceptions"] + r["b"]["exceptions"] for r in rows)
    print(f"  exceptions no total: {exceptions} (tem de ser 0)")
    print(f"  wall {(time.perf_counter() - t0) / 60:.1f} min")

    out = args.out or (OUT_DIR / f"{args.a}_vs_{args.b}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"a": args.a, "b": args.b, "opponent": args.opponent,
                   "games_per_cell": args.games,
                   "weighted_delta": weighted / total_w, "rows": rows}, fh,
                  indent=1)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
