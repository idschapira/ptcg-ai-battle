"""Portfolio harvest: the honest matrix, and which PAIR of finals covers it.

Kaggle lets us keep a small number of active submissions, so the real
question is not "which deck is best" but "which PAIR covers the meta".
Those are different questions whenever the candidates form a
rock-paper-scissors, which Phase 2 showed ours do:

    Crustle  >  Abomasnow  >  Grimmsnarl
    Starmie  >  Crustle,  and Abomasnow roughly holds Starmie

A single champion inherits its own worst cell. A PAIR only inherits the
cell that BOTH members lose — so the right score for a pair is the
per-opponent MAXIMUM over its members (you get to bring the better
matchup), weighted by how much that opponent matters.

Two coverage numbers are reported, and they answer different questions:

  radar coverage   weighted mean of max(member winrates) per opponent —
                   how the pair does on the meta as it is weighted today.
  worst cell       the opponent where the pair's BEST member is weakest —
                   the hole neither deck patches. This is the number that
                   decides tournaments, and it is why the mean alone is
                   not enough.

Everything is measured, not assumed: each cell carries a Wilson interval
and cells whose interval straddles 50% are marked `~` (directional).
The `vs alakazam` column stays called out because that cell historically
inflates against a module pilot (~30pp vs a competent one).

This FEEDS the portfolio decision. It does not make it.

Run from the repo root:
    python -m src.league.portfolio --games 120
    python -m src.league.portfolio --games 200 --candidates crustle_e10 grimmsnarl abomasnow
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Mapping, Sequence

from ..ingestion.build_card_model import REPO_ROOT
from ..ingestion.build_effect_model import EffectIndex
from ..ingestion.card_index import CardIndex
from .fitness import (BENCHMARK_CELL, DEFAULT_FIELD, RADAR_WEIGHTS, Cell,
                      Entry, Fitness, evaluate, load_decks)
from .gate import pooled
from .modules import module_for_deck

DEFAULT_CANDIDATES: Final[tuple[str, ...]] = ("crustle_e10", "grimmsnarl",
                                              "abomasnow")
DEFAULT_OUT: Final[Path] = REPO_ROOT / "data" / "league" / "portfolio.json"


@dataclass(frozen=True)
class PairCoverage:
    """What a two-submission portfolio covers."""

    members: tuple[str, ...]
    per_opponent: dict[str, float]
    best_member: dict[str, str]
    radar: float
    plain_mean: float
    worst_opponent: str
    worst_rate: float
    #: opponents where the pair NEEDS both members (they disagree a lot)
    complementarity: float

    def format(self) -> str:
        return (f"{' + '.join(self.members):<28s} radar {self.radar:6.1%} | "
                f"mean {self.plain_mean:6.1%} | worst {self.worst_opponent} "
                f"{self.worst_rate:6.1%} | complementarity "
                f"{self.complementarity:5.1%}")

    def to_dict(self) -> dict:
        return {"members": list(self.members), "radar": self.radar,
                "mean": self.plain_mean, "worst_opponent": self.worst_opponent,
                "worst_rate": self.worst_rate,
                "complementarity": self.complementarity,
                "per_opponent": self.per_opponent,
                "best_member": self.best_member}


def coverage(members: Sequence[str],
             matrix: Mapping[str, Fitness]) -> PairCoverage:
    """Best-member-per-opponent coverage of a portfolio.

    A portfolio brings the better matchup to each opponent, so the cell
    value is the MAX over members — not the average. That is precisely
    why a rock-paper-scissors trio can beat any single deck."""
    opponents: list[str] = []
    for name in members:
        for cell in matrix[name].cells:
            if cell.opponent not in opponents and cell.opponent not in members:
                opponents.append(cell.opponent)

    per_opponent: dict[str, float] = {}
    best_member: dict[str, str] = {}
    spread_sum = 0.0
    for opponent in opponents:
        rates: list[tuple[float, str]] = []
        for name in members:
            cell = matrix[name].cell(opponent)
            if cell is not None:
                rates.append((cell.winrate, name))
        if not rates:
            continue
        best_rate, owner = max(rates)
        per_opponent[opponent] = best_rate
        best_member[opponent] = owner
        if len(rates) > 1:
            spread_sum += best_rate - min(r for r, _ in rates)

    if not per_opponent:
        return PairCoverage(tuple(members), {}, {}, 0.0, 0.0, "n/a", 0.0, 0.0)

    weighted = weight_total = 0.0
    for opponent, rate in per_opponent.items():
        weight = RADAR_WEIGHTS.get(opponent, 1.0)
        weighted += weight * rate
        weight_total += weight
    worst_opponent = min(per_opponent, key=lambda o: per_opponent[o])
    return PairCoverage(
        members=tuple(members),
        per_opponent=per_opponent,
        best_member=best_member,
        radar=weighted / weight_total if weight_total else 0.0,
        plain_mean=sum(per_opponent.values()) / len(per_opponent),
        worst_opponent=worst_opponent,
        worst_rate=per_opponent[worst_opponent],
        complementarity=spread_sum / len(per_opponent),
    )


def rock_paper_scissors(candidates: Sequence[str],
                        matrix: Mapping[str, Fitness]) -> list[str]:
    """Head-to-head edges among the candidates, as readable lines."""
    lines = []
    for a, b in itertools.permutations(candidates, 2):
        cell = matrix[a].cell(b)
        if cell is None or cell.winrate <= 0.5:
            continue
        mark = "~" if cell.directional else " "
        low, high = cell.ci
        lines.append(f" {mark}{a} > {b}: {cell.winrate:.1%} "
                     f"[{low:.1%}-{high:.1%}]")
    return lines


def harvest(candidates: Sequence[str], field_decks: Sequence[str], games: int,
            seed: int, index: CardIndex, effects: EffectIndex,
            progress: bool = True) -> dict[str, Fitness]:
    """Every candidate against the field AND against each other."""
    cohort = [Entry(name=name, deck=name, is_candidate=True)
              for name in candidates]
    cohort += [Entry(name=name, deck=name) for name in field_decks
               if name not in set(candidates)]
    decks = load_decks(sorted({e.deck for e in cohort}), index)
    matrix: dict[str, Fitness] = {}
    for entry in cohort:
        if not entry.is_candidate:
            continue
        if progress:
            print(f"\n=== {entry.name} (module "
                  f"{module_for_deck(entry.deck).name!r}), {games} games/cell "
                  f"===", flush=True)
        matrix[entry.name] = evaluate(entry, cohort, decks, index, effects,
                                      games, seed, progress)
    return matrix


def format_matrix(candidates: Sequence[str],
                  matrix: Mapping[str, Fitness]) -> str:
    opponents: list[str] = []
    for name in candidates:
        for cell in matrix[name].cells:
            if cell.opponent not in opponents:
                opponents.append(cell.opponent)
    width = max(len(o) for o in opponents) + 1
    lines = ["winrate matrix (row = candidate, col = opponent); "
             "~ = Wilson CI straddles 50%",
             " " * (width + 2) + "".join(f"{n:>16s}" for n in candidates)]
    for opponent in opponents:
        cells = []
        for name in candidates:
            cell = matrix[name].cell(opponent)
            if cell is None:
                cells.append(f"{'-':>16s}")
            else:
                mark = "~" if cell.directional else " "
                cells.append(f"{mark}{cell.winrate:>14.1%} ")
        lines.append(f"{opponent:<{width}s}  " + "".join(cells))
    lines.append("")
    for name in candidates:
        fitness = matrix[name]
        worst = fitness.worst
        bench = fitness.cell(BENCHMARK_CELL)
        overall = pooled(fitness.cells)
        lines.append(
            f"  {name:<14s} pooled {overall} | radar {fitness.radar:.1%} | "
            f"worst {worst.opponent if worst else 'n/a'} "
            f"{worst.winrate if worst else 0:.1%} | vs-{BENCHMARK_CELL} "
            f"{bench.winrate if bench else 0:.1%}"
            + ("  (inflation-prone cell)" if bench else ""))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", nargs="+",
                        default=list(DEFAULT_CANDIDATES))
    parser.add_argument("--field", nargs="+", default=list(DEFAULT_FIELD))
    parser.add_argument("--games", type=int, default=120,
                        help="games per cell (>=100 for usable Wilson CIs)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=2,
                        help="portfolio size to rank (2 = pairs of finals)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    index, effects = CardIndex(), EffectIndex()
    t0 = time.perf_counter()
    matrix = harvest(args.candidates, args.field, args.games, args.seed,
                     index, effects)
    elapsed = time.perf_counter() - t0

    print("\n" + "=" * 70)
    print(format_matrix(args.candidates, matrix))

    print("\n" + "=" * 70)
    print("ROCK-PAPER-SCISSORS among the candidates (head to head):")
    for line in rock_paper_scissors(args.candidates, matrix) or ["  (none)"]:
        print(line)

    print("\n" + "=" * 70)
    print(f"PORTFOLIO COVERAGE (size {args.size}) — value per opponent is the "
          f"BEST member's,\nbecause a portfolio brings its better matchup:")
    coverages = [coverage(list(combo), matrix)
                 for combo in itertools.combinations(args.candidates,
                                                     args.size)]
    coverages.sort(key=lambda c: -c.radar)
    for cov in coverages:
        print("  " + cov.format())

    print("\n  singles, for reference:")
    singles = [coverage([name], matrix) for name in args.candidates]
    singles.sort(key=lambda c: -c.radar)
    for cov in singles:
        print("  " + cov.format())

    if coverages:
        best = coverages[0]
        print(f"\n  best pair by radar: {' + '.join(best.members)}")
        print(f"  its unpatched hole: {best.worst_opponent} "
              f"{best.worst_rate:.1%}")
        print("  who covers what:")
        for opponent in sorted(best.per_opponent,
                               key=lambda o: best.per_opponent[o]):
            print(f"    {opponent:<14s} {best.per_opponent[opponent]:6.1%} "
                  f"via {best.best_member[opponent]}")

    payload = {
        "games_per_cell": args.games, "seed": args.seed,
        "wall_seconds": elapsed,
        "candidates": {name: fitness.to_dict()
                       for name, fitness in matrix.items()},
        "portfolios": [c.to_dict() for c in coverages],
        "singles": [c.to_dict() for c in singles],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out} ({elapsed:.0f}s)")
    print("\nNOTE: this FEEDS the portfolio decision, it does not make it. "
          "Offline fitness is not the ladder.")


if __name__ == "__main__":
    main()
