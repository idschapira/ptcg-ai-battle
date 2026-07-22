"""League fitness: what a (deck, theta) candidate is WORTH against the cohort.

Fitness here is deliberately not "winrate vs the current best". A
co-evolutionary loop that optimizes a single number against a single
opponent collapses onto that opponent; the league measures a SPREAD and
reports the shape of it, so Phase 2's selection pressure can be aimed at
the worst cell rather than the mean.

Per candidate the harness reports:

  mean          unweighted mean winrate across the cohort
  radar         prize-weighted mean (`RADAR_WEIGHTS`), so cells that
                actually appear on the ladder count more than curiosities
  worst cell    the matchup that would sink us, with its Wilson interval
  vs Alakazam   ALWAYS reported separately, because it is the cell the
                gauntlet historically inflates (~30pp against a competent
                pilot — see the meta-decks note in CLAUDE.md's history).
                A candidate is not evaluated until this cell is printed.

NOBODY LEAVES. Every candidate is also an OPPONENT for every other
candidate: the cohort is candidates + fixed field, which is what keeps
the population from drifting into a private metagame.

Statistics: Wilson 95% intervals on decided games (reusing
`environment_wrapper.ab_test.wilson_interval`, never reimplemented). A
cell whose interval straddles 50% is DIRECTIONAL and is printed with a
`~` marker — with the game counts a co-evolution loop can afford, most
cells will be directional, and pretending otherwise is how you evolve
toward noise.

Every pairing plays both seats (`run_pair` from the gauntlet alternates
them), so seat advantage cancels.

OFFLINE/dev only — this never runs in the submission.

Run from the repo root:
    python -m src.league.fitness --games 40
    python -m src.league.fitness --games 60 --candidates grimmsnarl crustle
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Final, Iterable, Mapping, Sequence

from ..agent_heuristics.heuristic_agent import HeuristicAgent
from ..deckbuilding.gauntlet import DECKS_DIR, PairResult, run_pair
from ..deckbuilding.legality import read_deck_ids, validate_deck
from ..environment_wrapper.ab_test import wilson_interval
from ..environment_wrapper.selfplay import Agent
from ..ingestion.build_card_model import REPO_ROOT
from ..ingestion.build_effect_model import EffectIndex
from ..ingestion.card_index import CardIndex
from .modules import module_for_deck
from .parametric_agent import ParametricHeuristicAgent
from .theta import Theta

#: Candidate decks the league evolves pilots FOR (Phase 1 ships two of
#: the three; abomasnow joins in Phase 2 per the brief).
#: crustle_e10 is the SHIPPED deck — the incumbent champion is a
#: candidate like any other, so the league can measure it honestly.
DEFAULT_CANDIDATES: Final[tuple[str, ...]] = ("grimmsnarl", "crustle_e10")

#: Fixed opposition. These are also played by their own module when one
#: exists, so the cohort gets no free wins against a lobotomized foil.
DEFAULT_FIELD: Final[tuple[str, ...]] = (
    "alakazam", "spidops", "raging_bolt", "terastal_box", "mega_lucario",
    "dragapult", "starmie", "clefairy", "iono", "abomasnow",
)

#: Cells that decide ladder placement weigh more in the radar score.
#: Alakazam leads because it is both common and our historical hole.
RADAR_WEIGHTS: Final[dict[str, float]] = {
    "alakazam": 2.0,
    "grimmsnarl": 1.5,
    "spidops": 1.5,
    "dragapult": 1.25,
    "raging_bolt": 1.0,
    "terastal_box": 1.0,
    "mega_lucario": 1.0,
    "crustle": 1.0,
    "crustle_e10": 1.0,
    "abomasnow": 1.0,
}

#: The cell we refuse to let a report omit.
BENCHMARK_CELL: Final[str] = "alakazam"

DEFAULT_OUT: Final[Path] = REPO_ROOT / "data" / "league" / "fitness.json"


# ---------------------------------------------------------------------- #
# Deck resolution
# ---------------------------------------------------------------------- #

def _deck_paths() -> dict[str, Path]:
    """Every csv under data/decks/, keyed by its stem minus the prefix.

    Wider than gauntlet.discover_decks (which only strips seed_/
    placeholder_) because the league also ranges over meta_/candidate_
    decks, and a candidate must be addressable by its short name."""
    out: dict[str, Path] = {}
    for path in sorted(DECKS_DIR.glob("*.csv")):
        name = path.stem
        for prefix in ("seed_", "meta_", "candidate_", "placeholder_"):
            name = name.removeprefix(prefix)
        out.setdefault(name, path)
    return out


def load_decks(names: Iterable[str], index: CardIndex) -> dict[str, list[int]]:
    """Read + LEGALITY-VALIDATE the named decks; unknown names abort."""
    paths = _deck_paths()
    decks: dict[str, list[int]] = {}
    for name in names:
        path = paths.get(name)
        if path is None:
            raise SystemExit(f"unknown deck {name!r} "
                             f"(available: {sorted(paths)})")
        ids = read_deck_ids(path)
        report = validate_deck(ids, index)
        if not report.ok:
            for error in report.errors:
                print(f"  - {error}")
            raise SystemExit(f"deck {name!r} ({path}) is ILLEGAL — aborting")
        decks[name] = ids
    return decks


# ---------------------------------------------------------------------- #
# Cohort entries
# ---------------------------------------------------------------------- #

@dataclass(frozen=True)
class Entry:
    """One (deck, pilot) seat in the cohort — candidate or opposition."""

    name: str
    deck: str
    theta: Theta | None = None
    #: candidates are the things being SCORED; everyone plays as opponent
    is_candidate: bool = False

    def label(self) -> str:
        kind = "cand" if self.is_candidate else "field"
        return f"{self.name} [{kind}]"


def make_factory(entry: Entry, index: CardIndex,
                 effects: EffectIndex) -> Callable[[int], Agent]:
    """Agent factory for an entry: its deck module, its theta.

    A deck with no module gets the base DeckModule, i.e. plain generic
    heuristic play — the honest baseline, not a crippled one."""
    module = module_for_deck(entry.deck)
    theta = entry.theta

    def build(seed: int) -> Agent:
        return ParametricHeuristicAgent(module=module, theta=theta, seed=seed,
                                        index=index, effects=effects)

    return build


# ---------------------------------------------------------------------- #
# Results
# ---------------------------------------------------------------------- #

@dataclass(frozen=True)
class Cell:
    """One candidate-vs-opponent matchup with its Wilson interval."""

    opponent: str
    wins: int
    losses: int
    draws: int
    exceptions: int

    @property
    def decided(self) -> int:
        return self.wins + self.losses

    @property
    def winrate(self) -> float:
        return self.wins / self.decided if self.decided else 0.5

    @property
    def ci(self) -> tuple[float, float]:
        return wilson_interval(self.wins, self.decided)

    @property
    def directional(self) -> bool:
        """True when the interval straddles 50%: a hint, not a result."""
        low, high = self.ci
        return low <= 0.5 <= high

    def format(self) -> str:
        low, high = self.ci
        mark = "~" if self.directional else " "
        return (f"{mark}{self.opponent:<14s} {self.winrate:6.1%} "
                f"[{low:5.1%}-{high:5.1%}]  {self.wins}-{self.losses}"
                f"{f'-{self.draws}d' if self.draws else ''}"
                f"{f'  EXC {self.exceptions}' if self.exceptions else ''}")

    @classmethod
    def from_dict(cls, data: Mapping) -> "Cell":
        return cls(opponent=str(data.get("opponent", "?")),
                   wins=int(data.get("wins", 0)),
                   losses=int(data.get("losses", 0)),
                   draws=int(data.get("draws", 0)),
                   exceptions=int(data.get("exceptions", 0)))

    def to_dict(self) -> dict:
        low, high = self.ci
        return {"opponent": self.opponent, "wins": self.wins,
                "losses": self.losses, "draws": self.draws,
                "winrate": self.winrate, "ci_low": low, "ci_high": high,
                "directional": self.directional,
                "exceptions": self.exceptions}


@dataclass
class Fitness:
    """A candidate's spread against the whole cohort."""

    candidate: str
    deck: str
    cells: list[Cell] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return (sum(c.winrate for c in self.cells) / len(self.cells)
                if self.cells else 0.0)

    @property
    def radar(self) -> float:
        """Prize-weighted mean: ladder-relevant cells count more."""
        total = weight_sum = 0.0
        for cell in self.cells:
            weight = RADAR_WEIGHTS.get(cell.opponent, 1.0)
            total += weight * cell.winrate
            weight_sum += weight
        return total / weight_sum if weight_sum else 0.0

    @property
    def worst(self) -> Cell | None:
        return min(self.cells, key=lambda c: c.winrate) if self.cells else None

    @property
    def exceptions(self) -> int:
        return sum(c.exceptions for c in self.cells)

    def cell(self, opponent: str) -> Cell | None:
        for cell in self.cells:
            if cell.opponent == opponent:
                return cell
        return None

    def score(self) -> float:
        """The single number Phase 2 selects on.

        Mean pulls the whole spread up; the worst cell is what actually
        loses tournaments, so it gets real weight. Deliberately NOT pure
        mean — a candidate that farms four decks and auto-loses the fifth
        is not a champion."""
        worst = self.worst
        floor = worst.winrate if worst is not None else 0.0
        return 0.5 * self.radar + 0.3 * self.mean + 0.2 * floor

    def report(self) -> str:
        lines = [f"--- {self.candidate}  (deck: {self.deck}) ---"]
        for cell in sorted(self.cells, key=lambda c: c.winrate):
            lines.append("  " + cell.format())
        worst = self.worst
        bench = self.cell(BENCHMARK_CELL)
        lines.append(f"  mean {self.mean:.1%} | radar {self.radar:.1%} "
                     f"| score {self.score():.3f}")
        if worst is not None:
            low, high = worst.ci
            lines.append(f"  WORST CELL: {worst.opponent} {worst.winrate:.1%} "
                         f"[{low:.1%}-{high:.1%}]")
        # mandatory: the cell the gauntlet inflates
        if bench is not None:
            low, high = bench.ci
            lines.append(f"  vs {BENCHMARK_CELL.upper()} (inflation-prone): "
                         f"{bench.winrate:.1%} [{low:.1%}-{high:.1%}]"
                         + ("  ~directional" if bench.directional else ""))
        else:
            lines.append(f"  vs {BENCHMARK_CELL.upper()}: NOT MEASURED — "
                         f"this report is incomplete")
        lines.append(f"  exceptions: {self.exceptions}  (must be 0)")
        return "\n".join(lines)

    @classmethod
    def from_dict(cls, data: Mapping) -> "Fitness":
        """Rebuild a report from its JSON — lets the Hall of Fame archive
        a run without re-simulating it."""
        fitness = cls(candidate=str(data.get("candidate", "?")),
                      deck=str(data.get("deck", "?")))
        fitness.cells = [Cell.from_dict(c) for c in (data.get("cells") or [])]
        return fitness

    def to_dict(self) -> dict:
        worst = self.worst
        bench = self.cell(BENCHMARK_CELL)
        return {
            "candidate": self.candidate,
            "deck": self.deck,
            "mean": self.mean,
            "radar": self.radar,
            "score": self.score(),
            "worst_cell": worst.to_dict() if worst is not None else None,
            "benchmark_cell": bench.to_dict() if bench is not None else None,
            "exceptions": self.exceptions,
            "cells": [c.to_dict() for c in self.cells],
        }


# ---------------------------------------------------------------------- #
# The harness
# ---------------------------------------------------------------------- #

def _cell_from_pair(opponent: str, res: PairResult) -> Cell:
    return Cell(opponent=opponent, wins=res.a_wins, losses=res.b_wins,
                draws=res.draws, exceptions=len(res.errors))


def evaluate(candidate: Entry, cohort: Sequence[Entry],
             decks: Mapping[str, list[int]], index: CardIndex,
             effects: EffectIndex, games: int, seed: int,
             progress: bool = False) -> Fitness:
    """One candidate against every OTHER cohort entry, both seats."""
    fitness = Fitness(candidate=candidate.name, deck=candidate.deck)
    make_a = make_factory(candidate, index, effects)
    for other in cohort:
        if other.name == candidate.name:
            continue
        res = run_pair(make_a, make_factory(other, index, effects),
                       decks[candidate.deck], decks[other.deck], games, seed)
        cell = _cell_from_pair(other.name, res)
        fitness.cells.append(cell)
        if progress:
            print("  " + cell.format(), flush=True)
        for error in res.errors[:3]:
            print(f"    {error}")
    return fitness


def build_cohort(candidates: Sequence[str], field_decks: Sequence[str],
                 thetas: Mapping[str, Theta] | None = None) -> list[Entry]:
    """Candidates + fixed field. NOBODY LEAVES: every candidate is also
    an opponent for the others, so the population cannot drift into a
    private metagame."""
    thetas = thetas or {}
    entries = [Entry(name=name, deck=name, theta=thetas.get(name),
                     is_candidate=True) for name in candidates]
    entries += [Entry(name=name, deck=name, theta=thetas.get(name))
                for name in field_decks if name not in set(candidates)]
    return entries


def run_league(candidates: Sequence[str], field_decks: Sequence[str],
               games: int, seed: int, index: CardIndex, effects: EffectIndex,
               thetas: Mapping[str, Theta] | None = None,
               progress: bool = True) -> list[Fitness]:
    cohort = build_cohort(candidates, field_decks, thetas)
    decks = load_decks(sorted({e.deck for e in cohort}), index)
    results: list[Fitness] = []
    for entry in cohort:
        if not entry.is_candidate:
            continue
        if progress:
            module = module_for_deck(entry.deck)
            print(f"\n=== {entry.label()} — module {module.name!r}, "
                  f"{games} games/cell ===", flush=True)
        results.append(evaluate(entry, cohort, decks, index, effects, games,
                                seed, progress))
    return results


# ---------------------------------------------------------------------- #
# Baselines (what a candidate must BEAT to be worth a module)
# ---------------------------------------------------------------------- #

def evaluate_generic_baseline(deck: str, cohort: Sequence[Entry],
                              decks: Mapping[str, list[int]], index: CardIndex,
                              effects: EffectIndex, games: int,
                              seed: int) -> Fitness:
    """Same deck, same cohort, but piloted by the plain HeuristicAgent —
    the honest control for "did the module actually add anything?"."""
    fitness = Fitness(candidate=f"{deck}(generic)", deck=deck)
    def make_a(s: int) -> Agent:
        return HeuristicAgent(seed=s, index=index, effects=effects)
    for other in cohort:
        if other.deck == deck:
            continue
        res = run_pair(make_a, make_factory(other, index, effects),
                       decks[deck], decks[other.deck], games, seed)
        fitness.cells.append(_cell_from_pair(other.name, res))
    return fitness


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=40,
                        help="games per cell (both seats; >=40 for Wilson)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--candidates", nargs="+", default=list(DEFAULT_CANDIDATES))
    parser.add_argument("--field", nargs="+", default=list(DEFAULT_FIELD))
    parser.add_argument("--baseline", action="store_true",
                        help="also measure each candidate deck under the "
                             "plain generic pilot (module-lift control)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="write the report as JSON here")
    args = parser.parse_args()

    index = CardIndex()
    effects = EffectIndex()
    t0 = time.perf_counter()
    results = run_league(args.candidates, args.field, args.games, args.seed,
                         index, effects)
    elapsed = time.perf_counter() - t0

    print("\n" + "=" * 62)
    print("LEAGUE FITNESS (Wilson 95% CI; ~ = interval straddles 50%)")
    print("=" * 62)
    for fitness in sorted(results, key=lambda f: -f.score()):
        print(fitness.report())

    payload = {"games_per_cell": args.games, "seed": args.seed,
               "wall_seconds": elapsed,
               "candidates": [f.to_dict() for f in results]}

    if args.baseline:
        cohort = build_cohort(args.candidates, args.field)
        decks = load_decks(sorted({e.deck for e in cohort}), index)
        print("\n--- module lift vs the generic pilot on the same deck ---")
        baselines = []
        for name in args.candidates:
            base = evaluate_generic_baseline(name, cohort, decks, index,
                                             effects, args.games, args.seed)
            module = next(f for f in results if f.candidate == name)
            print(f"  {name:14s} generic mean {base.mean:6.1%} -> "
                  f"module mean {module.mean:6.1%}  "
                  f"({module.mean - base.mean:+.1%})")
            baselines.append(base.to_dict())
        payload["baselines"] = baselines

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}  ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
