"""Hall of Fame: the persisted, resumable memory of the league.

Phase 1 scaffold — it STORES and RANKS champions, it does not select or
mutate them. Phase 2 plugs its loop in through three methods and nothing
else:

    hof.add(champion)          after evaluating a mutated genome
    hof.opponents(deck=...)    the archive to play the next generation against
    hof.best(deck=...)         the incumbent to beat

Why an archive at all: co-evolution without one CYCLES. A population
that only ever plays the current generation rediscovers counters it
already beat and forgets why. Keeping beaten champions as live opponents
(the same "nobody leaves" rule the fitness cohort follows) is what turns
a cycle into a ratchet.

Persistence is one JSON file, written atomically (temp + replace) so an
interrupted run cannot corrupt the archive, and re-readable by a later
process — the loop is resumable by construction. Genomes round-trip by
NAME through `ThetaSchema.from_dict`, so adding a knob to a module does
not invalidate stored champions.

The file lives under data/league/ and is gitignored: it is a dev
artifact, like the parquet cache and the replay datasets.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Iterable, Sequence

from ..ingestion.build_card_model import REPO_ROOT
from .modules import module_for_deck
from .theta import Theta

DEFAULT_PATH: Final[Path] = REPO_ROOT / "data" / "league" / "hall_of_fame.json"
FORMAT_VERSION: Final[int] = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Champion:
    """One archived (deck, theta) with the fitness that earned its place."""

    deck: str
    module: str
    theta: Theta
    score: float
    mean: float
    radar: float
    worst_cell: str | None = None
    worst_winrate: float | None = None
    benchmark_winrate: float | None = None
    generation: int = 0
    created: str = field(default_factory=_utc_now)
    note: str = ""

    @property
    def key(self) -> str:
        return f"{self.deck}#g{self.generation}"

    def summary(self) -> str:
        worst = (f"worst {self.worst_cell} {self.worst_winrate:.1%}"
                 if self.worst_cell is not None
                 and self.worst_winrate is not None else "worst n/a")
        bench = (f"vs-alakazam {self.benchmark_winrate:.1%}"
                 if self.benchmark_winrate is not None else "vs-alakazam n/a")
        moved = len(self.theta.diff_from_defaults())
        return (f"{self.key:<24s} score {self.score:.3f} | mean {self.mean:.1%}"
                f" | radar {self.radar:.1%} | {worst} | {bench}"
                f" | {moved} knob(s) moved")

    def to_dict(self) -> dict[str, Any]:
        return {"deck": self.deck, "module": self.module,
                "theta": self.theta.to_dict(), "score": self.score,
                "mean": self.mean, "radar": self.radar,
                "worst_cell": self.worst_cell,
                "worst_winrate": self.worst_winrate,
                "benchmark_winrate": self.benchmark_winrate,
                "generation": self.generation, "created": self.created,
                "note": self.note}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Champion":
        deck = str(data.get("deck", ""))
        module = module_for_deck(deck)
        # by NAME, so a schema that gained a knob still loads
        theta = module.schema.from_dict(data.get("theta"))
        return cls(deck=deck, module=str(data.get("module", module.name)),
                   theta=theta, score=float(data.get("score", 0.0)),
                   mean=float(data.get("mean", 0.0)),
                   radar=float(data.get("radar", 0.0)),
                   worst_cell=data.get("worst_cell"),
                   worst_winrate=data.get("worst_winrate"),
                   benchmark_winrate=data.get("benchmark_winrate"),
                   generation=int(data.get("generation", 0)),
                   created=str(data.get("created", _utc_now())),
                   note=str(data.get("note", "")))

    @classmethod
    def from_fitness(cls, fitness, theta: Theta, generation: int = 0,
                     note: str = "") -> "Champion":
        """Build straight from a `fitness.Fitness` result."""
        worst = fitness.worst
        from .fitness import BENCHMARK_CELL
        bench = fitness.cell(BENCHMARK_CELL)
        return cls(deck=fitness.deck, module=module_for_deck(fitness.deck).name,
                   theta=theta, score=fitness.score(), mean=fitness.mean,
                   radar=fitness.radar,
                   worst_cell=worst.opponent if worst is not None else None,
                   worst_winrate=worst.winrate if worst is not None else None,
                   benchmark_winrate=bench.winrate if bench is not None else None,
                   generation=generation, note=note)


class HallOfFame:
    """Persisted, resumable champion archive. No selection logic yet."""

    __slots__ = ("path", "_champions", "capacity_per_deck")

    def __init__(self, path: Path | None = None,
                 capacity_per_deck: int = 8) -> None:
        self.path = Path(path) if path is not None else DEFAULT_PATH
        self.capacity_per_deck = capacity_per_deck
        self._champions: list[Champion] = []

    # ---- collection ---- #

    def __len__(self) -> int:
        return len(self._champions)

    def __iter__(self):
        return iter(self._champions)

    @property
    def champions(self) -> tuple[Champion, ...]:
        return tuple(self._champions)

    def decks(self) -> tuple[str, ...]:
        return tuple(sorted({c.deck for c in self._champions}))

    def for_deck(self, deck: str) -> list[Champion]:
        return sorted((c for c in self._champions if c.deck == deck),
                      key=lambda c: -c.score)

    def best(self, deck: str | None = None) -> Champion | None:
        """The incumbent — overall, or for one deck."""
        pool = self._champions if deck is None else self.for_deck(deck)
        return max(pool, key=lambda c: c.score) if pool else None

    def opponents(self, deck: str | None = None,
                  limit: int | None = None) -> list[Champion]:
        """The archive to play the next generation against.

        Defaults to the best champion PER DECK rather than the globally
        best N: an archive of eight near-identical Crustles teaches a
        challenger nothing. Pass `deck` for that deck's own lineage."""
        if deck is not None:
            found = self.for_deck(deck)
            return found[:limit] if limit else found
        best = [c for c in (self.best(d) for d in self.decks())
                if c is not None]
        best.sort(key=lambda c: -c.score)
        return best[:limit] if limit else best

    def next_generation(self, deck: str) -> int:
        existing = self.for_deck(deck)
        return 1 + max((c.generation for c in existing), default=-1)

    # ---- mutation of the ARCHIVE (not of genomes) ---- #

    def add(self, champion: Champion) -> bool:
        """Archive a champion; True when it was kept.

        Per-deck capacity keeps the archive diverse and the next
        generation's evaluation affordable: once full, a newcomer must
        beat the weakest entry for that deck."""
        same_deck = self.for_deck(champion.deck)
        if len(same_deck) < self.capacity_per_deck:
            self._champions.append(champion)
            return True
        weakest = min(same_deck, key=lambda c: c.score)
        if champion.score <= weakest.score:
            return False
        self._champions.remove(weakest)
        self._champions.append(champion)
        return True

    def extend(self, champions: Iterable[Champion]) -> int:
        return sum(1 for c in champions if self.add(c))

    # ---- persistence (atomic, resumable) ---- #

    def save(self, path: Path | None = None) -> Path:
        target = Path(path) if path is not None else self.path
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format": FORMAT_VERSION, "updated": _utc_now(),
                   "capacity_per_deck": self.capacity_per_deck,
                   "champions": [c.to_dict() for c in self._champions]}
        # temp + replace: an interrupted run cannot corrupt the archive
        handle, tmp_name = tempfile.mkstemp(dir=str(target.parent),
                                            suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2)
            os.replace(tmp_name, target)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
        return target

    @classmethod
    def load(cls, path: Path | None = None,
             capacity_per_deck: int = 8) -> "HallOfFame":
        """Read an archive; a missing or unreadable file yields an EMPTY
        one, so a first run and a resumed run take the same code path."""
        hof = cls(path=path, capacity_per_deck=capacity_per_deck)
        try:
            raw = hof.path.read_text(encoding="utf-8")
        except (OSError, ValueError):
            return hof
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return hof
        if not isinstance(data, dict):
            return hof
        hof.capacity_per_deck = int(data.get("capacity_per_deck",
                                             capacity_per_deck))
        for entry in data.get("champions") or []:
            if isinstance(entry, dict):
                try:
                    hof._champions.append(Champion.from_dict(entry))
                except Exception:
                    continue  # one bad row never sinks the archive
        return hof

    # ---- reporting ---- #

    def report(self) -> str:
        if not self._champions:
            return "Hall of Fame: empty"
        lines = [f"Hall of Fame ({len(self._champions)} champions, "
                 f"{len(self.decks())} decks) — {self.path}"]
        for deck in self.decks():
            lines.append(f"  [{deck}]")
            for champion in self.for_deck(deck):
                lines.append("    " + champion.summary())
        return "\n".join(lines)


def record_results(results: Sequence, thetas: dict[str, Theta],
                   path: Path | None = None, note: str = "") -> HallOfFame:
    """Convenience for Phase 1: archive a `run_league` result set.

    Phase 2 will call `add` directly from inside its loop; this exists so
    a manual fitness run can seed the archive today."""
    hof = HallOfFame.load(path)
    for fitness in results:
        theta = thetas.get(fitness.deck)
        if theta is None:
            theta = module_for_deck(fitness.deck).default_theta()
        hof.add(Champion.from_fitness(
            fitness, theta, generation=hof.next_generation(fitness.deck),
            note=note))
    hof.save()
    return hof


def main() -> None:
    """Seed / inspect the archive from a fitness report.

    Phase 2 will drive `add` from inside its loop; this CLI is how a
    manual `python -m src.league.fitness` run gets archived today.

        python -m src.league.hall_of_fame --seed data/league/fitness.json
        python -m src.league.hall_of_fame            # just show it
    """
    import argparse

    from .fitness import DEFAULT_OUT, Fitness

    parser = argparse.ArgumentParser(description="Hall of Fame (Phase 1)")
    parser.add_argument("--path", type=Path, default=None,
                        help=f"archive file (default {DEFAULT_PATH})")
    parser.add_argument("--seed", type=Path, nargs="?", const=DEFAULT_OUT,
                        default=None, metavar="FITNESS_JSON",
                        help="archive the candidates of a fitness report")
    parser.add_argument("--note", type=str, default="")
    args = parser.parse_args()

    hof = HallOfFame.load(args.path)
    if args.seed is not None:
        payload = json.loads(args.seed.read_text(encoding="utf-8"))
        for entry in payload.get("candidates") or []:
            fitness = Fitness.from_dict(entry)
            theta = module_for_deck(fitness.deck).default_theta()
            champion = Champion.from_fitness(
                fitness, theta,
                generation=hof.next_generation(fitness.deck),
                note=args.note or f"seeded from {args.seed.name}")
            kept = hof.add(champion)
            print(f"{'archived' if kept else 'rejected'}: "
                  f"{champion.summary()}")
        hof.save()
    print()
    print(hof.report())


__all__ = ["Champion", "HallOfFame", "record_results", "DEFAULT_PATH"]


if __name__ == "__main__":
    main()
