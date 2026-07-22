"""The co-evolutionary loop: a population of theta per candidate deck.

Phase 2. Each generation, per deck:

    evaluate (spread vs the GROUNDED opponent pool, Wilson CIs)
      -> select the elite
      -> mutate to refill the population (inside the schema's bands)
      -> promote the generation's best into the Hall of Fame
      -> checkpoint

GROUNDING is the invariant that keeps this honest, and it has three
independent parts:

1. REAL ANCHORS, always. The fixed field decks (`fitness.DEFAULT_FIELD`)
   are in every evaluation — they are the actual metagame, not something
   the population invented. `--anchor-agent` pins the Alakazam cell to a
   stronger real pilot when the compute budget allows; otherwise that
   cell is a module/BC-clone and is MARKED DIRECTIONAL in the report,
   because we know it inflates (~30pp historically).
2. THE ZOO, always. Every field deck stays in the pool even when it is a
   bad matchup for nobody — a genome that quietly breaks on an odd deck
   must pay for it in fitness, which is what buys robustness.
3. THE HALL OF FAME. Past champions are sampled in as opponents, so a
   genome cannot win by countering only the current generation. Without
   this, co-evolution CYCLES: the population rediscovers counters it
   already beat and forgets why. With it, the archive is a ratchet.

Also: NOBODY optimizes against a single opponent. `Fitness.score`
weights the WORST cell explicitly, so a genome that farms four decks and
auto-loses the fifth cannot win selection.

NOISE. A generation cannot afford Wilson-tight games on every genome, so
evaluation is RACED: all genomes get a cheap screen against a subset,
only the survivors get the full cohort. Screen results are never
reported as findings — the report carries the full-evaluation CIs, and
cells whose interval straddles 50% stay marked `~`.

RESILIENCE. Checkpoint after every generation (atomic write), resumable
from disk, and the collector's pause protocol is reused: a STOP file
ends the run cleanly, a PAUSE file or the nightly job window blocks.

This NEVER promotes anything to production. A generation's champion is a
CANDIDATE; the ladder is the judge (see the grounding gate in the
Phase 2 report).

Run from the repo root:
    python -m src.league.evolve --decks grimmsnarl abomasnow --generations 6
    python -m src.league.evolve --resume            # picks up the checkpoint
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Sequence

from ..ingestion.build_card_model import REPO_ROOT
from ..ingestion.build_effect_model import EffectIndex
from ..ingestion.card_index import CardIndex
from .fitness import (BENCHMARK_CELL, DEFAULT_FIELD, Entry, Fitness,
                      evaluate, load_decks)
from .hall_of_fame import Champion, HallOfFame
from .modules import module_for_deck
from .theta import ParamSpec, Theta

LEAGUE_DIR: Final[Path] = REPO_ROOT / "data" / "league"
DEFAULT_STATE: Final[Path] = LEAGUE_DIR / "evolve_state.json"

#: Same window the replay collector yields in (nightly jobs at ~21:00).
PAUSE_WINDOW: Final[tuple[datetime.time, datetime.time]] = (
    datetime.time(20, 58), datetime.time(21, 32))

#: Decks the loop evolves pilots for. Crustle is opt-in: it is the SHIP,
#: and its module is bit-equivalent to the shipped agent, so touching it
#: is a portfolio decision rather than a free experiment.
DEFAULT_DECKS: Final[tuple[str, ...]] = ("grimmsnarl", "abomasnow")

#: Cells used for the cheap screen: the two that historically decide
#: things (the inflation-prone benchmark and the structural hole).
SCREEN_CELLS: Final[tuple[str, ...]] = ("alakazam", "starmie")


# ---------------------------------------------------------------------- #
# Mutation
# ---------------------------------------------------------------------- #

def mutate(theta: Theta, rng: random.Random, sigma_frac: float = 0.12,
           rate: float = 0.35) -> Theta:
    """One mutated genome.

    Continuous knobs get a gaussian step scaled to their OWN band
    (sigma_frac of the band width), so a 0-100 targeting weight and a
    0.0-2.0 multiplier move by comparable amounts. Integral knobs take a
    +/-1..2 step. `ThetaSchema` clipping does the rest: a mutant can
    never leave the legal band, which is how the structural invariants
    (attacking never outranks development, abilities never outrank
    attaching) survive evolution without any rule-level guard."""
    values = list(theta.to_vector())
    for i, spec in enumerate(theta.schema.specs):
        if rng.random() > rate:
            continue
        span = spec.high - spec.low
        if span <= 0:
            continue
        if spec.integral:
            values[i] = values[i] + rng.choice((-2, -1, 1, 2))
        else:
            values[i] = rng.gauss(values[i], sigma_frac * span)
    return theta.schema.from_vector(values)


def crossover(a: Theta, b: Theta, rng: random.Random) -> Theta:
    """Uniform crossover — recombines two elites knob by knob."""
    values = [a.to_vector()[i] if rng.random() < 0.5 else b.to_vector()[i]
              for i in range(len(a.schema))]
    return a.schema.from_vector(values)


# ---------------------------------------------------------------------- #
# Population state
# ---------------------------------------------------------------------- #

@dataclass
class Individual:
    theta: Theta
    score: float | None = None
    mean: float | None = None
    worst_cell: str | None = None
    worst: float | None = None
    benchmark: float | None = None
    origin: str = "seed"


@dataclass
class DeckState:
    """Everything the loop needs to resume one deck's evolution."""

    deck: str
    generation: int = 0
    population: list[Individual] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    #: The default theta, re-measured against EVERY generation's pool.
    #: The pool changes between generations (the Hall of Fame slice is
    #: sampled), so raw scores are NOT comparable across generations —
    #: only the same-pool delta best-minus-default is.
    reference: Individual | None = None

    def to_dict(self) -> dict:
        return {
            "deck": self.deck, "generation": self.generation,
            "population": [{"theta": i.theta.to_dict(), "score": i.score,
                            "mean": i.mean, "worst_cell": i.worst_cell,
                            "worst": i.worst, "benchmark": i.benchmark,
                            "origin": i.origin} for i in self.population],
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DeckState":
        deck = str(data.get("deck", ""))
        schema = module_for_deck(deck).schema
        state = cls(deck=deck, generation=int(data.get("generation", 0)),
                    history=list(data.get("history") or []))
        for row in data.get("population") or []:
            if not isinstance(row, dict):
                continue
            state.population.append(Individual(
                theta=schema.from_dict(row.get("theta")),
                score=row.get("score"), mean=row.get("mean"),
                worst_cell=row.get("worst_cell"), worst=row.get("worst"),
                benchmark=row.get("benchmark"),
                origin=str(row.get("origin", "seed"))))
        return state


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------- #
# Pause / stop protocol (same contract as the replay collector)
# ---------------------------------------------------------------------- #

def wait_if_paused(control_dir: Path, quiet: bool = False) -> bool:
    """True = a STOP was requested. Blocks on PAUSE and in the job window."""
    announced = False
    while True:
        if (control_dir / "STOP").exists():
            return True
        now = datetime.datetime.now().time()
        in_window = PAUSE_WINDOW[0] <= now <= PAUSE_WINDOW[1]
        paused = (control_dir / "PAUSE").exists()
        if not (in_window or paused):
            return False
        if not announced and not quiet:
            reason = "nightly job window" if in_window else "PAUSE file"
            print(f"[evolve] yielding ({reason})", flush=True)
            announced = True
        time.sleep(60 if in_window else 30)


# ---------------------------------------------------------------------- #
# The loop
# ---------------------------------------------------------------------- #

class CoEvolution:
    """Population-per-deck co-evolution against a grounded opponent pool."""

    def __init__(self, decks: Sequence[str], field_decks: Sequence[str],
                 population: int = 8, elite: int = 3, games: int = 24,
                 screen_games: int = 8, seed: int = 0,
                 hof_samples: int = 2, sigma_frac: float = 0.12,
                 state_path: Path = DEFAULT_STATE,
                 hof_path: Path | None = None) -> None:
        self.decks = list(decks)
        self.field = list(field_decks)
        self.population_size = population
        self.elite_size = max(1, min(elite, population))
        self.games = games
        self.screen_games = screen_games
        self.hof_samples = hof_samples
        self.sigma_frac = sigma_frac
        self.state_path = state_path
        self.rng = random.Random(seed)
        self.seed = seed
        self.index = CardIndex()
        self.effects = EffectIndex()
        self.hof = HallOfFame.load(hof_path)
        self.states: dict[str, DeckState] = {}
        self.decks_cache: dict[str, list[int]] = {}

    # ---- setup ---- #

    def _all_deck_names(self) -> list[str]:
        return sorted(set(self.decks) | set(self.field)
                      | {c.deck for c in self.hof})

    def prepare(self) -> None:
        self.decks_cache = load_decks(self._all_deck_names(), self.index)
        for deck in self.decks:
            if deck not in self.states:
                self.states[deck] = DeckState(deck=deck)
            state = self.states[deck]
            if not state.population:
                # gen 0 seeds from the VERSIONED default theta plus
                # mutants of it: the loop starts from known-good, not
                # from noise.
                base = module_for_deck(deck).default_theta()
                state.population = [Individual(theta=base, origin="default")]
                while len(state.population) < self.population_size:
                    state.population.append(Individual(
                        theta=mutate(base, self.rng, self.sigma_frac),
                        origin="seed-mutant"))

    # ---- opponent pool (the grounding) ---- #

    def opponent_pool(self, deck: str) -> list[Entry]:
        """Real anchors + the zoo + Hall of Fame samples.

        The first two are FIXED — they are the metagame and they never
        leave. The HoF slice is what prevents the population from
        overfitting to the current generation."""
        pool = [Entry(name=name, deck=name) for name in self.field
                if name != deck]
        # the other candidate decks are opponents too: nobody leaves
        pool += [Entry(name=other, deck=other) for other in self.decks
                 if other != deck and other not in self.field]
        archive = [c for c in self.hof.opponents() if c.deck in self.decks_cache]
        self.rng.shuffle(archive)
        for champion in archive[:self.hof_samples]:
            pool.append(Entry(name=f"hof:{champion.key}", deck=champion.deck,
                              theta=champion.theta))
        return pool

    # ---- evaluation (raced) ---- #

    def _evaluate(self, deck: str, theta: Theta, pool: Sequence[Entry],
                  games: int) -> Fitness:
        candidate = Entry(name=deck, deck=deck, theta=theta, is_candidate=True)
        return evaluate(candidate, list(pool) + [candidate], self.decks_cache,
                        self.index, self.effects, games, self.seed)

    def _screen(self, deck: str, individuals: Sequence[Individual],
                pool: Sequence[Entry]) -> list[Individual]:
        """Cheap first pass on SCREEN_CELLS; keep the better half.

        Racing, not a result: these numbers never leave this method."""
        keep = max(self.elite_size, len(individuals) // 2)
        if len(individuals) <= keep or self.screen_games <= 0:
            return list(individuals)
        # the versioned default is the incumbent: it is never raced out,
        # so a generation can always report "evolution did not beat it"
        protected = [i for i in individuals if i.origin == "default"]
        contenders = [i for i in individuals if i.origin != "default"]
        subset = [e for e in pool if e.name in SCREEN_CELLS] or list(pool[:2])
        scored: list[tuple[float, Individual]] = []
        for individual in contenders:
            fitness = self._evaluate(deck, individual.theta, subset,
                                     self.screen_games)
            scored.append((fitness.score(), individual))
        scored.sort(key=lambda pair: -pair[0])
        room = max(1, keep - len(protected))
        return protected + [individual for _, individual in scored[:room]]

    # ---- one generation ---- #

    def step(self, deck: str, control_dir: Path) -> dict | None:
        """One generation for one deck. None when a STOP was requested."""
        state = self.states[deck]
        pool = self.opponent_pool(deck)
        if wait_if_paused(control_dir):
            return None

        # The REFERENCE run: the versioned default theta, measured
        # against THIS generation's pool. Because the pool changes
        # between generations, only the same-pool delta is a fair read
        # of whether co-evolution actually moved the needle.
        reference = self._evaluate(deck, module_for_deck(deck).default_theta(),
                                   pool, self.games)
        reference_worst = reference.worst
        state.reference = Individual(
            theta=module_for_deck(deck).default_theta(),
            score=reference.score(), mean=reference.mean,
            worst_cell=reference_worst.opponent if reference_worst else None,
            worst=reference_worst.winrate if reference_worst else None,
            origin="default")

        survivors = self._screen(deck, state.population, pool)
        evaluated: list[Individual] = []
        best_fitness: Fitness | None = None
        for individual in survivors:
            if wait_if_paused(control_dir):
                return None
            fitness = self._evaluate(deck, individual.theta, pool, self.games)
            worst = fitness.worst
            bench = fitness.cell(BENCHMARK_CELL)
            individual.score = fitness.score()
            individual.mean = fitness.mean
            individual.worst_cell = worst.opponent if worst else None
            individual.worst = worst.winrate if worst else None
            individual.benchmark = bench.winrate if bench else None
            evaluated.append(individual)
            if best_fitness is None or fitness.score() > best_fitness.score():
                best_fitness = fitness
            if fitness.exceptions:
                print(f"  !! {fitness.exceptions} exceptions on {deck}",
                      flush=True)

        evaluated.sort(key=lambda i: -(i.score or 0.0))
        elite = evaluated[:self.elite_size]

        # promote the generation's champion into the archive
        if best_fitness is not None and elite:
            champion = Champion.from_fitness(
                best_fitness, elite[0].theta, generation=state.generation,
                note=f"co-evolution gen {state.generation}")
            self.hof.add(champion)
            self.hof.save()

        record = {
            "generation": state.generation,
            "reference_score": state.reference.score if state.reference else None,
            "reference_mean": state.reference.mean if state.reference else None,
            "delta_vs_default": (
                elite[0].score - state.reference.score
                if elite and elite[0].score is not None
                and state.reference is not None
                and state.reference.score is not None else None),
            "best_score": elite[0].score if elite else None,
            "best_mean": elite[0].mean if elite else None,
            "best_worst_cell": elite[0].worst_cell if elite else None,
            "best_worst": elite[0].worst if elite else None,
            "best_benchmark": elite[0].benchmark if elite else None,
            "elite_origins": [i.origin for i in elite],
            "evaluated": len(evaluated),
            "opponents": [e.name for e in pool],
            "games_per_cell": self.games,
        }
        state.history.append(record)

        # refill: elites survive, the rest are mutants/crossovers of them
        next_population = list(elite)
        while len(next_population) < self.population_size:
            if len(elite) >= 2 and self.rng.random() < 0.3:
                parent_a, parent_b = self.rng.sample(elite, 2)
                child = crossover(parent_a.theta, parent_b.theta, self.rng)
                origin = "crossover"
            else:
                parent = self.rng.choice(elite)
                child = mutate(parent.theta, self.rng, self.sigma_frac)
                origin = "mutant"
            next_population.append(Individual(theta=child, origin=origin))
        state.population = next_population
        state.generation += 1
        return record

    # ---- driver ---- #

    def run(self, generations: int, control_dir: Path) -> None:
        self.prepare()
        for _ in range(generations):
            for deck in self.decks:
                state = self.states[deck]
                print(f"\n[gen {state.generation}] {deck} "
                      f"(pop {len(state.population)}, {self.games} games/cell)",
                      flush=True)
                t0 = time.perf_counter()
                record = self.step(deck, control_dir)
                if record is None:
                    print("[evolve] STOP requested — checkpointing and exiting",
                          flush=True)
                    self.save()
                    return
                if record["best_score"] is None:
                    print("  (no result)", flush=True)
                else:
                    delta = record.get("delta_vs_default")
                    print(f"  best {record['best_score']:.3f} vs default "
                          f"{record['reference_score']:.3f} on the SAME pool"
                          f"  -> {delta:+.3f}" if delta is not None else "",
                          flush=True)
                    print(f"  mean {record['best_mean']:.1%} | worst "
                          f"{record['best_worst_cell']} "
                          f"{record['best_worst']:.1%} | vs-{BENCHMARK_CELL} "
                          f"{record['best_benchmark']:.1%}", flush=True)
                print(f"  elite origins: {record['elite_origins']} "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)
                self.save()

    def baseline_score(self, deck: str) -> float | None:
        """The default theta's score on the LATEST generation's pool."""
        state = self.states.get(deck)
        if state is None or state.reference is None:
            return None
        return state.reference.score

    # ---- persistence ---- #

    def save(self) -> None:
        _atomic_write(self.state_path, {
            "format": 1,
            "updated": datetime.datetime.now(datetime.timezone.utc)
            .isoformat(timespec="seconds"),
            "seed": self.seed, "games": self.games,
            "population": self.population_size, "elite": self.elite_size,
            "sigma_frac": self.sigma_frac,
            "field": self.field,
            "decks": {name: state.to_dict()
                      for name, state in self.states.items()},
        })

    def load(self) -> bool:
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if not isinstance(data, dict):
            return False
        for name, raw in (data.get("decks") or {}).items():
            if isinstance(raw, dict):
                try:
                    self.states[name] = DeckState.from_dict(raw)
                except Exception:
                    continue
        return bool(self.states)

    # ---- reporting ---- #

    def report(self) -> str:
        lines = ["co-evolution state — " + str(self.state_path),
                 "  (raw scores are NOT comparable across generations: the "
                 "Hall of Fame slice of the",
                 "   opponent pool is resampled each generation. The fair "
                 "read is `delta`, which is",
                 "   best-minus-default measured on the SAME pool.)"]
        for deck, state in self.states.items():
            lines.append(f"  [{deck}] generation {state.generation}")
            for record in state.history:
                score = record.get("best_score")
                if score is None:
                    continue
                reference = record.get("reference_score")
                delta = record.get("delta_vs_default")
                delta_text = "" if delta is None else f" delta {delta:+.3f}"
                ref_text = "" if reference is None else f" (default {reference:.3f})"
                lines.append(
                    f"    gen {record['generation']:>2}: best {score:.3f}"
                    f"{ref_text}{delta_text} | mean "
                    f"{record.get('best_mean', 0):.1%} | worst "
                    f"{record.get('best_worst_cell')} "
                    f"{record.get('best_worst', 0):.1%} | "
                    f"vs-{BENCHMARK_CELL} {record.get('best_benchmark', 0):.1%}"
                    f" | elite {record.get('elite_origins')}")
        return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decks", nargs="+", default=list(DEFAULT_DECKS))
    parser.add_argument("--field", nargs="+", default=list(DEFAULT_FIELD))
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--elite", type=int, default=3)
    parser.add_argument("--games", type=int, default=24,
                        help="games per cell in the FULL evaluation")
    parser.add_argument("--screen-games", type=int, default=8,
                        help="games per cell in the racing screen (0 = off)")
    parser.add_argument("--hof-samples", type=int, default=2,
                        help="past champions sampled into the pool "
                             "(anti-overfit; 0 disables and is NOT advised)")
    parser.add_argument("--sigma-frac", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--control-dir", type=Path, default=LEAGUE_DIR,
                        help="where STOP / PAUSE sentinel files are read")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report", action="store_true",
                        help="print the checkpoint and exit")
    args = parser.parse_args()

    loop = CoEvolution(decks=args.decks, field_decks=args.field,
                       population=args.population, elite=args.elite,
                       games=args.games, screen_games=args.screen_games,
                       seed=args.seed, hof_samples=args.hof_samples,
                       sigma_frac=args.sigma_frac, state_path=args.state)
    if args.resume or args.report:
        if loop.load():
            print(f"resumed {args.state}")
        elif args.report:
            raise SystemExit(f"no checkpoint at {args.state}")
    if args.report:
        loop.prepare()
        print(loop.report())
        return

    args.control_dir.mkdir(parents=True, exist_ok=True)
    loop.run(args.generations, args.control_dir)
    print("\n" + loop.report())
    print("\n" + loop.hof.report())
    print("\nNOTE: these champions are CANDIDATES. Offline fitness is not "
          "the ladder — nothing here is promoted or shipped.")


if __name__ == "__main__":
    main()
