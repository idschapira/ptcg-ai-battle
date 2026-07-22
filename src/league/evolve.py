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
from .gate import (format_power_table, games_for_effect,
                   minimum_detectable_effect, pooled, promotion_gate)
from .hall_of_fame import Champion, HallOfFame
from .modules import module_for_deck
from .theta import Theta

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
    #: The reigning genome. Only the significance gate replaces it, so
    #: it is the one thing in the loop that noise cannot move.
    incumbent: Theta | None = None
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
            "incumbent": self.incumbent.to_dict() if self.incumbent else None,
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DeckState":
        deck = str(data.get("deck", ""))
        schema = module_for_deck(deck).schema
        state = cls(deck=deck, generation=int(data.get("generation", 0)),
                    history=list(data.get("history") or []))
        if data.get("incumbent"):
            state.incumbent = schema.from_dict(data["incumbent"])
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

class LoopLock:
    """Single-writer lock on the checkpoint directory.

    Two loops writing one `evolve_state.json` silently corrupt each
    other: generation counters interleave and the Hall of Fame gets
    champions from a run that no longer exists. That happened — a loop
    outlived the shell that launched it (a `nohup` child survives its
    wrapper being killed), a second was started, and both kept saving.
    Deleting the state files did not help, because the live process
    simply rewrote them from memory on its next save.

    So: acquire or refuse. The lock stores the PID and is stale-checked,
    which matters precisely because the failure mode here is a process
    that outlives its parent."""

    __slots__ = ("path", "_held")

    def __init__(self, control_dir: Path) -> None:
        self.path = control_dir / "evolve.lock"
        self._held = False

    @staticmethod
    def _alive(pid: int) -> bool:
        """Is this PID still running?

        NOT `os.kill(pid, 0)`. That is the POSIX idiom, but on Windows
        Python maps any signal other than CTRL_C_EVENT/CTRL_BREAK_EVENT
        onto TerminateProcess — so the "harmless existence check" would
        KILL the process it is asking about. Query the handle instead."""
        if pid <= 0:
            return False
        if os.name == "nt":
            import ctypes
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION,
                                          False, pid)
            if not handle:
                return False          # gone, or not ours to inspect
            try:
                code = ctypes.c_ulong()
                if kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                    return code.value == STILL_ACTIVE
                return True           # cannot tell: assume held
            finally:
                kernel32.CloseHandle(handle)
        try:
            os.kill(pid, 0)           # POSIX: genuinely just a probe
        except ProcessLookupError:
            return False
        except PermissionError:
            return True               # exists, owned by someone else
        except OSError:
            return False
        return True

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            owner = int(raw.get("pid", -1))
        except (OSError, ValueError, TypeError):
            owner = -1
        if owner > 0 and owner != os.getpid() and self._alive(owner):
            print(f"[evolve] REFUSING to start: another loop (pid {owner}) "
                  f"already owns {self.path.parent}. Stop it first, or "
                  f"delete {self.path.name} if you know it is stale.",
                  flush=True)
            return False
        _atomic_write(self.path, {"pid": os.getpid(),
                                  "started": datetime.datetime.now(
                                      datetime.timezone.utc).isoformat(
                                          timespec="seconds")})
        self._held = True
        return True

    def release(self) -> None:
        if not self._held:
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if int(raw.get("pid", -1)) == os.getpid():
                self.path.unlink()
        except (OSError, ValueError, TypeError):
            pass
        self._held = False

    def __enter__(self) -> "LoopLock":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


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
                 screen_games: int = 6, seed: int = 0,
                 hof_samples: int = 2, sigma_frac: float = 0.12,
                 gate_games: int = 100, min_gate_games: int = 600,
                 state_path: Path = DEFAULT_STATE,
                 hof_path: Path | None = None) -> None:
        self.decks = list(decks)
        self.field = list(field_decks)
        self.population_size = population
        self.elite_size = max(1, min(elite, population))
        self.games = games
        self.screen_games = screen_games
        #: games/cell for the DECISION measurement (both arms)
        self.gate_games = gate_games
        #: floor on decided games per arm before the gate may pass at all
        self.min_gate_games = min_gate_games
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

    # ---- screening: successive halving ---- #

    def successive_halving(self, deck: str, individuals: Sequence[Individual],
                           pool: Sequence[Entry]) -> list[Individual]:
        """Find the finalist cheaply, then let the gate decide promotion.

        Screening and testing are different problems. Here we only need
        a RANKING good enough to pick who faces the incumbent, so games
        are spent in rounds: everyone plays a few, the clearly worse
        half is dropped, survivors get double the budget, repeat. The
        expensive, properly-powered measurement happens once, in the
        gate — not P times here.

        Ranking uses POOLED WINRATE, not `Fitness.score`: the score's
        worst-cell term is a min over ~11 noisy cells and is far too
        jumpy to rank on at screening sample sizes."""
        survivors = list(individuals)
        if self.screen_games <= 0 or len(survivors) <= 1:
            return survivors[:1]
        games = self.screen_games
        rounds = 0
        while len(survivors) > 1:
            scored: list[tuple[float, Individual]] = []
            for individual in survivors:
                fitness = self._evaluate(deck, individual.theta, pool, games)
                rate = pooled(fitness.cells).rate
                individual.score = fitness.score()
                individual.mean = fitness.mean
                worst = fitness.worst
                individual.worst_cell = worst.opponent if worst else None
                individual.worst = worst.winrate if worst else None
                scored.append((rate, individual))
            scored.sort(key=lambda pair: -pair[0])
            keep = max(1, len(scored) // 2)
            survivors = [individual for _, individual in scored[:keep]]
            games *= 2
            rounds += 1
            if rounds > 8:  # guard against a pathological population size
                break
        return survivors

    # ---- one generation ---- #

    def step(self, deck: str, control_dir: Path) -> dict | None:
        """One generation for one deck. None when a STOP was requested.

        Structure: screen (cheap) -> gate (properly powered) -> refill.
        The incumbent only ever changes through the gate, which is what
        makes a promoted champion mean something."""
        state = self.states[deck]
        pool = self.opponent_pool(deck)
        if wait_if_paused(control_dir):
            return None

        default_theta = module_for_deck(deck).default_theta()
        if state.incumbent is None:
            state.incumbent = default_theta

        # A CHALLENGER MUST DIFFER FROM THE INCUMBENT. Otherwise the gate
        # is handed the same genome twice — the exact null case, which by
        # construction fires at the alpha rate (observed: a "promotion"
        # of a genome bit-identical to the default). The population can
        # degenerate this way because crossover(x, x) == x.
        challengers = [i for i in state.population
                       if i.theta != state.incumbent]
        if not challengers:
            challengers = [
                Individual(theta=mutate(state.incumbent, self.rng,
                                        self.sigma_frac, rate=1.0),
                           origin="reseed")
                for _ in range(self.population_size)]
        finalists = self.successive_halving(deck, challengers, pool)
        if not finalists:
            return None
        challenger = finalists[0]
        if challenger.theta == state.incumbent:  # belt and braces
            return None
        if wait_if_paused(control_dir):
            return None

        # The decision measurement: both arms at the gate's sample size,
        # against the SAME pool, in the same generation.
        challenger_fitness = self._evaluate(deck, challenger.theta, pool,
                                            self.gate_games)
        if wait_if_paused(control_dir):
            return None
        incumbent_fitness = self._evaluate(deck, state.incumbent, pool,
                                           self.gate_games)

        result = promotion_gate(pooled(challenger_fitness.cells),
                                pooled(incumbent_fitness.cells),
                                min_games=self.min_gate_games)

        worst = challenger_fitness.worst
        bench = challenger_fitness.cell(BENCHMARK_CELL)
        challenger.score = challenger_fitness.score()
        challenger.mean = challenger_fitness.mean
        challenger.worst_cell = worst.opponent if worst else None
        challenger.worst = worst.winrate if worst else None
        challenger.benchmark = bench.winrate if bench else None

        # The reigning genome is always in the archive, so the anti-
        # overfit pool is never empty; a promotion adds the new one.
        if not self.hof.for_deck(deck):
            self.hof.add(Champion.from_fitness(
                incumbent_fitness, state.incumbent, generation=0,
                note="incumbent baseline (reference arm, never tested)"))
        if result.promoted:
            state.incumbent = challenger.theta
            self.hof.add(Champion.from_fitness(
                challenger_fitness, challenger.theta,
                generation=state.generation,
                note=f"gate-promoted gen {state.generation}: "
                     f"{result.delta:+.1%} "
                     f"CI [{result.diff_low:+.1%},{result.diff_high:+.1%}]"))
        self.hof.save()

        exceptions = challenger_fitness.exceptions + incumbent_fitness.exceptions
        if exceptions:
            print(f"  !! {exceptions} exceptions on {deck}", flush=True)

        record = {
            "generation": state.generation,
            "promoted": result.promoted,
            "gate": result.to_dict(),
            "challenger_rate": result.challenger.rate,
            "incumbent_rate": result.incumbent.rate,
            "delta": result.delta,
            "reference_score": incumbent_fitness.score(),
            "reference_mean": incumbent_fitness.mean,
            "delta_vs_default": challenger_fitness.score()
            - incumbent_fitness.score(),
            "best_score": challenger_fitness.score(),
            "best_mean": challenger_fitness.mean,
            "best_worst_cell": challenger.worst_cell,
            "best_worst": challenger.worst,
            "best_benchmark": challenger.benchmark,
            "elite_origins": [challenger.origin],
            "evaluated": len(state.population),
            "opponents": [e.name for e in pool],
            "games_per_cell": self.gate_games,
            "exceptions": exceptions,
        }
        state.history.append(record)

        # refill: the incumbent seeds the next population, so evolution
        # always explores around the thing that actually survived a test
        # The population holds CHALLENGERS only; the incumbent is the
        # reference arm, not a competitor. It still breeds, so evolution
        # explores around the genome that actually survived a test.
        next_population: list[Individual] = []
        parents = [challenger.theta, state.incumbent]
        while len(next_population) < self.population_size:
            if self.rng.random() < 0.3:
                child = crossover(parents[0], parents[1], self.rng)
                origin = "crossover"
            else:
                child = mutate(self.rng.choice(parents), self.rng,
                               self.sigma_frac)
                origin = "mutant"
            # crossover(x, x) == x, and a mutation can land back on the
            # incumbent: never seed a clone of it into the population
            guard = 0
            while child == state.incumbent and guard < 8:
                child = mutate(child, self.rng, self.sigma_frac, rate=1.0)
                origin = "mutant"
                guard += 1
            if child == state.incumbent:
                continue
            next_population.append(Individual(theta=child, origin=origin))
        state.population = next_population
        state.generation += 1
        return record


    # ---- driver ---- #

    def run(self, generations: int, control_dir: Path) -> None:
        self.prepare()
        cells = len(self.field) + self.hof_samples
        print(format_power_table(max(1, cells)), flush=True)
        print(f"gate: {self.gate_games} games/cell x ~{cells} cells = "
              f"~{self.gate_games * cells} decided games per arm; "
              f"minimum detectable gain "
              f"{minimum_detectable_effect(self.gate_games * cells):.1%}",
              flush=True)
        for _ in range(generations):
            for deck in self.decks:
                state = self.states[deck]
                print(f"\n[gen {state.generation}] {deck} "
                      f"(pop {len(state.population)}, screen "
                      f"{self.screen_games} -> gate {self.gate_games} "
                      f"games/cell)", flush=True)
                t0 = time.perf_counter()
                record = self.step(deck, control_dir)
                if record is None:
                    print("[evolve] STOP requested — checkpointing and exiting",
                          flush=True)
                    self.save()
                    return
                gate = record.get("gate") or {}
                verdict = "PROMOTED" if record.get("promoted") else "held"
                print(f"  {verdict}: challenger "
                      f"{record['challenger_rate']:.1%} vs incumbent "
                      f"{record['incumbent_rate']:.1%} "
                      f"(delta {record['delta']:+.1%}, CI "
                      f"[{gate.get('diff_low', 0):+.1%}, "
                      f"{gate.get('diff_high', 0):+.1%}]) — "
                      f"{gate.get('reason', '')}", flush=True)
                print(f"  challenger mean {record['best_mean']:.1%} | worst "
                      f"{record['best_worst_cell']} "
                      f"{record['best_worst']:.1%} | vs-{BENCHMARK_CELL} "
                      f"{record['best_benchmark']:.1%} "
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
                gate = record.get("gate") or {}
                verdict = "PROMOTED" if record.get("promoted") else "held    "
                lines.append(
                    f"    gen {record['generation']:>2}: {verdict} "
                    f"challenger {record.get('challenger_rate', 0):.1%} vs "
                    f"incumbent {record.get('incumbent_rate', 0):.1%} "
                    f"(delta {record.get('delta', 0):+.1%}, CI "
                    f"[{gate.get('diff_low', 0):+.1%}, "
                    f"{gate.get('diff_high', 0):+.1%}]) | score {score:.3f} | "
                    f"worst {record.get('best_worst_cell')} "
                    f"{record.get('best_worst', 0):.1%}")
        return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decks", nargs="+", default=list(DEFAULT_DECKS))
    parser.add_argument("--field", nargs="+", default=list(DEFAULT_FIELD))
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--elite", type=int, default=3)
    parser.add_argument("--games", type=int, default=24,
                        help="(legacy) games per cell for ad-hoc evaluation")
    parser.add_argument("--screen-games", type=int, default=6,
                        help="games/cell in the FIRST successive-halving "
                             "round; doubles each round (0 = no screening)")
    parser.add_argument("--gate-games", type=int, default=100,
                        help="games/cell for the DECISION measurement; "
                             "sets the minimum detectable effect")
    parser.add_argument("--min-gate-games", type=int, default=600,
                        help="decided games per arm below which the gate "
                             "refuses to promote at all (anti-underpower)")
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
                       sigma_frac=args.sigma_frac, gate_games=args.gate_games,
                       min_gate_games=args.min_gate_games,
                       state_path=args.state)
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
    lock = LoopLock(args.control_dir)
    if not lock.acquire():
        raise SystemExit(2)
    try:
        loop.run(args.generations, args.control_dir)
    finally:
        lock.release()
    print("\n" + loop.report())
    print("\n" + loop.hof.report())
    print("\nNOTE: these champions are CANDIDATES. Offline fitness is not "
          "the ladder — nothing here is promoted or shipped.")


if __name__ == "__main__":
    main()
