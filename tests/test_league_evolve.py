"""Tests for the co-evolutionary loop (Phase 2).

The properties that are cheap to get silently wrong and expensive to
discover after a long run: mutation that leaves the legal band (which
would break the structural invariants the schemas encode), an opponent
pool that quietly loses its grounding, a checkpoint that does not
actually resume, and a pause protocol that ignores STOP.

The loop itself is exercised end-to-end at tiny scale (`test_one_real
_generation`) so the wiring is covered without a long simulation.

Run from the repo root:
    python -m unittest tests.test_league_evolve
"""

from __future__ import annotations

import datetime
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.league.evolve import (PAUSE_WINDOW, CoEvolution, DeckState,
                               Individual, crossover, mutate, wait_if_paused)
from src.league.hall_of_fame import Champion, HallOfFame
from src.league.modules import module_for_deck


class TestMutation(unittest.TestCase):

    def setUp(self) -> None:
        self.rng = random.Random(0)
        self.module = module_for_deck("grimmsnarl")
        self.theta = self.module.default_theta()

    def test_mutants_never_leave_the_legal_band(self) -> None:
        """The whole safety story: structural invariants live in the
        schema, so mutation cannot break them however wild sigma gets."""
        theta = self.theta
        for _ in range(400):
            theta = mutate(theta, self.rng, sigma_frac=3.0, rate=1.0)
            for spec in theta.schema.specs:
                self.assertGreaterEqual(theta[spec.name], spec.low, spec.name)
                self.assertLessEqual(theta[spec.name], spec.high, spec.name)

    def test_invariants_survive_extreme_mutation(self) -> None:
        """Concretely: abilities must never outrank attaching, and an
        attack must never reach the trainer band — after ANY mutation."""
        theta = self.theta
        for _ in range(200):
            theta = mutate(theta, self.rng, sigma_frac=5.0, rate=1.0)
            self.assertLess(theta["ability_band"]
                            + theta["adrena_brain_live_bonus"], 55.0)
            self.assertLess(theta["rare_candy_score"], 80.0)

        abom = module_for_deck("abomasnow").default_theta()
        for _ in range(200):
            abom = mutate(abom, self.rng, sigma_frac=5.0, rate=1.0)
            ceiling = (abom["attack_band_floor"]
                       + 450.0 / abom["attack_scale"]
                       + abom["attack_ko_bonus"])
            self.assertLessEqual(ceiling, 35.0 + 1e-9)

    def test_mutation_actually_moves_something(self) -> None:
        mutated = mutate(self.theta, self.rng, sigma_frac=0.2, rate=1.0)
        self.assertNotEqual(mutated, self.theta)
        self.assertGreater(len(mutated.diff_from_defaults()), 0)

    def test_integral_knobs_stay_whole(self) -> None:
        theta = self.theta
        for _ in range(50):
            theta = mutate(theta, self.rng, rate=1.0)
        for spec in theta.schema.specs:
            if spec.integral:
                self.assertEqual(theta[spec.name], float(int(theta[spec.name])),
                                 spec.name)

    def test_steps_are_scaled_to_each_knob_own_band(self) -> None:
        """A 0-100 weight and a 0.0-2.0 multiplier must move by
        comparable FRACTIONS, else evolution only explores the big
        knobs."""
        rng = random.Random(7)
        moves: dict[str, list[float]] = {}
        for _ in range(300):
            mutated = mutate(self.theta, rng, sigma_frac=0.1, rate=1.0)
            for spec in self.theta.schema.specs:
                if spec.integral or spec.high == spec.low:
                    continue
                delta = abs(mutated[spec.name] - self.theta[spec.name])
                moves.setdefault(spec.name, []).append(
                    delta / (spec.high - spec.low))
        averages = [sum(v) / len(v) for v in moves.values()]
        self.assertLess(max(averages), 4 * min(averages),
                        "band-normalized steps should be comparable")

    def test_crossover_takes_from_both_parents(self) -> None:
        rng = random.Random(1)
        a = self.theta
        b = mutate(a, rng, sigma_frac=0.4, rate=1.0)
        child = crossover(a, b, rng)
        from_a = sum(1 for i, v in enumerate(child.to_vector())
                     if v == a.to_vector()[i])
        from_b = sum(1 for i, v in enumerate(child.to_vector())
                     if v == b.to_vector()[i])
        self.assertGreater(from_a, 0)
        self.assertGreater(from_b, 0)


class TestGrounding(unittest.TestCase):
    """The invariant that keeps offline fitness honest."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.hof_path = Path(self.tmp.name) / "hof.json"
        self.loop = CoEvolution(
            decks=["grimmsnarl", "abomasnow"],
            field_decks=["alakazam", "starmie", "spidops"],
            population=4, elite=2, games=2, screen_games=0,
            state_path=Path(self.tmp.name) / "state.json",
            hof_path=self.hof_path)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_real_anchors_and_the_zoo_are_always_in_the_pool(self) -> None:
        self.loop.prepare()
        names = [e.name for e in self.loop.opponent_pool("grimmsnarl")]
        for anchor in ("alakazam", "starmie", "spidops"):
            self.assertIn(anchor, names, "the real metagame never leaves")

    def test_other_candidates_are_opponents_too(self) -> None:
        """Nobody leaves: candidates fight each other, so the population
        cannot drift into a private metagame."""
        self.loop.prepare()
        names = [e.name for e in self.loop.opponent_pool("grimmsnarl")]
        self.assertIn("abomasnow", names)
        self.assertNotIn("grimmsnarl", names, "never its own opponent")

    def test_hall_of_fame_champions_enter_the_pool(self) -> None:
        """Anti-overfit: without past champions the loop CYCLES."""
        module = module_for_deck("abomasnow")
        hof = HallOfFame(self.hof_path)
        hof.add(Champion(deck="abomasnow", module=module.name,
                         theta=module.default_theta(), score=0.6,
                         mean=0.6, radar=0.6, generation=0))
        hof.save()
        loop = CoEvolution(decks=["grimmsnarl"],
                           field_decks=["alakazam"], population=2, elite=1,
                           games=2, screen_games=0, hof_samples=2,
                           state_path=Path(self.tmp.name) / "s2.json",
                           hof_path=self.hof_path)
        loop.prepare()
        names = [e.name for e in loop.opponent_pool("grimmsnarl")]
        self.assertTrue(any(n.startswith("hof:") for n in names),
                        f"no archived opponent in {names}")

    def test_hof_opponents_carry_their_own_theta(self) -> None:
        module = module_for_deck("abomasnow")
        theta = module.default_theta().replace(hammer_min_deck=30)
        hof = HallOfFame(self.hof_path)
        hof.add(Champion(deck="abomasnow", module=module.name, theta=theta,
                         score=0.6, mean=0.6, radar=0.6))
        hof.save()
        loop = CoEvolution(decks=["grimmsnarl"], field_decks=["alakazam"],
                           population=2, elite=1, games=2, screen_games=0,
                           state_path=Path(self.tmp.name) / "s3.json",
                           hof_path=self.hof_path)
        loop.prepare()
        archived = [e for e in loop.opponent_pool("grimmsnarl")
                    if e.name.startswith("hof:")]
        self.assertTrue(archived)
        self.assertEqual(archived[0].theta["hammer_min_deck"], 30.0)


class TestPauseProtocol(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_stop_file_requests_a_clean_exit(self) -> None:
        (self.dir / "STOP").write_text("", encoding="utf-8")
        self.assertTrue(wait_if_paused(self.dir))

    def test_clean_directory_does_not_block(self) -> None:
        with mock.patch("src.league.evolve.datetime") as fake:
            fake.datetime.now.return_value.time.return_value = \
                datetime.time(12, 0)
            self.assertFalse(wait_if_paused(self.dir))

    def test_stop_wins_over_pause(self) -> None:
        """A STOP during a PAUSE must still exit, not block forever."""
        (self.dir / "PAUSE").write_text("", encoding="utf-8")
        (self.dir / "STOP").write_text("", encoding="utf-8")
        self.assertTrue(wait_if_paused(self.dir))

    def test_the_job_window_is_the_collector_window(self) -> None:
        self.assertEqual(PAUSE_WINDOW[0], datetime.time(20, 58))
        self.assertEqual(PAUSE_WINDOW[1], datetime.time(21, 32))


class TestCheckpointResume(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.state = Path(self.tmp.name) / "state.json"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _loop(self) -> CoEvolution:
        return CoEvolution(decks=["grimmsnarl"], field_decks=["alakazam"],
                           population=3, elite=1, games=2, screen_games=0,
                           state_path=self.state,
                           hof_path=Path(self.tmp.name) / "hof.json")

    def test_state_round_trips_with_genomes_intact(self) -> None:
        loop = self._loop()
        loop.prepare()
        state = loop.states["grimmsnarl"]
        state.generation = 4
        state.population[0].theta = state.population[0].theta.replace(
            ability_band=44.0)
        state.history.append({"generation": 3, "best_score": 0.61})
        loop.save()

        resumed = self._loop()
        self.assertTrue(resumed.load())
        back = resumed.states["grimmsnarl"]
        self.assertEqual(back.generation, 4)
        self.assertEqual(back.population[0].theta["ability_band"], 44.0)
        self.assertEqual(back.history[-1]["best_score"], 0.61)

    def test_resume_does_not_reseed_an_existing_population(self) -> None:
        loop = self._loop()
        loop.prepare()
        loop.states["grimmsnarl"].generation = 2
        loop.save()
        resumed = self._loop()
        resumed.load()
        resumed.prepare()
        self.assertEqual(resumed.states["grimmsnarl"].generation, 2)

    def test_missing_checkpoint_is_not_an_error(self) -> None:
        self.assertFalse(self._loop().load())

    def test_corrupt_checkpoint_is_not_an_error(self) -> None:
        self.state.parent.mkdir(parents=True, exist_ok=True)
        self.state.write_text("{broken", encoding="utf-8")
        self.assertFalse(self._loop().load())

    def test_save_is_atomic(self) -> None:
        loop = self._loop()
        loop.prepare()
        loop.save()
        self.assertTrue(self.state.exists())
        self.assertEqual(list(self.state.parent.glob("*.tmp")), [])

    def test_generation_zero_seeds_from_the_versioned_default(self) -> None:
        loop = self._loop()
        loop.prepare()
        population = loop.states["grimmsnarl"].population
        self.assertEqual(population[0].origin, "default")
        self.assertEqual(population[0].theta,
                         module_for_deck("grimmsnarl").default_theta())
        self.assertTrue(any(i.origin == "seed-mutant" for i in population[1:]))


class TestOneRealGeneration(unittest.TestCase):
    """End-to-end wiring at tiny scale (real games, tiny counts)."""

    def test_one_generation_runs_and_advances_the_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loop = CoEvolution(
                decks=["abomasnow"], field_decks=["starmie"],
                population=3, elite=1, games=2, screen_games=0,
                state_path=root / "state.json", hof_path=root / "hof.json")
            loop.prepare()
            record = loop.step("abomasnow", root)

            self.assertIsNotNone(record)
            self.assertEqual(record["generation"], 0)
            self.assertIsNotNone(record["best_score"])
            self.assertIn("starmie", record["opponents"])
            # both gate arms measured on the SAME pool, same generation
            self.assertIsNotNone(record["reference_score"])
            self.assertIsNotNone(record["delta_vs_default"])
            self.assertAlmostEqual(
                record["delta_vs_default"],
                record["best_score"] - record["reference_score"])
            # the incumbent is what only the gate may replace
            self.assertIsNotNone(loop.states["abomasnow"].incumbent)
            self.assertIn("promoted", record)
            self.assertIn("gate", record)
            promoted = record["promoted"]
            incumbent = loop.states["abomasnow"].incumbent
            default = module_for_deck("abomasnow").default_theta()
            if not promoted:
                self.assertEqual(incumbent, default,
                                 "a held generation must not move the "
                                 "incumbent")
            self.assertEqual(loop.states["abomasnow"].generation, 1)
            self.assertEqual(len(loop.states["abomasnow"].population), 3)
            self.assertGreaterEqual(len(loop.hof), 1,
                                    "the generation champion must be archived")
            # the next population is elite + offspring, not reseeded
            origins = [i.origin for i in loop.states["abomasnow"].population]
            self.assertTrue(any(o in ("mutant", "crossover") for o in origins),
                            origins)

    def test_the_incumbent_is_never_its_own_challenger(self) -> None:
        """REGRESSION (Phase 3, caught in the field): a population can
        degenerate into clones of the incumbent — crossover(x, x) == x —
        and then the gate is handed the SAME genome on both arms. That is
        the null case, so it promotes at the alpha rate: a champion
        bit-identical to the default theta was promoted this way."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loop = CoEvolution(
                decks=["abomasnow"], field_decks=["starmie"],
                population=4, elite=1, games=2, screen_games=2,
                gate_games=2, min_gate_games=0,
                state_path=root / "state.json", hof_path=root / "hof.json")
            loop.prepare()
            state = loop.states["abomasnow"]
            incumbent = module_for_deck("abomasnow").default_theta()
            state.incumbent = incumbent
            # the degenerate case: every member IS the incumbent
            state.population = [Individual(theta=incumbent, origin="clone")
                                for _ in range(4)]
            record = loop.step("abomasnow", root)

            self.assertIsNotNone(record)
            gate = record["gate"]
            self.assertNotEqual(gate["challenger_rate"], None)
            # whatever the gate decided, it was NOT handed a clone
            for champion in loop.hof:
                if "gate-promoted" in champion.note:
                    self.assertNotEqual(
                        champion.theta, incumbent,
                        "promoted a genome identical to the incumbent")

    def test_refill_never_seeds_a_clone_of_the_incumbent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loop = CoEvolution(
                decks=["abomasnow"], field_decks=["starmie"],
                population=6, elite=1, games=2, screen_games=2,
                gate_games=2, min_gate_games=0,
                state_path=root / "state.json", hof_path=root / "hof.json")
            loop.prepare()
            loop.step("abomasnow", root)
            state = loop.states["abomasnow"]
            for individual in state.population:
                self.assertNotEqual(individual.theta, state.incumbent,
                                    "a clone of the incumbent re-entered "
                                    "the population")

    def test_a_stop_mid_run_checkpoints_and_exits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "STOP").write_text("", encoding="utf-8")
            loop = CoEvolution(
                decks=["abomasnow"], field_decks=["starmie"],
                population=2, elite=1, games=2, screen_games=0,
                state_path=root / "state.json", hof_path=root / "hof.json")
            loop.run(generations=3, control_dir=root)
            self.assertTrue((root / "state.json").exists())
            self.assertEqual(loop.states["abomasnow"].generation, 0,
                             "STOP must be honoured before any work")


if __name__ == "__main__":
    unittest.main()
