"""Tests for the league foundation: Theta, the Hall of Fame, fitness math.

These cover the properties Phase 2's mutation loop will rely on and that
are cheap to get subtly wrong: defaults that are bit-exact, genomes that
cannot leave their bounds, an archive that survives a schema change and a
crash mid-write, and a fitness score that actually punishes a bad worst
cell instead of averaging it away.

Run from the repo root:
    python -m unittest tests.test_league_core
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.league.fitness import Cell, Fitness, build_cohort
from src.league.hall_of_fame import Champion, HallOfFame
from src.league.modules import (DECK_MODULE, default_theta_for_deck,
                                module_for_deck)
from src.league.modules.crustle import CrustleModule
from src.league.modules.grimmsnarl import GrimmsnarlModule
from src.league.theta import ParamSpec, Theta, ThetaSchema


class TestTheta(unittest.TestCase):

    def setUp(self) -> None:
        self.schema = ThetaSchema((
            ParamSpec("a", 34.8, 0.0, 50.0),
            ParamSpec("n", 15, 4, 30, integral=True),
        ))

    def test_defaults_are_bit_exact(self) -> None:
        """The whole retrofit rests on this: a default must survive
        construction unrounded and unrescaled."""
        theta = self.schema.defaults()
        self.assertEqual(theta["a"], 34.8)
        self.assertEqual(theta["n"], 15.0)
        self.assertEqual(theta.to_vector(), (34.8, 15.0))

    def test_every_module_default_is_bit_exact(self) -> None:
        for module in (CrustleModule(), GrimmsnarlModule()):
            theta = module.default_theta()
            for spec in module.schema:
                self.assertEqual(theta[spec.name], float(spec.default),
                                 f"{module.name}.{spec.name} drifted")
            self.assertEqual(theta.diff_from_defaults(), {})

    def test_construction_clips_into_bounds(self) -> None:
        theta = self.schema.from_vector([999.0, -50.0])
        self.assertEqual(theta["a"], 50.0)
        self.assertEqual(theta["n"], 4.0)

    def test_replace_clips_and_rejects_typos(self) -> None:
        theta = self.schema.defaults()
        self.assertEqual(theta.replace(a=1000.0)["a"], 50.0)
        with self.assertRaises(KeyError):
            theta.replace(nope=1.0)

    def test_integral_params_snap(self) -> None:
        self.assertEqual(self.schema.from_vector([0.0, 7.6])["n"], 8.0)
        self.assertEqual(self.schema.defaults().i("n"), 15)

    def test_garbage_values_fall_back_to_the_default(self) -> None:
        for junk in (None, "x", float("nan")):
            self.assertEqual(self.schema.from_dict({"a": junk})["a"], 34.8)

    def test_from_dict_is_forward_compatible(self) -> None:
        """A stored genome with an unknown knob (schema shrank) and a
        missing knob (schema grew) must still load."""
        theta = self.schema.from_dict({"a": 12.0, "gone": 5.0})
        self.assertEqual(theta["a"], 12.0)
        self.assertEqual(theta["n"], 15.0)

    def test_json_round_trip(self) -> None:
        theta = self.schema.defaults().replace(a=12.5)
        back = self.schema.from_dict(json.loads(theta.to_json()))
        self.assertEqual(theta, back)

    def test_immutability_and_hashability(self) -> None:
        theta = self.schema.defaults()
        moved = theta.replace(a=1.0)
        self.assertEqual(theta["a"], 34.8, "replace must not mutate in place")
        self.assertNotEqual(theta, moved)
        self.assertIsInstance(hash(theta), int)

    def test_distance_is_band_normalized(self) -> None:
        theta = self.schema.defaults()
        self.assertEqual(theta.distance(theta), 0.0)
        # opposite corners of one 50-wide band = exactly 1.0, whatever
        # the raw units are: diversity is comparable across knobs
        low = self.schema.from_vector([0.0, 4])
        high = self.schema.from_vector([50.0, 4])
        self.assertAlmostEqual(low.distance(high), 1.0)

    def test_bad_specs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ParamSpec("x", 5.0, 10.0, 20.0)   # default outside bounds
        with self.assertRaises(ValueError):
            ParamSpec("x", 5.0, 20.0, 10.0)   # inverted band
        with self.assertRaises(ValueError):
            ThetaSchema((ParamSpec("d", 1.0, 0.0, 2.0),
                         ParamSpec("d", 1.0, 0.0, 2.0)))


class TestModuleRegistry(unittest.TestCase):

    def test_known_decks_map_to_their_module(self) -> None:
        self.assertEqual(module_for_deck("crustle").name, "crustle")
        self.assertEqual(module_for_deck("grimmsnarl").name, "grimmsnarl")

    def test_unknown_deck_falls_back_to_generic(self) -> None:
        module = module_for_deck("a_deck_nobody_wrote_rules_for")
        self.assertEqual(module.name, "generic")
        self.assertEqual(len(module.schema), 0)

    def test_deck_theta_defaults_are_versioned_and_stable(self) -> None:
        for deck in DECK_MODULE:
            theta = default_theta_for_deck(deck)
            self.assertEqual(theta.diff_from_defaults(), {}, deck)


class TestFitnessMath(unittest.TestCase):

    @staticmethod
    def _fitness(cells: dict[str, tuple[int, int]]) -> Fitness:
        f = Fitness(candidate="c", deck="crustle")
        f.cells = [Cell(opponent=name, wins=w, losses=l, draws=0, exceptions=0)
                   for name, (w, l) in cells.items()]
        return f

    def test_wilson_interval_is_reported_per_cell(self) -> None:
        cell = Cell("alakazam", wins=20, losses=20, draws=0, exceptions=0)
        low, high = cell.ci
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)
        self.assertTrue(cell.directional, "an even split is not a result")

    def test_a_decisive_cell_is_not_directional(self) -> None:
        cell = Cell("spidops", wins=38, losses=2, draws=0, exceptions=0)
        self.assertFalse(cell.directional)

    def test_score_punishes_a_bad_worst_cell(self) -> None:
        """The reason fitness is not a plain mean: a candidate that farms
        the field and auto-loses one cell must rank below a flat one."""
        flat = self._fitness({"alakazam": (24, 16), "spidops": (24, 16),
                              "raging_bolt": (24, 16)})
        spiky = self._fitness({"alakazam": (2, 38), "spidops": (37, 3),
                               "raging_bolt": (33, 7)})
        self.assertAlmostEqual(spiky.mean, flat.mean,
                               msg="identical mean by construction")
        self.assertGreater(flat.score(), spiky.score(),
                           "the worst cell must carry real weight")

    def test_worst_cell_and_benchmark_are_always_in_the_report(self) -> None:
        fitness = self._fitness({"alakazam": (10, 30), "spidops": (30, 10)})
        report = fitness.report()
        self.assertIn("WORST CELL: alakazam", report)
        self.assertIn("vs ALAKAZAM", report)
        self.assertEqual(fitness.worst.opponent, "alakazam")

    def test_missing_benchmark_is_called_out_not_hidden(self) -> None:
        fitness = self._fitness({"spidops": (30, 10)})
        self.assertIn("NOT MEASURED", fitness.report())

    def test_radar_weights_the_ladder_relevant_cells(self) -> None:
        fitness = self._fitness({"alakazam": (10, 30), "starmie": (30, 10)})
        self.assertLess(fitness.radar, fitness.mean,
                        "losing the double-weighted cell must cost more")

    def test_cohort_keeps_every_candidate_as_an_opponent(self) -> None:
        cohort = build_cohort(["grimmsnarl", "crustle"], ["alakazam"])
        names = [e.name for e in cohort]
        self.assertEqual(sorted(names), ["alakazam", "crustle", "grimmsnarl"])
        self.assertEqual(sum(1 for e in cohort if e.is_candidate), 2)

    def test_cohort_does_not_duplicate_a_candidate_present_in_the_field(self):
        cohort = build_cohort(["crustle"], ["crustle", "alakazam"])
        self.assertEqual(len(cohort), 2)


class TestHallOfFame(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "hof.json"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    @staticmethod
    def _champion(deck: str, score: float, generation: int = 0) -> Champion:
        module = module_for_deck(deck)
        return Champion(deck=deck, module=module.name,
                        theta=module.default_theta(), score=score,
                        mean=score, radar=score, worst_cell="alakazam",
                        worst_winrate=score - 0.1, benchmark_winrate=0.4,
                        generation=generation)

    def test_empty_load_of_a_missing_file(self) -> None:
        hof = HallOfFame.load(self.path)
        self.assertEqual(len(hof), 0)
        self.assertIsNone(hof.best())
        self.assertIn("empty", hof.report())

    def test_round_trip_preserves_the_genome(self) -> None:
        hof = HallOfFame(self.path)
        theta = module_for_deck("crustle").default_theta().replace(
            low_deck=20, rebuild_score=48.0)
        hof.add(Champion(deck="crustle", module="crustle", theta=theta,
                         score=0.61, mean=0.6, radar=0.62))
        hof.save()

        back = HallOfFame.load(self.path)
        self.assertEqual(len(back), 1)
        self.assertEqual(back.champions[0].theta, theta)
        self.assertEqual(back.champions[0].theta["low_deck"], 20.0)

    def test_load_tolerates_a_schema_that_gained_a_knob(self) -> None:
        """A genome stored before a knob existed must still load, with
        the new knob at its default — Phase 2 will add knobs."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "format": 1, "champions": [
                {"deck": "crustle", "theta": {"low_deck": 22.0},
                 "score": 0.5, "mean": 0.5, "radar": 0.5}]}),
            encoding="utf-8")
        hof = HallOfFame.load(self.path)
        champion = hof.champions[0]
        self.assertEqual(champion.theta["low_deck"], 22.0)
        self.assertEqual(champion.theta["rebuild_score"], 42.0)

    def test_corrupt_file_yields_an_empty_archive_not_a_crash(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("{not json", encoding="utf-8")
        self.assertEqual(len(HallOfFame.load(self.path)), 0)

    def test_a_bad_row_does_not_sink_the_archive(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "format": 1, "champions": [
                "garbage",
                {"deck": "crustle", "theta": {}, "score": 0.5,
                 "mean": 0.5, "radar": 0.5}]}), encoding="utf-8")
        self.assertEqual(len(HallOfFame.load(self.path)), 1)

    def test_save_is_atomic(self) -> None:
        hof = HallOfFame(self.path)
        hof.add(self._champion("crustle", 0.5))
        hof.save()
        first = self.path.read_text(encoding="utf-8")
        hof.add(self._champion("grimmsnarl", 0.6, generation=1))
        hof.save()
        self.assertNotEqual(first, self.path.read_text(encoding="utf-8"))
        # no temp files left behind
        self.assertEqual([p.name for p in self.path.parent.glob("*.tmp")], [])

    def test_capacity_keeps_only_the_strongest_per_deck(self) -> None:
        hof = HallOfFame(self.path, capacity_per_deck=2)
        self.assertTrue(hof.add(self._champion("crustle", 0.50, 0)))
        self.assertTrue(hof.add(self._champion("crustle", 0.60, 1)))
        self.assertFalse(hof.add(self._champion("crustle", 0.40, 2)),
                         "a weaker challenger must not evict a champion")
        self.assertTrue(hof.add(self._champion("crustle", 0.70, 3)))
        self.assertEqual(sorted(c.score for c in hof.for_deck("crustle")),
                         [0.60, 0.70])

    def test_capacity_is_per_deck_not_global(self) -> None:
        hof = HallOfFame(self.path, capacity_per_deck=1)
        hof.add(self._champion("crustle", 0.5))
        hof.add(self._champion("grimmsnarl", 0.4))
        self.assertEqual(len(hof), 2)

    def test_opponents_defaults_to_one_per_deck(self) -> None:
        """The archive must stay DIVERSE: eight near-identical Crustles
        teach a challenger nothing."""
        hof = HallOfFame(self.path)
        hof.add(self._champion("crustle", 0.5, 0))
        hof.add(self._champion("crustle", 0.6, 1))
        hof.add(self._champion("grimmsnarl", 0.55, 0))
        opponents = hof.opponents()
        self.assertEqual([c.deck for c in opponents], ["crustle", "grimmsnarl"])
        self.assertEqual(opponents[0].score, 0.6)
        self.assertEqual(len(hof.opponents(deck="crustle")), 2)

    def test_best_and_next_generation_drive_the_phase2_loop(self) -> None:
        hof = HallOfFame(self.path)
        self.assertEqual(hof.next_generation("crustle"), 0)
        hof.add(self._champion("crustle", 0.5, generation=0))
        hof.add(self._champion("crustle", 0.7, generation=1))
        self.assertEqual(hof.next_generation("crustle"), 2)
        self.assertEqual(hof.best("crustle").score, 0.7)
        self.assertIsNone(hof.best("never_seen"))

    def test_resumability(self) -> None:
        """A fresh process must pick the loop up where it stopped."""
        hof = HallOfFame(self.path)
        hof.add(self._champion("crustle", 0.5, generation=0))
        hof.save()
        resumed = HallOfFame.load(self.path)
        resumed.add(self._champion("crustle", 0.8,
                                   generation=resumed.next_generation("crustle")))
        resumed.save()
        again = HallOfFame.load(self.path)
        self.assertEqual(again.next_generation("crustle"), 2)
        self.assertEqual(again.best("crustle").score, 0.8)

    def test_champion_from_fitness(self) -> None:
        fitness = Fitness(candidate="crustle", deck="crustle")
        fitness.cells = [
            Cell("alakazam", 8, 32, 0, 0), Cell("spidops", 30, 10, 0, 0)]
        champion = Champion.from_fitness(
            fitness, module_for_deck("crustle").default_theta(), generation=3)
        self.assertEqual(champion.deck, "crustle")
        self.assertEqual(champion.worst_cell, "alakazam")
        self.assertAlmostEqual(champion.benchmark_winrate, 0.2)
        self.assertEqual(champion.generation, 3)
        self.assertIn("crustle#g3", champion.summary())


if __name__ == "__main__":
    unittest.main()
