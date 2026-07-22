"""Tests for the portfolio harvest and pair-coverage maths (Phase 3).

The one idea worth testing hard: a portfolio's value at each opponent is
its BEST member's winrate, not the average. That is what makes a
rock-paper-scissors trio worth more than its best single deck, and
getting it wrong would silently recommend the wrong pair of finals.

Run from the repo root:
    python -m unittest tests.test_league_portfolio
"""

from __future__ import annotations

import unittest

from src.league.fitness import Cell, Fitness
from src.league.portfolio import coverage, rock_paper_scissors


def _fitness(name: str, cells: dict[str, float], games: int = 100) -> Fitness:
    fitness = Fitness(candidate=name, deck=name)
    fitness.cells = [
        Cell(opponent=opponent, wins=round(rate * games),
             losses=games - round(rate * games), draws=0, exceptions=0)
        for opponent, rate in cells.items()]
    return fitness


class TestPairCoverage(unittest.TestCase):

    def setUp(self) -> None:
        # a deliberate rock-paper-scissors, mirroring the real finding
        self.matrix = {
            "crustle": _fitness("crustle", {
                "starmie": 0.27, "alakazam": 0.85, "spidops": 0.75,
                "grimmsnarl": 0.57, "abomasnow": 0.71}),
            "abomasnow": _fitness("abomasnow", {
                "starmie": 0.52, "alakazam": 0.68, "spidops": 0.78,
                "grimmsnarl": 0.76, "crustle": 0.29}),
            "grimmsnarl": _fitness("grimmsnarl", {
                "starmie": 0.34, "alakazam": 0.77, "spidops": 0.70,
                "crustle": 0.43, "abomasnow": 0.24}),
        }

    def test_a_pair_takes_the_best_member_per_opponent(self) -> None:
        cov = coverage(["crustle", "abomasnow"], self.matrix)
        # crustle is terrible into starmie, abomasnow holds it
        self.assertAlmostEqual(cov.per_opponent["starmie"], 0.52)
        self.assertEqual(cov.best_member["starmie"], "abomasnow")
        # ... and crustle is the better alakazam answer
        self.assertAlmostEqual(cov.per_opponent["alakazam"], 0.85)
        self.assertEqual(cov.best_member["alakazam"], "crustle")

    def test_pair_members_are_not_their_own_opponents(self) -> None:
        cov = coverage(["crustle", "abomasnow"], self.matrix)
        self.assertNotIn("crustle", cov.per_opponent)
        self.assertNotIn("abomasnow", cov.per_opponent)
        self.assertIn("grimmsnarl", cov.per_opponent)

    def test_a_pair_beats_either_of_its_members_alone(self) -> None:
        """The whole reason to think in portfolios."""
        pair = coverage(["crustle", "abomasnow"], self.matrix)
        for member in ("crustle", "abomasnow"):
            self.assertGreater(pair.radar, coverage([member], self.matrix).radar)

    def test_the_pair_worst_cell_is_the_hole_neither_patches(self) -> None:
        pair = coverage(["crustle", "abomasnow"], self.matrix)
        # both are fine-ish everywhere; starmie at 52% is the floor
        self.assertEqual(pair.worst_opponent, "starmie")
        self.assertAlmostEqual(pair.worst_rate, 0.52)
        # and it is strictly better than crustle's own hole
        self.assertGreater(pair.worst_rate,
                           coverage(["crustle"], self.matrix).worst_rate)

    def test_covering_the_hole_is_what_ranks_pairs(self) -> None:
        """crustle+abomasnow patches starmie; crustle+grimmsnarl does
        not, because grimmsnarl loses to starmie too."""
        with_patch = coverage(["crustle", "abomasnow"], self.matrix)
        without = coverage(["crustle", "grimmsnarl"], self.matrix)
        self.assertGreater(with_patch.worst_rate, without.worst_rate)
        self.assertEqual(without.worst_opponent, "starmie")

    def test_complementarity_rewards_disagreement(self) -> None:
        """Two decks that answer the SAME opponents add little. The
        metric should say so."""
        clone = dict(self.matrix)
        clone["crustle_twin"] = _fitness("crustle_twin", {
            "starmie": 0.27, "alakazam": 0.85, "spidops": 0.75,
            "grimmsnarl": 0.57, "abomasnow": 0.71})
        twins = coverage(["crustle", "crustle_twin"], clone)
        mixed = coverage(["crustle", "abomasnow"], self.matrix)
        self.assertAlmostEqual(twins.complementarity, 0.0, places=6)
        self.assertGreater(mixed.complementarity, 0.1)

    def test_radar_weighting_favours_the_cells_that_matter(self) -> None:
        """alakazam is double-weighted; a pair that wins it should rank
        above one that wins an equally sized but lighter cell."""
        matrix = {
            "wins_heavy": _fitness("wins_heavy",
                                   {"alakazam": 0.90, "clefairy": 0.50}),
            "wins_light": _fitness("wins_light",
                                   {"alakazam": 0.50, "clefairy": 0.90}),
        }
        heavy = coverage(["wins_heavy"], matrix)
        light = coverage(["wins_light"], matrix)
        self.assertAlmostEqual(heavy.plain_mean, light.plain_mean)
        self.assertGreater(heavy.radar, light.radar)

    def test_empty_and_degenerate_inputs(self) -> None:
        cov = coverage(["crustle"], {"crustle": Fitness("crustle", "crustle")})
        self.assertEqual(cov.worst_opponent, "n/a")
        self.assertEqual(cov.radar, 0.0)

    def test_to_dict_is_serializable(self) -> None:
        import json
        payload = coverage(["crustle", "abomasnow"], self.matrix).to_dict()
        self.assertIsInstance(json.dumps(payload), str)
        self.assertEqual(payload["members"], ["crustle", "abomasnow"])


class TestRockPaperScissors(unittest.TestCase):

    def test_it_reports_only_the_winning_direction(self) -> None:
        matrix = {
            "a": _fitness("a", {"b": 0.71}),
            "b": _fitness("b", {"a": 0.29}),
        }
        lines = rock_paper_scissors(["a", "b"], matrix)
        self.assertEqual(len(lines), 1)
        self.assertIn("a > b", lines[0])

    def test_directional_edges_are_marked(self) -> None:
        matrix = {"a": _fitness("a", {"b": 0.53}, games=40),
                  "b": _fitness("b", {"a": 0.47}, games=40)}
        lines = rock_paper_scissors(["a", "b"], matrix)
        self.assertTrue(lines[0].startswith(" ~"), lines)

    def test_a_real_cycle_shows_all_three_edges(self) -> None:
        matrix = {
            "rock": _fitness("rock", {"scissors": 0.80, "paper": 0.20}),
            "paper": _fitness("paper", {"rock": 0.80, "scissors": 0.20}),
            "scissors": _fitness("scissors", {"paper": 0.80, "rock": 0.20}),
        }
        lines = rock_paper_scissors(["rock", "paper", "scissors"], matrix)
        self.assertEqual(len(lines), 3, lines)


if __name__ == "__main__":
    unittest.main()
