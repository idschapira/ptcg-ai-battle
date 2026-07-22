"""Tests for the noise-proof promotion gate (Phase 3).

The gate exists because Phase 2's loop promoted two champions that were
bit-identical to the default theta. So the tests that matter most are
the ones that assert the gate REFUSES: identical arms, tiny differences,
and underpowered samples must all be held.

`tests/test_league_null_control.py` runs the same check end-to-end
against the real simulator; this file pins the maths.

Run from the repo root:
    python -m unittest tests.test_league_gate
"""

from __future__ import annotations

import random
import unittest

from src.league.gate import (Proportion, games_for_effect,
                             minimum_detectable_effect, newcombe_difference,
                             pooled, power_table, promotion_gate)


class _Cell:
    """Minimal stand-in for fitness.Cell."""

    def __init__(self, wins: int, losses: int) -> None:
        self.wins = wins
        self.losses = losses


class TestPooledStatistic(unittest.TestCase):
    """The gate's statistic: one proportion, whole-cohort sample size."""

    def test_pooled_sums_wins_and_decided(self) -> None:
        proportion = pooled([_Cell(30, 10), _Cell(20, 20), _Cell(5, 35)])
        self.assertEqual(proportion.wins, 55)
        self.assertEqual(proportion.decided, 120)
        self.assertAlmostEqual(proportion.rate, 55 / 120)

    def test_draws_are_excluded_not_counted_as_losses(self) -> None:
        self.assertEqual(pooled([_Cell(10, 10)]).decided, 20)

    def test_empty_is_safe(self) -> None:
        self.assertEqual(pooled([]).rate, 0.5)


class TestNewcombeDifference(unittest.TestCase):

    def test_identical_arms_bracket_zero(self) -> None:
        arm = Proportion(750, 1000)
        low, high = newcombe_difference(arm, arm)
        self.assertLess(low, 0.0)
        self.assertGreater(high, 0.0)

    def test_a_large_real_difference_excludes_zero(self) -> None:
        low, _ = newcombe_difference(Proportion(850, 1000),
                                     Proportion(700, 1000))
        self.assertGreater(low, 0.0)

    def test_interval_narrows_with_sample_size(self) -> None:
        small = newcombe_difference(Proportion(80, 100), Proportion(70, 100))
        large = newcombe_difference(Proportion(800, 1000),
                                    Proportion(700, 1000))
        self.assertLess(large[1] - large[0], small[1] - small[0])

    def test_it_is_less_conservative_than_non_overlap(self) -> None:
        """Requiring the two Wilson intervals not to overlap is roughly a
        99% test, not 95% — it throws away real improvements. Newcombe is
        the correct Wilson-family comparison."""
        a, b = Proportion(800, 1000), Proportion(760, 1000)
        a_low, _ = a.interval()
        _, b_high = b.interval()
        overlap = a_low <= b_high            # non-overlap test would FAIL
        low, _ = newcombe_difference(a, b)
        self.assertTrue(overlap)
        self.assertGreater(low, 0.0, "Newcombe should still detect this")

    def test_degenerate_inputs_do_not_crash(self) -> None:
        self.assertEqual(newcombe_difference(Proportion(0, 0),
                                             Proportion(5, 10)), (-1.0, 1.0))


class TestPromotionGate(unittest.TestCase):

    def test_identical_arms_are_never_promoted(self) -> None:
        """The Phase 2 failure, as a unit test."""
        arm = Proportion(750, 1000)
        result = promotion_gate(arm, arm, min_games=600)
        self.assertFalse(result.promoted)
        self.assertIn("includes zero", result.reason)

    def test_a_worse_challenger_is_never_promoted(self) -> None:
        result = promotion_gate(Proportion(700, 1000), Proportion(800, 1000),
                                min_games=600)
        self.assertFalse(result.promoted)

    def test_a_clearly_better_challenger_is_promoted(self) -> None:
        result = promotion_gate(Proportion(850, 1000), Proportion(700, 1000),
                                min_games=600)
        self.assertTrue(result.promoted)
        self.assertGreater(result.diff_low, 0.0)

    def test_underpowered_samples_are_held_even_if_ahead(self) -> None:
        """A tiny sample that happens to look great must NOT pass — this
        is exactly how the old loop started eating noise."""
        result = promotion_gate(Proportion(20, 24), Proportion(12, 24),
                                min_games=600)
        self.assertFalse(result.promoted)
        self.assertIn("underpowered", result.reason)

    def test_a_small_true_difference_is_held_at_realistic_n(self) -> None:
        """2pp at ~1100 games/arm is below the detectable floor; the gate
        must decline rather than pretend."""
        result = promotion_gate(Proportion(int(0.77 * 1100), 1100),
                                Proportion(int(0.75 * 1100), 1100),
                                min_games=600)
        self.assertFalse(result.promoted)

    def test_the_gate_result_is_auditable(self) -> None:
        result = promotion_gate(Proportion(850, 1000), Proportion(700, 1000),
                                min_games=600)
        payload = result.to_dict()
        for key in ("challenger_rate", "incumbent_rate", "delta", "diff_low",
                    "diff_high", "promoted", "reason", "min_games"):
            self.assertIn(key, payload)
        self.assertIn("PROMOTE", result.format())


class TestFalsePositiveRate(unittest.TestCase):
    """Simulated null: two arms drawn from the SAME true winrate."""

    def test_false_positive_rate_is_at_or_under_alpha(self) -> None:
        rng = random.Random(12345)
        trials, promotions, n, p = 600, 0, 1100, 0.75
        for _ in range(trials):
            a = sum(1 for _ in range(n) if rng.random() < p)
            b = sum(1 for _ in range(n) if rng.random() < p)
            if promotion_gate(Proportion(a, n), Proportion(b, n),
                              min_games=600).promoted:
                promotions += 1
        rate = promotions / trials
        self.assertLessEqual(rate, 0.05,
                             f"false-positive rate {rate:.1%} exceeds alpha")

    def test_it_still_detects_a_real_effect(self) -> None:
        """Guard against a gate that passes the null by never promoting.
        At 10pp — well above the ~4.6pp floor at this N — it must fire
        most of the time."""
        rng = random.Random(999)
        trials, promotions, n = 200, 0, 1100
        for _ in range(trials):
            a = sum(1 for _ in range(n) if rng.random() < 0.85)
            b = sum(1 for _ in range(n) if rng.random() < 0.75)
            if promotion_gate(Proportion(a, n), Proportion(b, n),
                              min_games=600).promoted:
                promotions += 1
        self.assertGreater(promotions / trials, 0.90)


class TestPowerPlanning(unittest.TestCase):

    def test_minimum_detectable_effect_shrinks_with_n(self) -> None:
        self.assertGreater(minimum_detectable_effect(264),
                           minimum_detectable_effect(1100))

    def test_the_old_budget_could_not_see_what_it_promoted(self) -> None:
        """Phase 2 ran 24 games/cell x 11 cells = 264 decided games and
        promoted on deltas of a few points. The floor there was ~9pp."""
        self.assertGreater(minimum_detectable_effect(264), 0.08)

    def test_the_new_budget_floor_is_reported_honestly(self) -> None:
        floor = minimum_detectable_effect(1100)
        self.assertLess(floor, 0.05)
        self.assertGreater(floor, 0.04, "no free lunch: still ~4-5pp")

    def test_games_for_effect_round_trips(self) -> None:
        for effect in (0.03, 0.05, 0.10):
            n = games_for_effect(effect)
            self.assertLessEqual(minimum_detectable_effect(n), effect + 1e-3)

    def test_power_table_is_monotone(self) -> None:
        rows = power_table(11)
        effects = [effect for _, _, effect in rows]
        self.assertEqual(effects, sorted(effects, reverse=True))


if __name__ == "__main__":
    unittest.main()
