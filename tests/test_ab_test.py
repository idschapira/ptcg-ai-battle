"""Unit tests for the A/B harness statistics layer + a short live smoke.

The statistics are pure functions checked against known reference
values (Wilson 95% for 50/100 is the textbook [0.4038, 0.5962]); the
verdict discipline (PASS/HOLD/FAIL) is pinned with the real historical
sample sizes we decide on (200-game A/Bs). The smoke runs a few real
engine games through the harness to lock the contract (0 exceptions).

Run from the repo root:  python -m unittest tests.test_ab_test
"""

from __future__ import annotations

import unittest

from src.environment_wrapper.ab_test import (ArmSpec, binomial_p_value,
                                             compare, verdict,
                                             wilson_interval)


class TestWilsonInterval(unittest.TestCase):

    def test_textbook_half_split(self) -> None:
        lo, hi = wilson_interval(50, 100)
        self.assertAlmostEqual(lo, 0.4038, places=4)
        self.assertAlmostEqual(hi, 0.5962, places=4)

    def test_zero_n_is_uninformative(self) -> None:
        self.assertEqual(wilson_interval(0, 0), (0.0, 1.0))

    def test_extremes_stay_bounded(self) -> None:
        lo, hi = wilson_interval(0, 20)
        self.assertEqual(lo, 0.0)
        self.assertLess(hi, 0.2)
        lo, hi = wilson_interval(20, 20)
        self.assertGreater(lo, 0.8)
        self.assertEqual(hi, 1.0)

    def test_narrows_with_n(self) -> None:
        lo1, hi1 = wilson_interval(60, 100)
        lo2, hi2 = wilson_interval(600, 1000)
        self.assertLess(hi2 - lo2, hi1 - lo1)


class TestVerdict(unittest.TestCase):
    """Pinned to the real decisions this repo made by hand."""

    def test_v2_ab_would_pass(self) -> None:
        # v2 vs v1: 154/200 -> CI low ~70.5% > 50%
        self.assertEqual(verdict(154, 200, bar=0.5), "PASS")

    def test_first_gate_c_would_hold(self) -> None:
        # (Crustle,H) vs (Abomasnow,5D): 99/200 straddles 50%
        self.assertEqual(verdict(99, 200, bar=0.5), "HOLD")

    def test_clear_loss_fails(self) -> None:
        self.assertEqual(verdict(20, 200, bar=0.5), "FAIL")

    def test_bar_55_demands_more(self) -> None:
        # 53% over 200 games cannot clear a 55% bar
        self.assertEqual(verdict(106, 200, bar=0.55), "HOLD")
        # 77% over 200 games clears it
        self.assertEqual(verdict(154, 200, bar=0.55), "PASS")


class TestPValue(unittest.TestCase):

    def test_exact_null_is_one(self) -> None:
        self.assertAlmostEqual(binomial_p_value(100, 200, 0.5), 1.0, places=6)

    def test_strong_effect_is_tiny(self) -> None:
        self.assertLess(binomial_p_value(154, 200, 0.5), 1e-6)

    def test_degenerate_inputs_are_safe(self) -> None:
        self.assertEqual(binomial_p_value(0, 0, 0.5), 1.0)
        self.assertEqual(binomial_p_value(5, 10, 0.0), 1.0)


class TestArmSpec(unittest.TestCase):

    def test_parse_kind_only(self) -> None:
        spec = ArmSpec.parse("crustle-v2")
        self.assertEqual((spec.kind, spec.weights, spec.stats),
                         ("crustle-v2", None, None))

    def test_parse_with_overrides(self) -> None:
        spec = ArmSpec.parse("network,models/w.npz,models/s.npz")
        self.assertEqual(spec.kind, "network")
        self.assertEqual(spec.weights.name, "w.npz")
        self.assertEqual(spec.stats.name, "s.npz")

    def test_unknown_kind_aborts(self) -> None:
        with self.assertRaises(SystemExit):
            ArmSpec.parse("mcts")


class TestHarnessSmoke(unittest.TestCase):
    """A few real engine games through the full pipeline: contract only."""

    def test_agent_mode_smoke(self) -> None:
        from src.deckbuilding.legality import read_deck_ids
        from src.ingestion.build_card_model import REPO_ROOT
        from src.ingestion.build_effect_model import EffectIndex
        from src.ingestion.card_index import CardIndex

        index, effects = CardIndex(), EffectIndex()
        deck = read_deck_ids(REPO_ROOT / "data" / "decks" / "seed_crustle.csv")
        c = compare("smoke", ArmSpec.parse("heuristic"),
                    ArmSpec.parse("random"), deck, deck,
                    n_games=6, seed=11, bar=0.5, index=index, effects=effects)
        self.assertEqual(len(c.pair.errors), 0, c.pair.errors)
        self.assertEqual(c.pair.games, 6)
        self.assertGreater(c.a_metrics.calls + c.b_metrics.calls, 0)
        self.assertIn(c.verdict, ("PASS", "HOLD", "FAIL"))
        lo, hi = c.ci
        self.assertLessEqual(lo, c.pair.winrate_a)
        self.assertGreaterEqual(hi, c.pair.winrate_a)


if __name__ == "__main__":
    unittest.main()
