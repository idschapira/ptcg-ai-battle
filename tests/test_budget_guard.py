"""Contract: the time bank is never busted, and degradation is monotone.

Busting ``remainingOverageTime`` is not slowness, it is a TIMEOUT status
and a forfeited game — the corpus shows the top-10 team losing four
games that way. So these tests are about one property above all: the
guard must refuse to search before the bank runs out, under every input
it can be handed, including hostile ones (a missing field, a negative
number, a machine 100x slower than the one the ladder was tuned on).

The degradation ladder is checked for MONOTONICITY rather than for
specific rungs: the exact thresholds are tuning and will move, but "a
smaller bank never buys a bigger search" is the invariant that keeps the
guard safe when they do.

Run from the repo root:  python -m unittest tests.test_budget_guard
"""

from __future__ import annotations

import unittest

from src.rl_models.budget import (BANK_START_S, DEFAULT_LADDER,
                                  DEFAULT_RESERVE_S, BudgetGuard, SearchTier)


def _obs(remaining: object) -> dict:
    """Minimal observation carrying a bank reading."""
    return {"remainingOverageTime": remaining, "select": {}, "current": {}}


class TestBudgetGuard(unittest.TestCase):

    # ------------------------------------------------------------------ #
    # The reserve is never spent
    # ------------------------------------------------------------------ #

    def test_refuses_to_search_at_or_below_the_reserve(self) -> None:
        guard = BudgetGuard()
        for remaining in (DEFAULT_RESERVE_S, DEFAULT_RESERVE_S - 0.1,
                          10.0, 1.0, 0.0):
            with self.subTest(remaining=remaining):
                self.assertIsNone(guard.choose(_obs(remaining)),
                                  f"searched with {remaining}s left")
        self.assertGreater(guard.stats.skipped_reserve, 0)

    def test_a_slow_machine_degrades_instead_of_overspending(self) -> None:
        """The projection, not the ladder, is what makes this portable."""
        guard = BudgetGuard()
        # teach it that rollouts cost 30s each (a catastrophically slow box)
        for _ in range(40):
            guard.record(DEFAULT_LADDER[-1], 30.0 * DEFAULT_LADDER[-1].rollouts,
                         DEFAULT_LADDER[-1].rollouts)
        guard.reset()
        self.assertGreater(guard.rollout_cost_s, 20.0)
        # a full bank still must not buy a search it cannot pay for
        tier = guard.choose(_obs(BANK_START_S))
        if tier is not None:
            projected = tier.rollouts * guard.rollout_cost_s
            self.assertLess(projected, BANK_START_S - DEFAULT_RESERVE_S,
                            "granted a tier whose projection breaks the reserve")

    def test_repeated_spending_never_drives_the_bank_negative(self) -> None:
        """Drive the guard to exhaustion and check it stops on its own.

        Self-accounting only (no reported bank), which is the pessimistic
        path the agent falls back to when the field is missing.
        """
        guard = BudgetGuard()
        obs = {"select": {}}          # deliberately no remainingOverageTime
        searches = 0
        for _ in range(10_000):
            tier = guard.choose(obs)
            if tier is None:
                break
            searches += 1
            # every search costs 5s per rollout — far worse than measured
            guard.record(tier, 5.0 * tier.rollouts, tier.rollouts)
        self.assertGreater(searches, 0, "never searched at all")
        self.assertIsNone(guard.choose(obs), "still searching when broke")
        self.assertLessEqual(guard.spent_s, BANK_START_S - DEFAULT_RESERVE_S,
                             f"spent {guard.spent_s:.1f}s of the bank")

    # ------------------------------------------------------------------ #
    # Degradation is monotone
    # ------------------------------------------------------------------ #

    def test_smaller_bank_never_buys_a_bigger_search(self) -> None:
        guard = BudgetGuard()
        guard.record(DEFAULT_LADDER[0], 0.05 * DEFAULT_LADDER[0].rollouts,
                     DEFAULT_LADDER[0].rollouts)
        previous = None
        for remaining in range(int(BANK_START_S), 0, -10):
            tier = guard.choose(_obs(float(remaining)))
            size = tier.rollouts if tier is not None else 0
            if previous is not None:
                self.assertLessEqual(
                    size, previous,
                    f"bank {remaining}s bought {size} rollouts after "
                    f"{previous} at a larger bank")
            previous = size
        self.assertEqual(previous, 0, "still searching at an empty bank")

    def test_ladder_is_ordered_richest_first(self) -> None:
        sizes = [t.rollouts for t in DEFAULT_LADDER]
        self.assertEqual(sizes, sorted(sizes, reverse=True))
        banks = [t.min_bank_s for t in DEFAULT_LADDER]
        self.assertEqual(banks, sorted(banks, reverse=True),
                         "a cheaper tier must not need a bigger bank")

    # ------------------------------------------------------------------ #
    # Hostile / missing inputs
    # ------------------------------------------------------------------ #

    def test_malformed_bank_readings_fall_back_to_self_accounting(self) -> None:
        for value in (None, "600", float("nan"), -5.0, 10_000.0, True, [600]):
            with self.subTest(value=value):
                guard = BudgetGuard()
                bank = guard.bank_s(_obs(value))
                self.assertGreaterEqual(bank, 0.0)
                self.assertLessEqual(bank, BANK_START_S)

    def test_non_dict_observation_is_survivable(self) -> None:
        guard = BudgetGuard()
        for obs in (None, [], "obs", 42):
            with self.subTest(obs=obs):
                self.assertLessEqual(guard.bank_s(obs), BANK_START_S)

    def test_believes_the_more_pessimistic_of_the_two_sources(self) -> None:
        """A generous report must not override our own measured spend."""
        guard = BudgetGuard()
        guard.record(SearchTier("x", 1, 1, 0.0), 300.0, 1)
        # environment claims the bank is untouched; we know better
        self.assertLessEqual(guard.bank_s(_obs(BANK_START_S)),
                             BANK_START_S - 300.0 + 1e-6)
        # and the reverse: a stingy report wins over our optimism
        fresh = BudgetGuard()
        self.assertAlmostEqual(fresh.bank_s(_obs(42.0)), 42.0, places=6)

    def test_reset_clears_spend_but_keeps_the_cost_model(self) -> None:
        guard = BudgetGuard()
        guard.record(DEFAULT_LADDER[0], 8.0, DEFAULT_LADDER[0].rollouts)
        learned = guard.rollout_cost_s
        guard.reset()
        self.assertEqual(guard.spent_s, 0.0)
        self.assertAlmostEqual(guard.rollout_cost_s, learned, places=9)

    def test_untimed_spend_still_counts_against_the_bank(self) -> None:
        guard = BudgetGuard()
        guard.note_untimed_spend(100.0)
        self.assertAlmostEqual(guard.bank_s({}), BANK_START_S - 100.0,
                               places=6)


if __name__ == "__main__":
    unittest.main()
