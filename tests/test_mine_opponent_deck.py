"""Reconstruction must key on CARD ID, never on card name.

The pool has name collisions and they are not cosmetic. "Alakazam" is
both id 743 (Powerful Hand — the entire win condition of the Alakazam
box) and id 245 (Strange Hacking / Psychic, an unrelated card). An
earlier version of mine_opponent_deck aggregated observations by name and
then resolved each name back to "the first card id that has it", which
silently rebuilt the archetype around a card that cannot execute its own
game plan — and the gauntlet cell would then have measured a deck that
does not exist anywhere.

These tests are hermetic: they drive reconstruct()/to_ids() with
synthetic observations instead of replays, so they run without the
(gitignored) episode corpus and cannot go green just because a corpus
happens to be absent.

Run from the repo root:  python -m unittest tests.test_mine_opponent_deck
"""

from __future__ import annotations

import unittest
from collections import Counter

import cg.api as api

from src.analysis.mine_opponent_deck import _mode, reconstruct, to_ids
from src.ingestion.card_index import CardIndex

ALAKAZAM_POWERFUL_HAND = 743   # the real one
ALAKAZAM_OTHER = 245           # same NAME, different card
POWERFUL_HAND = 1072


class TestNameCollisionIsNotCollapsed(unittest.TestCase):
    index: CardIndex

    @classmethod
    def setUpClass(cls) -> None:
        cls.index = CardIndex()

    def test_the_collision_this_guards_still_exists(self) -> None:
        """If the pool ever stops colliding, this test is vacuous."""
        names = {}
        for card in api.all_card_data():
            names.setdefault(card.name, []).append(card.cardId)
        self.assertIn(ALAKAZAM_POWERFUL_HAND, names.get("Alakazam", []))
        self.assertIn(ALAKAZAM_OTHER, names.get("Alakazam", []))
        attacks = {c.cardId: (c.attacks or [])
                   for c in api.all_card_data()}
        self.assertIn(POWERFUL_HAND, attacks[ALAKAZAM_POWERFUL_HAND])
        self.assertNotIn(POWERFUL_HAND, attacks[ALAKAZAM_OTHER],
                         "the two printings must remain distinguishable "
                         "by attack, else this guard proves nothing")

    def test_reconstruct_keeps_the_observed_printing(self) -> None:
        per_game = [Counter({ALAKAZAM_POWERFUL_HAND: 4}) for _ in range(10)]
        kept, _dropped = reconstruct(per_game, self.index, min_presence=0.5)
        self.assertEqual(kept.get(ALAKAZAM_POWERFUL_HAND), 4)
        self.assertNotIn(ALAKAZAM_OTHER, kept,
                         "reconstruction invented a printing that was "
                         "never observed — it is keying on the name")

    def test_to_ids_does_not_resolve_through_names(self) -> None:
        ids = to_ids(Counter({ALAKAZAM_POWERFUL_HAND: 2}))
        self.assertEqual(ids, [ALAKAZAM_POWERFUL_HAND] * 2)

    def test_rare_printing_below_presence_is_dropped(self) -> None:
        """Tech seen once in 10 games is not part of the archetype."""
        per_game = [Counter({ALAKAZAM_POWERFUL_HAND: 4}) for _ in range(10)]
        per_game[0][ALAKAZAM_OTHER] = 1
        kept, dropped = reconstruct(per_game, self.index, min_presence=0.5)
        self.assertNotIn(ALAKAZAM_OTHER, kept)
        self.assertTrue(any(str(ALAKAZAM_OTHER) in d for d in dropped))


class TestModeChoice(unittest.TestCase):
    def test_mode_not_max(self) -> None:
        """One lucky game must not set the count for the whole archetype."""
        self.assertEqual(_mode([3, 3, 3, 3, 4]), 3)

    def test_ties_break_high(self) -> None:
        self.assertEqual(_mode([2, 2, 3, 3]), 3)


if __name__ == "__main__":
    unittest.main()
