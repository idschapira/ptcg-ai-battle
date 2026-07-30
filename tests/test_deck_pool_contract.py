"""Every deck file we keep must be 60 legal cards the ENGINE accepts.

data/decks/ is swept automatically by the gauntlet and by ab_test, and
decks also arrive there from reconstruction (mine_opponent_deck.py builds
a list out of observed replays and infers the energy count). A bad file
therefore fails somewhere far away — mid-run, as an exception inside a
worker — instead of here.

Two oracles, because our copy of the rules is not the rules:
  legality.py    our reading of deck construction, and
  battle_start   the engine's, which is the one that counts.

The candidate decks (variant_*.csv) are included on purpose: they are
excluded from the FIELD, not from the correctness bar.

Run from the repo root:  python -m unittest tests.test_deck_pool_contract
"""

from __future__ import annotations

import unittest

from cg import game

from src.deckbuilding.gauntlet import discover_decks
from src.deckbuilding.legality import read_deck_ids, validate_deck
from src.ingestion.card_index import CardIndex

DECK_SIZE = 60


class TestDeckPoolContract(unittest.TestCase):
    index: CardIndex

    @classmethod
    def setUpClass(cls) -> None:
        cls.index = CardIndex()
        cls.decks = discover_decks(include_candidates=True)

    def test_pool_is_not_empty(self) -> None:
        self.assertGreater(len(self.decks), 5,
                           "deck pool vanished — the sweep found almost "
                           "nothing, so the per-deck checks below would "
                           "pass vacuously")

    def test_every_deck_is_sixty_legal_cards(self) -> None:
        for name, path in sorted(self.decks.items()):
            with self.subTest(deck=name):
                ids = read_deck_ids(path)
                self.assertEqual(len(ids), DECK_SIZE,
                                 f"{path.name} tem {len(ids)} cartas")
                report = validate_deck(ids, self.index)
                self.assertTrue(report.ok,
                                f"{path.name} ilegal: {report.errors}")

    def test_every_deck_starts_a_battle(self) -> None:
        """The engine is the authority; legality.py is our reading of it."""
        for name, path in sorted(self.decks.items()):
            with self.subTest(deck=name):
                ids = read_deck_ids(path)
                obs, start = game.battle_start(list(ids), list(ids))
                try:
                    self.assertIsNotNone(
                        obs, f"{path.name}: engine recusou "
                             f"(errorType={start.errorType})")
                finally:
                    game.battle_finish()

    def test_field_excludes_candidate_variants(self) -> None:
        """A field holding copies of our own list is not the field."""
        field = discover_decks()
        self.assertFalse(
            [n for n in field if n.startswith("variant_")],
            "variant_*.csv leaked into the auto-discovered field — every "
            "field average would silently be reweighted toward mirrors")
        self.assertIn("meta_archaludon", field,
                      "the Archaludon cell must stay in the field: it is "
                      "14.2% of real games and was untested until it had "
                      "a decklist")


if __name__ == "__main__":
    unittest.main()
