"""Unit tests for CardIndex fail-safe lookups.

Run from the repo root:  python -m unittest discover tests
"""

from __future__ import annotations

import unittest

from src.ingestion.card_index import Attack, Card, CardIndex


class TestCardIndexFailSafeLookups(unittest.TestCase):
    index: CardIndex

    @classmethod
    def setUpClass(cls) -> None:
        cls.index = CardIndex()

    def test_get_card_unknown_id_returns_none(self) -> None:
        self.assertIsNone(self.index.get_card(0))
        self.assertIsNone(self.index.get_card(999_999))
        self.assertIsNone(self.index.get_card(-1))

    def test_get_attack_unknown_id_returns_none(self) -> None:
        self.assertIsNone(self.index.get_attack(0))
        self.assertIsNone(self.index.get_attack(999_999))

    def test_get_skill_unknown_id_returns_none(self) -> None:
        self.assertIsNone(self.index.get_skill(999_999))

    def test_get_card_known_id_returns_card(self) -> None:
        card = self.index.get_card(1)
        self.assertIsInstance(card, Card)
        assert card is not None
        self.assertEqual(card.card_id, 1)

    def test_get_attack_known_id_returns_attack(self) -> None:
        attack = self.index.get_attack(1)
        self.assertIsInstance(attack, Attack)
        assert attack is not None
        self.assertEqual(attack.attack_id, 1)

    def test_strict_lookup_still_raises(self) -> None:
        with self.assertRaises(KeyError):
            self.index.card(999_999)
        with self.assertRaises(KeyError):
            self.index.attack(999_999)

    def test_attacks_of_unknown_card_returns_empty(self) -> None:
        self.assertEqual(self.index.attacks_of(999_999), ())

    def test_skills_of_unknown_card_returns_empty(self) -> None:
        self.assertEqual(self.index.skills_of(999_999), ())


if __name__ == "__main__":
    unittest.main()
