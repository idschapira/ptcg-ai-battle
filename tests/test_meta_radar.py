"""Unit tests for the meta-radar deck miner (synthetic replay, no corpus).

Covers the three load-bearing behaviors:
  - observed-deck extraction dedupes by (playerIndex, serial) across
    steps and walks nested structures (hand, active stack, discard);
  - archetype labeling matches core-card rules apostrophe-insensitively
    and falls back to "unknown";
  - our own team is flagged (is_ours) so it can be split from the
    leader distribution.

Run from the repo root:  python -m unittest tests.test_meta_radar
"""

from __future__ import annotations

import unittest

from src.analysis.meta_radar import extract_decks, label_archetype
from src.ingestion.card_index import CardIndex

CRUSTLE = 345
DWEBBLE = 344
MEGA_KANGASKHAN = 756


def _card(card_id: int, player: int, serial: int) -> dict:
    return {"id": card_id, "playerIndex": player, "serial": serial}


def _replay() -> dict:
    # p0: Crustle-line cards seen in hand then (same serials) in discard;
    # p1: Mega Kangaskhan seen on the active stack. serial 999 is a
    # repeat sighting of serial 1's card and must not double-count.
    step_a = [{"observation": {"current": {"players": [
        {"hand": [_card(DWEBBLE, 0, 1), _card(CRUSTLE, 0, 2)]},
        {"hand": None},
    ]}}}, {}]
    step_b = [{"observation": {"current": {"players": [
        {"discard": [_card(DWEBBLE, 0, 1)], "hand": [_card(CRUSTLE, 0, 3)]},
        {"active": [{"cards": [_card(MEGA_KANGASKHAN, 1, 61)]}]},
    ]}}}, {}]
    return {"info": {"EpisodeId": 1, "TeamNames": ["Ilan Schapira", "rival"]},
            "steps": [step_a, step_b]}


class TestMetaRadar(unittest.TestCase):
    index: CardIndex

    @classmethod
    def setUpClass(cls) -> None:
        cls.index = CardIndex()

    def test_extract_decks_dedupes_serials_and_flags_ours(self) -> None:
        decks = extract_decks(_replay(), "2026-07-10", self.index)
        self.assertEqual(len(decks), 2)
        ours, theirs = decks
        self.assertTrue(ours.is_ours)
        self.assertFalse(theirs.is_ours)
        # serial 1 seen twice (hand then discard) counts once
        self.assertEqual(ours.copies_by_name.get("Dwebble"), 1)
        self.assertEqual(ours.copies_by_name.get("Crustle"), 2)
        self.assertEqual(ours.n_observed, 3)
        # nested active stack is walked
        self.assertEqual(theirs.copies_by_name.get("Mega Kangaskhan ex"), 1)

    def test_labeling_rules(self) -> None:
        self.assertEqual(label_archetype(["Alakazam", "Kadabra", "Abra"]),
                         "Alakazam box (non-ex)")
        # apostrophe-insensitive: engine uses the straight quote here
        self.assertEqual(label_archetype(["Team Rocket’s Spidops"]),
                         "Team Rocket Spidops (non-ex)")
        self.assertEqual(
            label_archetype(["Crustle", "Mega Kangaskhan ex"]),
            "Crustle + Mega Kangaskhan stall")
        self.assertEqual(label_archetype(["Crustle", "Great Tusk"]),
                         "Crustle mill (ours)")
        self.assertEqual(label_archetype(["Crustle"]),
                         "Crustle stall (other)")
        self.assertEqual(label_archetype(["Pikachu"]), "unknown")

    def test_unknown_ids_are_none_safe(self) -> None:
        replay = _replay()
        replay["steps"][0][0]["observation"]["current"]["players"][0][
            "hand"].append(_card(999_999, 0, 9))
        decks = extract_decks(replay, "2026-07-10", self.index)
        self.assertEqual(decks[0].unknown_ids, (999_999,))


if __name__ == "__main__":
    unittest.main()
