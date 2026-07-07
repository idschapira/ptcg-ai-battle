"""Regression tests for the replay action<->observation step pairing.

Locks in the kaggle-environments convention verified on real 2026-07-05
episodes: steps[t][agent].action answers steps[t-1][agent].observation.
The pre-fix parser paired action and observation of the SAME step, which
produced silently wrong labels (on real data: 251 actions fit only the
prev-step options, 0 fit only the same-step options).

Deterministic: no network, no credentials — replays come from
generate_sample_replays() into a tmpdir that is removed afterwards. The
real-replay check runs only if data/raw/replays already holds downloaded
episodes, and skips (never fails) otherwise.

Run from the repo root:  python -m unittest tests.test_replay_pairing
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.ingestion.replays_download import REPLAYS_DIR, generate_sample_replays
from src.ingestion.replays_parse import _iter_decisions, parse_replays

N_SAMPLE_GAMES = 5
SEED = 123


def _option_count(obs_dict: dict) -> int:
    select = obs_dict.get("select") or {}
    return len(select.get("option") or [])


def _pairing_counts(replays: list[dict], pairing: str) -> tuple[int, int]:
    """(decisions, invalid) — invalid = some chosen index outside the
    option list of the observation that `pairing` matched it with."""
    decisions = invalid = 0
    for replay in replays:
        for _agent, obs_dict, action in _iter_decisions(replay, pairing=pairing):
            decisions += 1
            n_options = _option_count(obs_dict)
            if any(a < 0 or a >= n_options for a in action):
                invalid += 1
    return decisions, invalid


class TestReplayPairingSampleGames(unittest.TestCase):
    """Self-play sample replays (same schema as real episodes), tmpdir-only."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.replay_dir = Path(cls._tmp.name)
        generate_sample_replays(N_SAMPLE_GAMES, dest=cls.replay_dir, seed=SEED)
        cls.replays = [json.loads(path.read_text(encoding="utf-8"))
                       for path in sorted(cls.replay_dir.glob("*.json"))]

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_prev_step_pairing_all_labels_valid(self) -> None:
        decisions, invalid = _pairing_counts(self.replays, pairing="prev")
        self.assertGreater(decisions, 0)
        self.assertEqual(invalid, 0,
                         "prev-step pairing must place every chosen index "
                         "inside the previous step's option list")

    def test_same_step_pairing_is_detectably_wrong(self) -> None:
        # The actual regression guard: if a refactor silently reverts to
        # same-step pairing, this invariant (wrong convention => invalid
        # actions exist) is what would have caught it.
        _decisions, invalid = _pairing_counts(self.replays, pairing="same")
        self.assertGreater(invalid, 0,
                           "same-step pairing (the pre-fix bug) must produce "
                           "out-of-range actions on these replays")

    def test_default_pairing_is_prev_step(self) -> None:
        default = [(a, json.dumps(o, sort_keys=True), tuple(act))
                   for a, o, act in _iter_decisions(self.replays[0])]
        explicit = [(a, json.dumps(o, sort_keys=True), tuple(act))
                    for a, o, act in _iter_decisions(self.replays[0],
                                                     pairing="prev")]
        self.assertEqual(default, explicit)

    def test_parse_replays_end_to_end_zero_bad_actions(self) -> None:
        out_dir = self.replay_dir / "_out"
        out_dir.mkdir(exist_ok=True)
        with mock.patch("src.ingestion.replays_parse.DATASET_PATH",
                        out_dir / "dataset.npz"), \
             mock.patch("src.ingestion.replays_parse.META_PATH",
                        out_dir / "dataset.meta.json"):
            stats = parse_replays(self.replay_dir, sides="both", emit_viewer=0)
        self.assertEqual(stats.games, N_SAMPLE_GAMES)
        self.assertGreater(stats.decision_pairs, 0)
        self.assertEqual(stats.skipped_no_action, 0)
        self.assertEqual(stats.skipped_unknown_id, 0)
        self.assertEqual(stats.coverage, 1.0)


class TestReplayPairingRealEpisodes(unittest.TestCase):
    """Same invariant against downloaded episodes, if any are on disk."""

    MAX_FILES = 50

    def test_real_replays_prev_pairing_all_labels_valid(self) -> None:
        files = ([path for path in sorted(REPLAYS_DIR.rglob("*.json"))
                  if "_index" not in path.parts and "sample" not in path.parts]
                 if REPLAYS_DIR.exists() else [])
        if not files:
            self.skipTest("no downloaded replays under data/raw/replays "
                          "(python -m src.ingestion.replays_download)")
        replays = []
        for path in files[:self.MAX_FILES]:
            try:
                replays.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue  # unreadable file is the parser's concern, not ours
        decisions, invalid = _pairing_counts(replays, pairing="prev")
        self.assertGreater(decisions, 0)
        self.assertEqual(invalid, 0)


if __name__ == "__main__":
    unittest.main()
