"""Parse kaggle-environments replays into (state, action) training pairs.

Uses the Sprint-4A encoders from src/rl_models/encoding.py — imported,
never reimplemented — so replay-derived samples are bit-compatible with
anything encoded live. Output feeds the Sprint-5 supervised warm start
(the training itself does NOT live here).

Dataset layout (all np.savez_compressed):
    states        [N, ENCODING_DIM] float32
    options_flat  [sum(counts), OPTION_DIM] float32  (legal options only)
    option_counts [N] uint16   (reconstruct ragged rows / masks)
    labels        [N] uint16   (chosen option index)
    values        [N] int8     (z from the decider's view: +1 win/-1 loss/0)
    episode_ids   [N] int64
plus a sibling .meta.json with provenance and the validation counters.

Default output: data/processed/replay_dataset.npz (unchanged). With
--date D the input becomes data/raw/replays/D and the output
data/processed/replays/replay_dataset_D.npz — one file per day, nothing
overwritten, ready for the multi-day merge (replays_merge.py).

None-safe reconciliation: any card/attack id in a decision point that the
CardIndex does not know skips that sample (counted per id), never raises.

Usage (repo root):
    python -m src.ingestion.replays_parse --sides winner --emit-viewer 1
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Iterator

import numpy as np

from cg.api import Observation, OptionType, to_observation_class

from ..environment_wrapper.recorder import GameRecorder
from ..rl_models import encoding
from ..rl_models.encoding import (
    ENCODING_DIM,
    MAX_OPTIONS,
    OPTION_DIM,
    OptionEncoder,
    StateEncoder,
)
from .build_card_model import PROCESSED_DIR, REPO_ROOT
from .card_index import CardIndex
from .replays_download import REPLAYS_DIR

logger = logging.getLogger(__name__)

DATASET_PATH: Final[Path] = PROCESSED_DIR / "replay_dataset.npz"
META_PATH: Final[Path] = PROCESSED_DIR / "replay_dataset.meta.json"
REPLAYS_OUT_DIR: Final[Path] = PROCESSED_DIR / "replays"
VIEWER_OUT_DIR: Final[Path] = REPO_ROOT / "viewer" / "replays"


@dataclass
class ParseStats:
    games: int = 0
    decision_pairs: int = 0
    skipped_unknown_id: int = 0
    skipped_no_action: int = 0
    skipped_side: int = 0
    unknown_ids: Counter = field(default_factory=Counter)
    overflow_lens: list[int] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        seen = self.decision_pairs + self.skipped_unknown_id
        return self.decision_pairs / seen if seen else 1.0


class _OverflowCounter(logging.Handler):
    """Counts the MAX_OPTIONS overflow warnings emitted by encoding.py."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        if "option overflow" in record.getMessage():
            self.count += 1


def _engine_obs(observation: dict) -> dict | None:
    """Strip kaggle-injected keys down to the engine observation dict."""
    if not isinstance(observation, dict) or observation.get("select") is None:
        return None  # not a decision point for this agent (or deck selection)
    return {"select": observation.get("select"),
            "logs": observation.get("logs") or [],
            "current": observation.get("current")}


def _iter_decisions(replay: dict,
                    pairing: str = "prev") -> Iterator[tuple[int, dict, list[int]]]:
    """Yield (agent_index, engine_obs_dict, action) per decision point.

    kaggle-environments pairing (verified on real 2026-07-05 episodes):
    steps[t][agent].action answers steps[t-1][agent].observation — the
    action is stored one step AFTER the observation it responded to.
    The deck-selection action pairs with an observation whose select is
    None, so it drops out here without counting as a decision.

    `pairing="same"` (action paired with its OWN step's observation — the
    pre-fix bug) exists only so tests/test_replay_pairing.py can assert
    that the wrong convention is detectably wrong. Production callers must
    use the default.
    """
    shift = 1 if pairing == "prev" else 0
    steps = replay.get("steps") or []
    for step_index in range(shift, len(steps)):
        prev_pair, step_pair = steps[step_index - shift], steps[step_index]
        if not isinstance(prev_pair, list) or not isinstance(step_pair, list):
            continue
        for agent_index in range(min(2, len(prev_pair), len(step_pair))):
            prev_entry, entry = prev_pair[agent_index], step_pair[agent_index]
            if not isinstance(prev_entry, dict) or not isinstance(entry, dict):
                continue
            obs_dict = _engine_obs(prev_entry.get("observation") or {})
            action = entry.get("action")
            if obs_dict is None or not isinstance(action, list) or not action:
                continue
            if not all(isinstance(a, int) for a in action):
                continue
            yield agent_index, obs_dict, action


def _referenced_ids(index: CardIndex, obs: Observation) -> tuple[set[int], set[int]]:
    """Card and attack ids a decision point references (state + options)."""
    card_ids: set[int] = set()
    attack_ids: set[int] = set()
    state = obs.current
    if state is not None:
        for player in state.players:
            for pokemon in list(player.active or []) + list(player.bench or []):
                if pokemon is not None:
                    card_ids.add(pokemon.id)
            for card in player.hand or []:
                if card is not None:
                    card_ids.add(card.id)
    if obs.select is not None:
        for option in obs.select.option:
            if option.attackId is not None and option.type == OptionType.ATTACK:
                attack_ids.add(option.attackId)
            if option.cardId is not None and option.cardId != 0:
                card_ids.add(option.cardId)
    return card_ids, attack_ids


def _unknown_ids(index: CardIndex, obs: Observation) -> list[int]:
    card_ids, attack_ids = _referenced_ids(index, obs)
    unknown = [cid for cid in card_ids if index.get_card(cid) is None]
    unknown += [aid for aid in attack_ids if index.get_attack(aid) is None]
    return unknown


def parse_replays(
    replay_dir: Path = REPLAYS_DIR,
    sides: str = "winner",
    emit_viewer: int = 0,
    index: CardIndex | None = None,
    out_path: Path | None = None,
    team: str | None = None,
) -> ParseStats:
    """`team` (dev tooling, behavior cloning de um líder): mantém apenas
    as decisões do agente cujo TeamNames == team (as duas metades win/
    loss dele — z continua por resultado), ignorando o filtro `sides`;
    jogos sem esse time são pulados inteiros."""
    # resolved late so tests can patch the module-level default
    out_path = out_path if out_path is not None else DATASET_PATH
    meta_path = out_path.with_name(out_path.name.replace(".npz", ".meta.json"))
    index = index if index is not None else CardIndex()
    state_encoder = StateEncoder(index)
    option_encoder = OptionEncoder(index)

    overflow_counter = _OverflowCounter()
    encoding.logger.addHandler(overflow_counter)

    states: list[np.ndarray] = []
    options_flat: list[np.ndarray] = []
    option_counts: list[int] = []
    labels: list[int] = []
    values: list[int] = []
    episode_ids: list[int] = []
    stats = ParseStats()

    replay_files = sorted(p for p in replay_dir.rglob("*.json")
                          if p.name != "manifest.csv" and "_index" not in p.parts)
    for replay_path in replay_files:
        try:
            with open(replay_path, encoding="utf-8") as fh:
                replay = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("unreadable replay %s: %s", replay_path.name, exc)
            continue
        rewards = replay.get("rewards") or []
        # None-safe: an errored agent leaves reward None (seen on real
        # 2026-07 episodes) — treat as no decided winner, never argmax
        winner = (int(np.argmax(rewards))
                  if (len(rewards) == 2
                      and all(isinstance(r, (int, float)) for r in rewards)
                      and rewards[0] != rewards[1]) else None)
        episode_id = (replay.get("info") or {}).get("EpisodeId") or 0
        team_index: int | None = None
        if team is not None:
            names = (replay.get("info") or {}).get("TeamNames") or []
            team_index = names.index(team) if team in names else None
            if team_index is None:
                continue  # game without the cloned leader
        stats.games += 1

        recorder = (GameRecorder(index, tuple((replay.get("info") or {})
                                              .get("TeamNames") or ("p0", "p1")))
                    if emit_viewer > 0 else None)

        for agent_index, obs_dict, action in _iter_decisions(replay):
            if team_index is not None:
                if agent_index != team_index:
                    stats.skipped_side += 1
                    continue
            elif sides == "winner" and winner is not None and agent_index != winner:
                stats.skipped_side += 1
                continue
            try:
                obs = to_observation_class(obs_dict)
            except Exception as exc:
                stats.skipped_no_action += 1
                logger.warning("unparseable observation in %s: %s",
                               replay_path.name, exc)
                continue
            unknown = _unknown_ids(index, obs)
            if unknown:
                stats.skipped_unknown_id += 1
                stats.unknown_ids.update(unknown)
                logger.warning("skipping sample in %s: unknown ids %s",
                               replay_path.name, unknown[:5])
                continue

            n_options = len(obs.select.option) if obs.select else 0
            if n_options > MAX_OPTIONS:
                stats.overflow_lens.append(n_options)
            if any(a >= min(n_options, MAX_OPTIONS) or a < 0 for a in action):
                stats.skipped_no_action += 1  # chosen index beyond cap/range
                continue

            state_vec = state_encoder.encode(obs)
            option_matrix, mask = encoding.build_action_mask(
                obs, state_encoder, option_encoder)
            legal = option_matrix[mask]
            # one supervised pair per chosen index (multi-select answers
            # contribute one sample each, same state)
            z = 0 if winner is None else (1 if agent_index == winner else -1)
            for chosen in action:
                states.append(state_vec)
                options_flat.append(legal)
                option_counts.append(legal.shape[0])
                labels.append(chosen)
                values.append(z)
                episode_ids.append(int(episode_id))
                stats.decision_pairs += 1

            if recorder is not None:
                recorder.record_step(obs_dict, action, None)

        if recorder is not None:
            final_turn = 0
            for _agent, obs_dict, _action in _iter_decisions(replay):
                final_turn = max(final_turn,
                                 (obs_dict.get("current") or {}).get("turn") or 0)
            VIEWER_OUT_DIR.mkdir(parents=True, exist_ok=True)
            recorder.save(VIEWER_OUT_DIR / f"replay_{episode_id}.json",
                          winner if winner is not None else 2, final_turn)
            emit_viewer -= 1

    encoding.logger.removeHandler(overflow_counter)
    stats.overflow_warnings = overflow_counter.count  # type: ignore[attr-defined]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        states=np.asarray(states, dtype=np.float32).reshape(-1, ENCODING_DIM),
        options_flat=(np.concatenate(options_flat, axis=0)
                      if options_flat else np.zeros((0, OPTION_DIM), np.float32)),
        option_counts=np.asarray(option_counts, dtype=np.uint16),
        labels=np.asarray(labels, dtype=np.uint16),
        values=np.asarray(values, dtype=np.int8),
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
    )
    meta = {
        "schema": "ptcg-replay-dataset-v2",  # v2: + values (z per sample)
        "encoding_dim": ENCODING_DIM,
        "option_dim": OPTION_DIM,
        "max_options": MAX_OPTIONS,
        "sides": sides,
        "games": stats.games,
        "decision_pairs": stats.decision_pairs,
        "skipped_unknown_id": stats.skipped_unknown_id,
        "skipped_no_action": stats.skipped_no_action,
        "skipped_other_side": stats.skipped_side,
        "coverage": stats.coverage,
        "unknown_ids": dict(stats.unknown_ids.most_common(50)),
        "overflow_warnings": overflow_counter.count,
        "overflow_lens": stats.overflow_lens,
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    return stats


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=None, metavar="YYYY-MM-DD",
                        help="parse data/raw/replays/DATE into "
                             "data/processed/replays/replay_dataset_DATE.npz")
    parser.add_argument("--replay-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--sides", choices=("winner", "both"), default="winner")
    parser.add_argument("--team", default=None,
                        help="behavior cloning: só decisões deste TeamNames "
                             "(ignora --sides; jogos sem o time são pulados)")
    parser.add_argument("--emit-viewer", type=int, default=1, metavar="N",
                        help="also write N games as ptcg-devrecord-v1 JSON")
    args = parser.parse_args()

    replay_dir = args.replay_dir
    out_path = args.out
    if args.date is not None:
        replay_dir = replay_dir if replay_dir is not None else REPLAYS_DIR / args.date
        out_path = (out_path if out_path is not None
                    else REPLAYS_OUT_DIR / f"replay_dataset_{args.date}.npz")
    if replay_dir is None:
        replay_dir = REPLAYS_DIR

    stats = parse_replays(replay_dir, args.sides, args.emit_viewer,
                          out_path=out_path, team=args.team)
    print(f"games parsed:          {stats.games}")
    print(f"decision pairs:        {stats.decision_pairs}")
    print(f"skipped (unknown id):  {stats.skipped_unknown_id}")
    print(f"skipped (bad action):  {stats.skipped_no_action}")
    print(f"skipped (other side):  {stats.skipped_side}")
    print(f"id coverage:           {stats.coverage:.4%}")
    if stats.unknown_ids:
        print(f"unknown ids (top):     {dict(stats.unknown_ids.most_common(10))}")
    print(f"MAX_OPTIONS overflows: {getattr(stats, 'overflow_warnings', 0)}"
          f"{' lens=' + str(stats.overflow_lens) if stats.overflow_lens else ''}")
    written = out_path if out_path is not None else DATASET_PATH
    print(f"dataset: {written.relative_to(REPO_ROOT)} "
          f"({written.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
