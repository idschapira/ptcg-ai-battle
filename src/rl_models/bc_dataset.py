"""Behavioral-cloning dataset: heuristic self-play decisions -> npz.

Plays heuristic×heuristic and heuristic×random games (both seatings) and
records every HEURISTIC decision: encoded state, encoded legal options,
label = index the heuristic chose, and z = final game result from the
decider's perspective (+1 win / -1 loss / 0 draw) for the value head.
Random decisions are never labels — random×random games only feed the
normalization corpus (src/rl_models/normalization.py), not this dataset.

Same ragged layout as the Sprint-4C replay dataset, plus values:
    states        [N, ENCODING_DIM] float32   (RAW — normalize at train time)
    options_flat  [sum(counts), OPTION_DIM] float32  (legal options only)
    option_counts [N] uint16
    labels        [N] uint16
    values        [N] int8
    game_ids      [N] int32
Multi-select answers contribute one sample per chosen index (same state),
exactly like the replay parser. Encoders come from encoding.py — imported,
never reimplemented.

Usage (repo root):
    python -m src.rl_models.bc_dataset --games-hh 800 --games-hr 500
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np

from cg import game as cg_game

from ..agent_heuristics.heuristic_agent import HeuristicAgent
from ..agent_heuristics.random_agent import RandomAgent, read_deck_csv
from ..environment_wrapper.selfplay import MAX_SELECTIONS_PER_GAME, RESULT_DRAW
from ..environment_wrapper.wrapper import EnvironmentWrapper
from ..ingestion.build_card_model import PROCESSED_DIR
from ..ingestion.build_effect_model import EffectIndex
from ..ingestion.card_index import CardIndex
from . import encoding
from .encoding import ENCODING_DIM, MAX_OPTIONS, OPTION_DIM, OptionEncoder, StateEncoder

logger = logging.getLogger(__name__)

DATASET_PATH: Final[Path] = PROCESSED_DIR / "bc_dataset.npz"
META_PATH: Final[Path] = PROCESSED_DIR / "bc_dataset.meta.json"


@dataclass
class BcStats:
    games: int = 0
    decisions: int = 0
    samples: int = 0
    skipped_overflow_label: int = 0
    draws: int = 0
    exceptions: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class _Buffers:
    states: list[np.ndarray] = field(default_factory=list)
    options_flat: list[np.ndarray] = field(default_factory=list)
    option_counts: list[int] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)
    values: list[int] = field(default_factory=list)
    game_ids: list[int] = field(default_factory=list)


def _play_and_record(game_id: int, heuristic_seats: tuple[int, ...],
                     agents: tuple, deck: list[int],
                     state_encoder: StateEncoder, option_encoder: OptionEncoder,
                     wrapper: EnvironmentWrapper, buffers: _Buffers,
                     stats: BcStats) -> None:
    """One game; buffer (state, legal options, label, seat) per heuristic pick."""
    pending: list[tuple[np.ndarray, np.ndarray, int, int]] = []
    obs_dict, start = cg_game.battle_start(list(deck), list(deck))
    if obs_dict is None:
        raise RuntimeError(f"battle_start failed: {start.errorType}")
    try:
        result = -1
        for _ in range(MAX_SELECTIONS_PER_GAME):
            state = obs_dict["current"]
            result = state["result"]
            if result != -1:
                break
            acting = state["yourIndex"]
            answer = agents[acting](obs_dict)
            if acting in heuristic_seats:
                obs = wrapper.parse(obs_dict)
                if obs.select is not None:
                    n_options = len(obs.select.option)
                    state_vec = state_encoder.encode(obs)
                    option_matrix, mask = encoding.build_action_mask(
                        obs, state_encoder, option_encoder)
                    legal = option_matrix[mask]
                    stats.decisions += 1
                    for chosen in answer:
                        if not 0 <= chosen < min(n_options, MAX_OPTIONS):
                            stats.skipped_overflow_label += 1
                            continue
                        pending.append((state_vec, legal, chosen, acting))
            obs_dict = cg_game.battle_select(answer)
    finally:
        cg_game.battle_finish()

    if result == RESULT_DRAW:
        stats.draws += 1
    for state_vec, legal, chosen, seat in pending:
        z = 0 if result == RESULT_DRAW else (1 if seat == result else -1)
        buffers.states.append(state_vec)
        buffers.options_flat.append(legal)
        buffers.option_counts.append(legal.shape[0])
        buffers.labels.append(chosen)
        buffers.values.append(z)
        buffers.game_ids.append(game_id)
        stats.samples += 1


def generate(games_hh: int, games_hr: int, seed: int = 0,
             out_path: Path = DATASET_PATH) -> BcStats:
    index = CardIndex()
    effects = EffectIndex()
    state_encoder = StateEncoder(index, effects)
    option_encoder = OptionEncoder(index, effects)
    wrapper = EnvironmentWrapper(index)
    deck = read_deck_csv()

    buffers = _Buffers()
    stats = BcStats()
    # (n games, heuristic seats, agent factory per seat)
    schedule: list[tuple[int, tuple[int, ...]]] = (
        [(game_index, (0, 1)) for game_index in range(games_hh)]
        + [(games_hh + game_index, (game_index % 2,))  # alternate the seat
           for game_index in range(games_hr)]
    )
    for game_id, heuristic_seats in schedule:
        game_seed = seed + 2 * game_id
        agents = tuple(
            HeuristicAgent(seed=game_seed + s, index=index, effects=effects)
            if s in heuristic_seats else RandomAgent(seed=game_seed + s)
            for s in range(2)
        )
        stats.games += 1
        try:
            _play_and_record(game_id, heuristic_seats, agents, deck,
                             state_encoder, option_encoder, wrapper,
                             buffers, stats)
        except Exception as exc:  # noqa: BLE001 — a bad game must not kill the run
            stats.exceptions += 1
            stats.errors.append(f"game {game_id}: {type(exc).__name__}: {exc}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        states=(np.stack(buffers.states).astype(np.float32)
                if buffers.states else np.zeros((0, ENCODING_DIM), np.float32)),
        options_flat=(np.concatenate(buffers.options_flat, axis=0)
                      if buffers.options_flat
                      else np.zeros((0, OPTION_DIM), np.float32)),
        option_counts=np.asarray(buffers.option_counts, dtype=np.uint16),
        labels=np.asarray(buffers.labels, dtype=np.uint16),
        values=np.asarray(buffers.values, dtype=np.int8),
        game_ids=np.asarray(buffers.game_ids, dtype=np.int32),
    )
    meta = {
        "schema": "ptcg-bc-dataset-v1",
        "encoding_dim": ENCODING_DIM,
        "option_dim": OPTION_DIM,
        "max_options": MAX_OPTIONS,
        "games_hh": games_hh,
        "games_hr": games_hr,
        "seed": seed,
        "games": stats.games,
        "decisions": stats.decisions,
        "samples": stats.samples,
        "skipped_overflow_label": stats.skipped_overflow_label,
        "draws": stats.draws,
        "exceptions": stats.exceptions,
    }
    with open(META_PATH, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    return stats


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games-hh", type=int, default=800,
                        help="heuristic×heuristic games (both seats recorded)")
    parser.add_argument("--games-hr", type=int, default=500,
                        help="heuristic×random games (heuristic seat alternates)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DATASET_PATH)
    args = parser.parse_args()

    t0 = time.perf_counter()
    stats = generate(args.games_hh, args.games_hr, args.seed, args.out)
    elapsed = time.perf_counter() - t0
    print(f"games:                 {stats.games} in {elapsed:.1f}s")
    print(f"heuristic decisions:   {stats.decisions}")
    print(f"training samples:      {stats.samples}")
    print(f"skipped (overflow):    {stats.skipped_overflow_label}")
    print(f"draws:                 {stats.draws}")
    print(f"exceptions:            {stats.exceptions}  (must be 0)")
    print(f"dataset: {args.out} ({args.out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
