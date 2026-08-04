"""Stage A — position corpus for a Crustle value head (dev only).

The search we shipped as a candidate evaluated its leaves by ROLLOUT:
one simulated game per node. That bought at most ~32 evaluations per
decision and, worse, made the leaf value a function of how well our
rollout policy modelled the opponent's pilot — which is precisely the
axis the 29/Jul fidelity ladder showed the whole effect living on
(+14.7pp against our own model, -2.5pp against a behaviour clone).

A LEARNED leaf changes both. Measured on this box: a rollout leaf costs
~160ms, a value leaf (parse + encode + forward) ~0.24ms — ~660x more
nodes for the same bank. And evaluating a POSITION does not require
simulating the opponent's policy to the end of the game, so the leaf
stops being a bet on our model of them.

This module builds the corpus that head is trained on. It plays our
ship (deck.csv + CrustleAgent v3) against the whole corrected field AND
against the behaviour clones of real ladder humans, recording every
sampled position with the outcome of the game it came from.

BOTH SEATS ARE RECORDED, and z is always from the perspective of the
player TO MOVE in that observation. That is not extra data for its own
sake: inside the search, a candidate action that ends our turn lands on
a node where the OPPONENT is to move, so the head must be able to price
those too. One net, one convention — value is always "win probability
for whoever is on turn", and the search negates it when that is not us.

Positions are subsampled (--stride) because consecutive selections in
one game are nearly the same position; game diversity per byte is what
the head actually learns from. States are stored float16 (the encoding
is bounded ratios and one-hots — see encoding.py) to keep a
multi-hundred-thousand-position corpus in RAM at train time.

Run from the repo root (dev):
    python -m src.rl_models.value_collect --games 400 --workers 12
    python -m src.rl_models.value_collect --cells alakazam-majkel --games 200
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from pathlib import Path
from typing import Final

import numpy as np

from ..ingestion.build_card_model import REPO_ROOT
from .encoding import ENCODING_DIM

OUT_DIR: Final[Path] = REPO_ROOT / "data" / "processed" / "value_crustle"

# The corrected internal field — the same opponent pilot pilot_ab.py
# measures against, so the head is trained on the distribution the A/B
# will later be read on.
CORRECTED_FIELD: Final[str] = "heuristic-tempo-scaled-routing-gust-conserve,8"
STATS_ARG: Final[str] = "models/feature_stats.npz"

# (cell name, opponent decklist, opponent arm, share of OUR real ladder).
#
# The first eight are pilot_ab.FIELD: the corrected field, weighted by
# what we actually met on the ladder. The clone cells that follow carry
# share 0.0 — they are not extra field, they are the SAME archetypes
# flown by clones of real humans, and they exist so Gate A can ask
# whether the head is calibrated against opponents that are not our own
# heuristic. A head that only prices positions produced by our pilot's
# own model of the field would repeat the exact failure of the rollout
# search.
CELLS: Final[tuple[tuple[str, str, str, float], ...]] = (
    ("alakazam",     "data/decks/meta_alakazam.csv",           CORRECTED_FIELD, 0.2744),
    ("grimmsnarl",   "data/decks/meta_grimmsnarl.csv",         CORRECTED_FIELD, 0.1674),
    ("lucario",      "data/decks/meta_mega_lucario.csv",       CORRECTED_FIELD, 0.1488),
    ("archaludon",   "data/decks/meta_archaludon.csv",         CORRECTED_FIELD, 0.1488),
    ("kangaskhan",   "data/decks/meta_crustle_kangaskhan.csv", CORRECTED_FIELD, 0.0558),
    ("starmie",      "data/decks/meta_starmie.csv",            CORRECTED_FIELD, 0.0372),
    ("spidops",      "data/decks/meta_spidops.csv",            CORRECTED_FIELD, 0.0372),
    ("mirror",       "deck.csv",                               "crustle-v3",    0.0186),
    # clones of real ladder humans — NOT our rollout model
    ("alakazam-majkel", "data/decks/meta_alakazam.csv",
     f"network,models/bc_majkel.npz,{STATS_ARG}", 0.0),
    ("alakazam-yushin", "data/decks/meta_alakazam.csv",
     f"network,models/bc_yushin.npz,{STATS_ARG}", 0.0),
    ("grimmsnarl-luca", "data/decks/meta_grimmsnarl.csv",
     f"network,models/bc_luca.npz,{STATS_ARG}", 0.0),
    ("grimmsnarl-dries", "data/decks/meta_grimmsnarl.csv",
     f"network,models/bc_dries.npz,{STATS_ARG}", 0.0),
    ("spidops-bc", "data/decks/meta_spidops.csv",
     f"network,models/bc_spidops_v2.npz,{STATS_ARG}", 0.0),
)

CELL_INDEX: Final[dict[str, int]] = {c[0]: i for i, c in enumerate(CELLS)}

OUR_ARM: Final[str] = "crustle-v3"
OUR_DECK: Final[str] = "deck.csv"

RESULT_DRAW: Final[int] = 2
EPISODE_BASE: Final[int] = 700_000_000
# room for 10^6 games per cell before ids from different cells collide
CELL_BLOCK: Final[int] = 1_000_000


class _PositionRecorder:
    """play_one_game observer: encodes sampled positions, holds seats.

    ``z`` is unknown until the game ends, so the seat of each sampled
    position is kept and the outcome is applied in ``finish``.
    """

    __slots__ = ("_encoder", "_stride", "_offset", "_seen", "states", "seats",
                 "plies")

    def __init__(self, encoder, stride: int, offset: int = 0) -> None:
        self._encoder = encoder
        self._stride = max(1, stride)
        # A fixed phase would sample plies 0, S, 2S... of every game, so
        # whole stretches of the turn cycle would never be seen. The
        # offset is drawn per game, which makes the sample uniform over
        # plies once averaged across games.
        self._offset = offset % self._stride
        self._seen = 0
        self.states: list[np.ndarray] = []
        self.seats: list[int] = []
        self.plies: list[int] = []

    def __call__(self, obs_dict: dict) -> None:
        state = obs_dict.get("current") or {}
        # terminal observations carry no decision and no useful features
        if state.get("result", -1) != -1:
            return
        seat = state.get("yourIndex")
        if seat not in (0, 1):
            return
        ply = self._seen
        self._seen += 1
        if (ply - self._offset) % self._stride:
            return
        try:
            from cg.api import to_observation_class
            vec = self._encoder.encode(to_observation_class(obs_dict))
        except Exception:  # noqa: BLE001 — a bad position is not a dead run
            return
        self.states.append(np.asarray(vec, dtype=np.float16))
        self.seats.append(seat)
        self.plies.append(ply)


def _run_shard(payload: tuple[str, int, int, int]) -> dict:
    """Worker: ``games`` of one cell. Fresh engine, own CardIndex."""
    cell_name, games, seed, stride = payload
    from ..deckbuilding.gauntlet import read_deck_ids
    from ..environment_wrapper.ab_test import ArmMetrics, ArmSpec, arm_factory
    from ..environment_wrapper.selfplay import play_one_game
    from ..ingestion.build_effect_model import EffectIndex
    from ..ingestion.card_index import CardIndex
    from .encoding import StateEncoder

    _, opp_deck_text, opp_arm, _ = CELLS[CELL_INDEX[cell_name]]
    index, effects = CardIndex(), EffectIndex()
    encoder = StateEncoder(index, effects)
    our_deck = read_deck_ids(REPO_ROOT / OUR_DECK)
    opp_deck = read_deck_ids(REPO_ROOT / opp_deck_text)
    make_ours = arm_factory(ArmSpec.parse(OUR_ARM), index, effects,
                            ArmMetrics(), our_deck)
    make_opp = arm_factory(ArmSpec.parse(opp_arm), index, effects,
                           ArmMetrics(), opp_deck)

    states: list[np.ndarray] = []
    zs: list[int] = []
    ours: list[int] = []
    episodes: list[int] = []
    plies: list[int] = []
    wins = losses = draws = errors = 0

    for game_index in range(games):
        game_seed = seed + game_index
        our_seat = game_index % 2          # alternate seats
        agent_ours = make_ours(game_seed)
        agent_opp = make_opp(game_seed + 10_000)
        agents = ((agent_ours, agent_opp) if our_seat == 0
                  else (agent_opp, agent_ours))
        decks = ((our_deck, opp_deck) if our_seat == 0
                 else (opp_deck, our_deck))
        rec = _PositionRecorder(encoder, stride, offset=game_index * 7 + seed)
        try:
            result, _turns = play_one_game(agents, list(decks[0]),
                                           list(decks[1]), observer=rec)
        except Exception:  # noqa: BLE001 — counted, never fatal
            errors += 1
            continue
        if result == RESULT_DRAW:
            draws += 1
        elif result == our_seat:
            wins += 1
        else:
            losses += 1
        episode_id = (EPISODE_BASE + CELL_INDEX[cell_name] * CELL_BLOCK
                      + seed + game_index)
        for vec, seat, ply in zip(rec.states, rec.seats, rec.plies):
            if result == RESULT_DRAW:
                z = 0
            else:
                z = 1 if result == seat else -1
            states.append(vec)
            zs.append(z)
            ours.append(int(seat == our_seat))
            episodes.append(episode_id)
            plies.append(ply)

    return {
        "cell": cell_name,
        "states": (np.stack(states) if states
                   else np.zeros((0, ENCODING_DIM), np.float16)),
        "values": np.asarray(zs, np.int8),
        "ours": np.asarray(ours, np.int8),
        "episode_ids": np.asarray(episodes, np.int64),
        "plies": np.asarray(plies, np.int32),
        "wins": wins, "losses": losses, "draws": draws, "errors": errors,
        "games": games,
    }


def collect(cells: tuple[str, ...], games: int, workers: int, seed: int,
            stride: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    grand_positions = 0
    summary: list[dict] = []

    for cell_name in cells:
        path = out_dir / f"cell_{cell_name}.npz"
        if path.exists():
            with np.load(path) as data:
                n = len(data["values"])
            print(f"  {cell_name:18s} SKIP (exists, {n} positions)", flush=True)
            grand_positions += n
            continue

        per_worker = max(1, games // workers)
        payloads = [(cell_name, per_worker, seed + 100_000 * i, stride)
                    for i in range(workers)]
        cell_t0 = time.perf_counter()
        with mp.Pool(processes=workers) as pool:
            shards = pool.map(_run_shard, payloads)

        states = np.concatenate([s["states"] for s in shards], axis=0)
        values = np.concatenate([s["values"] for s in shards])
        ours = np.concatenate([s["ours"] for s in shards])
        episodes = np.concatenate([s["episode_ids"] for s in shards])
        plies = np.concatenate([s["plies"] for s in shards])
        wins = sum(s["wins"] for s in shards)
        losses = sum(s["losses"] for s in shards)
        draws = sum(s["draws"] for s in shards)
        errors = sum(s["errors"] for s in shards)
        played = wins + losses + draws
        np.savez_compressed(
            path, states=states, values=values, ours=ours,
            episode_ids=episodes, plies=plies,
            cell=np.asarray([CELL_INDEX[cell_name]], np.int32))
        grand_positions += len(values)
        winrate = wins / max(wins + losses, 1)
        summary.append({"cell": cell_name, "positions": int(len(values)),
                        "games": played, "winrate": winrate,
                        "errors": errors})
        print(f"  {cell_name:18s} {len(values):7d} positions  "
              f"{played:4d} games  our winrate {winrate:6.1%}  "
              f"errors {errors}  ({time.perf_counter() - cell_t0:.0f}s)",
              flush=True)

    print(f"\ntotal {grand_positions} positions in "
          f"{(time.perf_counter() - t0) / 60:.1f} min -> {out_dir}")
    total_errors = sum(s["errors"] for s in summary)
    print(f"exceptions across all cells: {total_errors} (must be 0)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=400,
                        help="games per cell (split across workers)")
    parser.add_argument("--cells", nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stride", type=int, default=3,
                        help="keep 1 position in N (consecutive selections "
                             "in one game are near-duplicates)")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    names = tuple(c[0] for c in CELLS)
    if args.cells:
        wanted = {c.casefold() for c in args.cells}
        unknown = wanted - {n.casefold() for n in names}
        if unknown:
            raise SystemExit(f"unknown cells: {sorted(unknown)}")
        names = tuple(n for n in names if n.casefold() in wanted)
    collect(names, args.games, args.workers, args.seed, args.stride,
            args.out_dir)


if __name__ == "__main__":
    main()
