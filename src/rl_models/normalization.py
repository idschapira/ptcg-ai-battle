"""Frozen per-feature input normalization shared by training and inference.

Statistics are computed ONCE over a self-play corpus (mixed matchups so
the state distribution covers heuristic and random play), frozen into
models/feature_stats.npz, and applied IDENTICALLY by the torch trainer
(src/rl_models/network.py, dev only) and the pure-numpy runtime forward
(src/rl_models/network_numpy.py):

    normalize(x) = clip((x - mean) / (std + EPS), -CLIP, CLIP)

numpy only — this module ships in the submission. None-safe: if the
stats file is missing, FeatureStats.load falls back to identity stats
(mean 0 / std 1) so the agent still answers legally.

Regenerate (repo root):
    python -m src.rl_models.normalization --games-per-matchup 40
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from ..ingestion.build_card_model import REPO_ROOT
from .encoding import ENCODING_DIM, OPTION_DIM

logger = logging.getLogger(__name__)

MODELS_DIR: Final[Path] = REPO_ROOT / "models"
FEATURE_STATS_PATH: Final[Path] = MODELS_DIR / "feature_stats.npz"

# clip((x - mean) / (std + EPS), -CLIP, CLIP): EPS is deliberately large
# (1e-2, not 1e-8) so near-constant binary features saturate at the clip
# instead of exploding; CLIP bounds every input the net ever sees (the
# raw encoding's observed max was 5.625 — see test_encoding sweep).
EPS: Final[float] = 1e-2
CLIP: Final[float] = 6.0


def normalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """The single normalization formula (train == inference, bit-for-bit)."""
    return np.clip((x - mean) / (std + EPS), -CLIP, CLIP).astype(np.float32)


@dataclass(frozen=True)
class FeatureStats:
    """Frozen mean/std for the state vector and the option vector."""

    state_mean: np.ndarray   # [ENCODING_DIM] float32
    state_std: np.ndarray    # [ENCODING_DIM] float32
    option_mean: np.ndarray  # [OPTION_DIM] float32
    option_std: np.ndarray   # [OPTION_DIM] float32

    def normalize_state(self, x: np.ndarray) -> np.ndarray:
        return normalize(x, self.state_mean, self.state_std)

    def normalize_options(self, x: np.ndarray) -> np.ndarray:
        return normalize(x, self.option_mean, self.option_std)

    @classmethod
    def identity(cls) -> "FeatureStats":
        return cls(
            state_mean=np.zeros(ENCODING_DIM, dtype=np.float32),
            state_std=np.ones(ENCODING_DIM, dtype=np.float32),
            option_mean=np.zeros(OPTION_DIM, dtype=np.float32),
            option_std=np.ones(OPTION_DIM, dtype=np.float32),
        )

    @classmethod
    def load(cls, path: Path = FEATURE_STATS_PATH) -> "FeatureStats":
        """Load frozen stats; identity fallback keeps the agent legal."""
        try:
            with np.load(path) as data:
                stats = cls(
                    state_mean=data["state_mean"].astype(np.float32),
                    state_std=data["state_std"].astype(np.float32),
                    option_mean=data["option_mean"].astype(np.float32),
                    option_std=data["option_std"].astype(np.float32),
                )
            if (stats.state_mean.shape != (ENCODING_DIM,)
                    or stats.option_mean.shape != (OPTION_DIM,)):
                raise ValueError(f"stats shape mismatch in {path}")
            return stats
        except (OSError, KeyError, ValueError) as exc:
            logger.warning("feature stats unavailable (%s); using identity", exc)
            return cls.identity()

    def save(self, path: Path = FEATURE_STATS_PATH,
             extra_meta: dict[str, float] | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            state_mean=self.state_mean, state_std=self.state_std,
            option_mean=self.option_mean, option_std=self.option_std,
            eps=np.float32(EPS), clip=np.float32(CLIP),
            **{k: np.float32(v) for k, v in (extra_meta or {}).items()},
        )


# --------------------------------------------------------------------------- #
# Corpus generation (dev only — imports agents and the engine)
# --------------------------------------------------------------------------- #

MATCHUPS: Final[tuple[tuple[str, str], ...]] = (
    ("heuristic", "heuristic"),
    ("heuristic", "random"),
    ("random", "random"),
)


def build_corpus(games_per_matchup: int,
                 seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Play mixed self-play and return (states [N,D], options [M,OPTION_DIM])."""
    from cg import game as cg_game

    from ..agent_heuristics.random_agent import read_deck_csv
    from ..environment_wrapper.selfplay import _make_agent
    from ..environment_wrapper.wrapper import EnvironmentWrapper
    from ..ingestion.build_effect_model import EffectIndex
    from ..ingestion.card_index import CardIndex

    index = CardIndex()
    effects = EffectIndex()
    from .encoding import OptionEncoder, StateEncoder
    state_encoder = StateEncoder(index, effects)
    option_encoder = OptionEncoder(index, effects)
    wrapper = EnvironmentWrapper(index)

    deck = read_deck_csv()
    states: list[np.ndarray] = []
    options: list[np.ndarray] = []
    game_seed = seed
    for p0_kind, p1_kind in MATCHUPS:
        for _ in range(games_per_matchup):
            agents = (_make_agent(p0_kind, game_seed),
                      _make_agent(p1_kind, game_seed + 1))
            game_seed += 2
            obs_dict, start = cg_game.battle_start(list(deck), list(deck))
            if obs_dict is None:
                raise RuntimeError(f"battle_start failed: {start.errorType}")
            try:
                for _ in range(20_000):
                    state = obs_dict["current"]
                    if state["result"] != -1:
                        break
                    obs = wrapper.parse(obs_dict)
                    states.append(state_encoder.encode(obs))
                    if obs.select is not None:
                        for option in obs.select.option:
                            options.append(option_encoder.encode(obs, option))
                    obs_dict = cg_game.battle_select(agents[state["yourIndex"]](obs_dict))
            finally:
                cg_game.battle_finish()
    return (np.stack(states).astype(np.float32),
            np.stack(options).astype(np.float32))


def compute_stats(states: np.ndarray, options: np.ndarray) -> FeatureStats:
    return FeatureStats(
        state_mean=states.mean(axis=0).astype(np.float32),
        state_std=states.std(axis=0).astype(np.float32),
        option_mean=options.mean(axis=0).astype(np.float32),
        option_std=options.std(axis=0).astype(np.float32),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games-per-matchup", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FEATURE_STATS_PATH)
    args = parser.parse_args()

    t0 = time.perf_counter()
    states, options = build_corpus(args.games_per_matchup, args.seed)
    stats = compute_stats(states, options)
    stats.save(args.out, extra_meta={"n_state_samples": len(states),
                                     "n_option_samples": len(options)})
    normalized = stats.normalize_state(states)
    print(f"corpus: {len(states)} states, {len(options)} options "
          f"({time.perf_counter() - t0:.1f}s, "
          f"{3 * args.games_per_matchup} games)")
    print(f"raw state range:        [{states.min():.3f}, {states.max():.3f}]")
    print(f"normalized state range: [{normalized.min():.3f}, {normalized.max():.3f}] "
          f"(clip ±{CLIP})")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
