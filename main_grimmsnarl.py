"""Grimmsnarl probe entrypoint (meta_grimmsnarl + BC-Grimmsnarl). Packaged
AS main.py at the root of submission_grimmsnarl.tar.gz by build_submission
--target grimmsnarl — the repo-root main.py (Final A, Crustle) is untouched.

Kaggle loads the packaged file with exec(source, env) — NOT as a module
import — so `__file__` may be ABSENT from the namespace. Every path here
must resolve without it (same contract as main.py).
"""

from __future__ import annotations

import os
import sys

_KAGGLE_AGENT_DIR = "/kaggle_simulations/agent"


def _resolve_here() -> str:
    """Bundle root: __file__ when present (module import / smoke-as-file),
    else probe cwd, else the contracted Kaggle agent dir."""
    if "__file__" in globals():
        return os.path.dirname(os.path.abspath(__file__))
    for candidate in (os.getcwd(), _KAGGLE_AGENT_DIR):
        if os.path.exists(os.path.join(candidate, "deck.csv")):
            return candidate
    return _KAGGLE_AGENT_DIR


_HERE = _resolve_here()
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from src.rl_models.network_agent import NetworkAgent

# Grimmsnarl probe = (meta_grimmsnarl deck, BC-Grimmsnarl): behavior clone
# of the Grimmsnarl leaders (Luca, 94 games on 07-17), pure-numpy
# inference. The weights/stats pair is MATED — never promote one without
# the other. Missing weights degrade to HeuristicAgent inside
# NetworkAgent; any per-call exception degrades to a legal raw answer.
_agent = NetworkAgent(
    deck_path=os.path.join(_HERE, "deck.csv"),
    weights_path=os.path.join(_HERE, "models", "bc_grimmsnarl.npz"),
    stats_path=os.path.join(_HERE, "models", "feature_stats.npz"),
)


def agent(obs_dict: dict) -> list[int]:
    """The function the competition harness calls every selection."""
    return _agent(obs_dict)
