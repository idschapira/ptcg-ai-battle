"""Final B entrypoint (Spidops + BC-Spidops v2). Packaged AS main.py at
the root of submission_spidops.tar.gz by build_submission --target
spidops — the repo-root main.py (Final A, Crustle) is untouched.

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

# Final B = (meta_spidops deck, BC-Spidops v2): behavior clone of the
# Spidops leaders (kashiwashira + Oshbocker, 151 games), pure-numpy
# inference. The weights/stats pair is MATED — never promote one without
# the other. Missing weights degrade to HeuristicAgent inside
# NetworkAgent; any per-call exception degrades to a legal raw answer.
_agent = NetworkAgent(
    deck_path=os.path.join(_HERE, "deck.csv"),
    weights_path=os.path.join(_HERE, "models", "bc_spidops_v2.npz"),
    stats_path=os.path.join(_HERE, "models", "feature_stats.npz"),
)


def agent(obs_dict: dict) -> list[int]:
    """The function the competition harness calls every selection."""
    return _agent(obs_dict)
