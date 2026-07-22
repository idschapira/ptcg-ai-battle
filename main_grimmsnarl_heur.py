"""Grimmsnarl HEURISTIC entrypoint (meta_grimmsnarl + ParametricHeuristicAgent).
Packaged AS main.py at the root of submission_grimmsnarl_heur.tar.gz by
build_submission --target grimmsnarl_heur — the repo-root main.py (Final A,
Crustle) is untouched.

This is the pilot half of a ladder A/B: the SAME deck as the BC-Grimmsnarl
probe (submission 54841794), flown by the league's rule-based pilot instead
of the behavior clone. Same deck, same field, two curves — which is what
makes the comparison a clean test of the pilot rather than the deck.

No .npz, no mated pair: the strategy is a 53-knob theta over rule bands,
shipped as data/theta/grimmsnarl_heur_v1.json.

Kaggle loads the packaged file with exec(source, env) — NOT as a module
import — so `__file__` may be ABSENT from the namespace. Every path here
must resolve without it (same contract as main.py).
"""

from __future__ import annotations

import json
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

from src.league.modules.grimmsnarl import GrimmsnarlModule
from src.league.parametric_agent import ParametricHeuristicAgent

_module = GrimmsnarlModule()

# The bundled theta wins; a missing/corrupt file degrades to the module
# defaults (identical today — the shipped genome IS the default). The
# fallback is silent BECAUSE the build smoke asserts the sha256 of the
# theta the live agent ended up holding: if the json failed to ship, the
# build fails before a submission can be made.
_theta = _module.default_theta()
try:
    with open(os.path.join(_HERE, "data", "theta",
                           "grimmsnarl_heur_v1.json"), encoding="utf-8") as fh:
        # from_dict is keyed by NAME, clips into the legal bands and
        # tolerates missing/unknown knobs — it never raises.
        _theta = _module.schema.from_dict(json.load(fh))
except Exception:
    pass

# Every module hook runs inside try/except and falls back to the generic
# heuristic score, and module-driven selections still go through _pick_top,
# which clamps counts against the real option list — the answer stays legal
# by construction.
_agent = ParametricHeuristicAgent(module=_module, theta=_theta,
                                  deck_path=os.path.join(_HERE, "deck.csv"))


def agent(obs_dict: dict) -> list[int]:
    """The function the competition harness calls every selection."""
    return _agent(obs_dict)
