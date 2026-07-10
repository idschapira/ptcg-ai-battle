"""Competition entrypoint. Kaggle loads this file with exec(source, env) —
NOT as a module import — so `__file__` may be ABSENT from the namespace
(kaggle_environments loader). Every path here must resolve without it.

Must live at the root of submission.tar.gz next to deck.csv, cg/ and src/.
The agent is defined at module level: the Kaggle exec never runs a
`if __name__ == "__main__"` block.
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

from src.agent_heuristics.crustle_agent import CrustleAgent

# Ship = (Crustle deck, CrustleAgent v2): kernel-blueprint stall rules
# (gust-to-trap, mill>damage, Guidance-aware search/discard) over the
# generic heuristic; every path degrades to a legal answer, never
# crashes. Rollbacks, in order: remove variant="v2" -> v1 pilot (the
# previous ship); swap the import/constructor to NetworkAgent -> the 5D
# par (models/*.npz stay bundled). read_deck_csv itself falls back to
# /kaggle_simulations/agent/deck.csv.
_agent = CrustleAgent(deck_path=os.path.join(_HERE, "deck.csv"),
                      variant="v2")


def agent(obs_dict: dict) -> list[int]:
    """The function the competition harness calls every selection."""
    return _agent(obs_dict)
