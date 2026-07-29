"""Competition entrypoint — Crustle deck flown by RuntimeSearchAgent.

Packaged AS main.py by `python -m src.build_submission --target
search_crustle`. Kaggle loads this with exec(source, env), NOT as a
module import, so `__file__` may be ABSENT from the namespace. Every
path here must resolve without it.

Ship = (deck.csv = Crustle e10, RuntimeSearchAgent over CrustleAgent v3).
The pilot is the current ship plus a shallow determinized search that
only runs when it has BOTH a confident read on the opponent's archetype
and the time bank to pay for it; on every other decision it is exactly
CrustleAgent v3. That is the rollback story too: the floor is the thing
this replaces, so the downside is bounded by construction rather than by
a configuration flag.

ROLLBACKS, in order of severity:
    enable_search=False        -> byte-identical to the Final A pilot
    swap to CrustleAgent(v3)   -> the Final A bundle itself
Final A (submission.tar.gz) is built from the untouched repo main.py and
is not affected by anything in this file.
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

from src.rl_models.runtime_search_agent import RuntimeSearchAgent

# Defaults are deliberate, not incidental:
#   - the budget guard's own ladder and 150s reserve (see budget.py); the
#     projection re-calibrates itself to Kaggle's slower cores, so no
#     machine-specific constant is baked in here;
#   - the estimator's measured thresholds (100% precision when confident
#     over 271k decisions of real ladder replay, see
#     src/analysis/estimator_accuracy.py).
_agent = RuntimeSearchAgent(deck_path=os.path.join(_HERE, "deck.csv"),
                            variant="v3")


def agent(obs_dict: dict) -> list[int]:
    """The function the competition harness calls every selection."""
    return _agent(obs_dict)
