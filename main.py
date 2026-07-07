"""Competition entrypoint. Kaggle imports this module and calls agent().

Must live at the root of submission.tar.gz next to deck.csv, cg/ and src/.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from src.agent_heuristics.random_agent import RandomAgent

_agent = RandomAgent(deck_path=os.path.join(_HERE, "deck.csv"))


def agent(obs_dict: dict) -> list[int]:
    """The function the competition harness calls every selection."""
    return _agent(obs_dict)
