"""Primitives shared by the offline and the submission search agents.

The two agents differ in exactly two things — where the opponent's
decklist comes from (known offline, estimated at runtime) and whether a
time bank constrains them — so everything else lives here and has one
implementation. In particular the ELIGIBILITY filter is shared: a state
the offline search refuses to branch on is one the submission must
refuse too, and letting those two drift would mean the offline
measurements no longer describe the shipped agent.

Runtime-safe: cg.api and the standard library only. Nothing here may
import the analysis or self-play harnesses.
"""

from __future__ import annotations

from typing import Callable, Final

from cg import api

from .determinize import _to_dict

Agent = Callable[[dict], list[int]]

RESULT_DRAW: Final[int] = 2
DRAW_VALUE: Final[float] = 0.5


def eligibility_reason(obs_dict: dict, answer: list[int],
                       scores: list[float] | None) -> str | None:
    """Why this decision cannot be searched, or None if it can.

    Mirrors the filter validated by the counterfactual analysis. Each
    rejection is a state where branching is either impossible (no
    ``search_begin_input``), meaningless (a single option), or not
    modelled by the determinizer (multi-select, an open ``looking``
    zone, a hidden opponent active).
    """
    select = obs_dict.get("select") or {}
    state = obs_dict.get("current") or {}
    options = select.get("option") or []
    if obs_dict.get("search_begin_input") is None:
        return "no-search-input"
    if select.get("deck") is not None:
        return "deck-select"
    if state.get("looking") is not None:
        return "looking-open"
    if select.get("maxCount") != 1:
        return "multi-select"
    if len(options) < 2:
        return "single-option"
    if len(answer) != 1 or not 0 <= answer[0] < len(options):
        return "prior-answer-shape"
    if not scores or len(scores) < 2:
        return "no-prior-scores"
    if len(scores) != len(options):
        # candidates are indices into `scores`; if the two ever drift we
        # would hand search_step an index that is not a legal option
        return "score-option-mismatch"
    your = state.get("yourIndex")
    players = state.get("players") or []
    if your not in (0, 1) or len(players) != 2:
        return "bad-state"
    opp_active = players[1 - your].get("active") or []
    if not opp_active or opp_active[0] is None:
        return "hidden-opp-active"
    return None


def rank_candidates(scores: list[float], prior_choice: int,
                    k: int) -> list[int]:
    """Top-k options by prior score, with the prior's own pick guaranteed.

    Returns fewer than 2 entries when there is nothing to compare, which
    the callers treat as "let the prior decide".
    """
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    candidates = order[:max(2, k)]
    if prior_choice not in candidates and candidates:
        candidates[-1] = prior_choice
    return candidates


def rollout_to_terminal(branch: api.SearchState, seat: int, ours: Agent,
                        theirs: Agent, cap: int) -> tuple[float, bool]:
    """Play a branch to the end. Returns (value for ``seat``, hit_cap).

    The leaf value is the REAL outcome of the rollout, never a learned
    value head: the 5D critic is frozen and miscalibrated (see
    CLAUDE.md), so an unbiased noisy estimate beats a biased confident
    one. A rollout that hits the cap scores as a draw.
    """
    node = branch
    for _ in range(cap):
        current = node.observation.current
        if current is not None and current.result != -1:
            if current.result == seat:
                return 1.0, False
            return (DRAW_VALUE if current.result == RESULT_DRAW else 0.0), False
        node_dict = _to_dict(node.observation)
        acting = node_dict["current"]["yourIndex"]
        agent = ours if acting == seat else theirs
        node = api.search_step(node.searchId, agent(node_dict))
    return DRAW_VALUE, True


__all__ = ["Agent", "DRAW_VALUE", "RESULT_DRAW", "eligibility_reason",
           "rank_candidates", "rollout_to_terminal"]
