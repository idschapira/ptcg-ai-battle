"""ParametricHeuristicAgent: HeuristicAgent + (DeckModule, Theta).

The runtime shape of the league. Behaviour is the generic heuristic
UNLESS the bound deck module has an opinion, and every opinion is a
function of the parameter vector theta — so the whole strategy is a
point in a bounded search space that Phase 2 can mutate.

Safety is structural, not per-rule:

- every module hook is called inside try/except and falls back to the
  generic score, so a module bug degrades to generic play;
- a module that takes over a SelectContext still goes through
  `_pick_top`, which clamps counts against the real option list, so the
  answer stays legal by construction;
- with no module (or the base DeckModule) the agent is behaviourally
  identical to HeuristicAgent.

Deterministic: like HeuristicAgent, scoring never consults the rng, so
the same observation always produces the same answer — which is what
makes the CrustleAgent equivalence test a real bit-for-bit check.
"""

from __future__ import annotations

from cg.api import Observation, Option, SelectContext

from ..agent_heuristics.heuristic_agent import HeuristicAgent
from ..ingestion.build_effect_model import EffectIndex
from ..ingestion.card_index import CardIndex
from .board import BoardView
from .module import DeckModule
from .theta import Theta


class ParametricHeuristicAgent(HeuristicAgent):
    """Generic heuristic with a theta-parameterized deck module on top."""

    __slots__ = ("_module", "_theta")

    def __init__(
        self,
        module: DeckModule | None = None,
        theta: Theta | None = None,
        seed: int | None = None,
        deck_path: str | None = None,
        index: CardIndex | None = None,
        effects: EffectIndex | None = None,
    ) -> None:
        super().__init__(seed=seed, deck_path=deck_path, index=index,
                         effects=effects)
        self._module = module if module is not None else DeckModule()
        if theta is None:
            theta = self._module.default_theta()
        elif theta.schema is not self._module.schema:
            # a genome from another schema: re-key by NAME so a stored
            # Hall of Fame entry never silently lands on wrong knobs
            theta = self._module.schema.from_dict(theta.to_dict())
        self._theta = theta
        self._module.bind(self)

    # ------------------------------------------------------------------ #
    # Introspection (used by the fitness harness and the HoF)
    # ------------------------------------------------------------------ #

    @property
    def module(self) -> DeckModule:
        return self._module

    @property
    def theta(self) -> Theta:
        return self._theta

    def __repr__(self) -> str:
        return (f"<ParametricHeuristicAgent module={self._module.name!r} "
                f"knobs={len(self._theta)}>")

    # ------------------------------------------------------------------ #
    # Overlay plumbing
    # ------------------------------------------------------------------ #

    def _view(self, obs: Observation) -> BoardView:
        return BoardView(self, obs, self._theta)

    def _main_score(self, obs: Observation, option: Option) -> float:
        base = super()._main_score(obs, option)
        try:
            value = self._module.main_score(self._view(obs), option, base)
        except Exception:
            return base
        return base if value is None else value

    def _own_pokemon_score(self, obs: Observation, option: Option,
                           for_active: bool) -> float:
        base = super()._own_pokemon_score(obs, option, for_active)
        try:
            value = self._module.own_pokemon_score(self._view(obs), option,
                                                   for_active, base)
        except Exception:
            return base
        return base if value is None else value

    def _attach_score(self, obs: Observation, option: Option) -> float:
        base = super()._attach_score(obs, option)
        try:
            value = self._module.attach_score(self._view(obs), option, base)
        except Exception:
            return base
        return base if value is None else value

    def _decide(self, obs: Observation) -> list[int]:
        select = obs.select
        if select is not None:
            try:
                plan = self._module.select_score(self._view(obs),
                                                 select.context)
            except Exception:
                plan = None
            if plan is not None:
                try:
                    score_fn, min_count, max_count = plan
                    # _pick_top clamps against len(options): legal by
                    # construction even if the module asks for nonsense.
                    return self._pick_top(select.option, min_count, max_count,
                                          lambda i, o: score_fn(o))
                except Exception:
                    pass  # fall through to the generic (always legal) path
        return super()._decide(obs)


def build_agent(module: DeckModule, theta: Theta | None = None, *,
                seed: int | None = None, deck_path: str | None = None,
                index: CardIndex | None = None,
                effects: EffectIndex | None = None) -> ParametricHeuristicAgent:
    """Factory used by the fitness harness (one fresh agent per game)."""
    return ParametricHeuristicAgent(module=module, theta=theta, seed=seed,
                                    deck_path=deck_path, index=index,
                                    effects=effects)


__all__ = ["ParametricHeuristicAgent", "build_agent"]
