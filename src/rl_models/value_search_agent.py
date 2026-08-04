"""ValueSearchAgent — CrustleAgent v3 plus a value-leaf turn search.

Same risk profile as RuntimeSearchAgent, and for the same reason: every
path that is not "confident deck estimate AND affordable AND eligible
AND the determinizer closed AND the value head loaded" ends at the
prior, and the prior IS the current ship. The floor of this agent is
the thing it would replace.

What is different from RuntimeSearchAgent:

    leaf        the value head (value_net.NumpyValueNet) instead of a
                rollout to the end of the game
    depth       a beam over the whole remainder of OUR turn instead of
                one micro-action
    opponent    not modelled at all inside the search. Every node from
                the root to the horizon is our own decision; the
                opponent appears only as the position it inherits.

That last line is the point. The rollout search's measured benefit was
a function of how closely its rollout policy imitated the true
opponent, so it evaporated against anything that was not our own
heuristic. There is no rollout policy here to be wrong.

The estimator is still used -- not to model how they PLAY, but to know
which 60 cards to determinize their hidden zones from. A wrong deck
read costs a worse determinization; it no longer costs a wrong model of
their decisions.

Offline evaluation can pass ``opponent_deck_override`` to hold the deck
read perfect and ask what the search alone buys.
"""

from __future__ import annotations

import copy
import logging
import random
import time
from pathlib import Path
from typing import Sequence

from cg import api

from ..agent_heuristics.crustle_agent import CrustleAgent
from ..agent_heuristics.random_agent import read_deck_csv
from ..ingestion.build_effect_model import EffectIndex
from ..ingestion.card_index import CardIndex
from .budget import BudgetGuard
from .determinize import runtime_determinize
from .opponent_estimator import OpponentDeckEstimator
from .runtime_search_agent import DEFAULT_CONTESTED_MARGIN, _is_contested
from .search_core import eligibility_reason, rank_candidates
from .value_search import (DEFAULT_VALUE_LADDER, NODE_PRIOR_S,
                           ValueLeafEvaluator, ValueSearchStats, ValueTier,
                           beam_search_turn)

logger = logging.getLogger(__name__)


class ValueSearchAgent:
    """Kaggle agent contract: ``agent(obs_dict) -> list[int]``."""

    def __init__(
        self,
        deck_path: str | Path | None = None,
        index: CardIndex | None = None,
        effects: EffectIndex | None = None,
        variant: str = "v3",
        estimator: OpponentDeckEstimator | None = None,
        guard: BudgetGuard | None = None,
        seed: int = 0,
        stats: ValueSearchStats | None = None,
        opponent_deck_override: Sequence[int] | None = None,
        own_deck_ids: Sequence[int] | None = None,
        enable_search: bool = True,
        contested_margin: float | None = DEFAULT_CONTESTED_MARGIN,
        override_margin: float = 0.0,
        value_weights: Path | None = None,
        ladder: tuple[ValueTier, ...] = DEFAULT_VALUE_LADDER,
        fixed_tier: ValueTier | None = None,
        select_worst: bool = False,
    ) -> None:
        self._index = index if index is not None else CardIndex()
        self._effects = effects if effects is not None else EffectIndex()
        self._deck_path = str(deck_path) if deck_path else None
        self._prior = CrustleAgent(seed=seed, deck_path=self._deck_path,
                                   index=self._index, effects=self._effects,
                                   variant=variant)
        self._own_deck = ([int(c) for c in own_deck_ids] if own_deck_ids
                          else self._read_own_deck())
        self._enable_search = enable_search
        self._contested_margin = contested_margin
        self._override_margin = max(0.0, override_margin)
        self._rng = random.Random(seed)
        self.stats = stats if stats is not None else ValueSearchStats()
        self.estimator = (estimator if estimator is not None
                          else OpponentDeckEstimator(index=self._index))
        self.guard = (guard if guard is not None
                      else BudgetGuard(ladder=ladder,
                                       rollout_prior_s=NODE_PRIOR_S))
        self._fixed_tier = fixed_tier
        # DIAGNOSTIC ONLY. Picks the candidate the head likes LEAST.
        # If argmin and argmax perform alike, the head carries no usable
        # signal at this granularity and no amount of tuning will help;
        # if argmin is much worse, the signal is real and the sign is
        # right, and the loss is coming from somewhere else. Never ship.
        self._select_worst = select_worst
        self._override = (tuple(int(c) for c in opponent_deck_override)
                          if opponent_deck_override else None)
        self.last_candidate_values: dict[int, float] | None = None

        # The value head is loaded ONCE. A missing or corrupt file means
        # the search has no leaf, and an agent with no leaf must be its
        # prior rather than something improvised -- so the failure is
        # recorded here and every later decision short-circuits.
        from .encoding import StateEncoder
        from .value_net import NumpyValueNet
        net = NumpyValueNet.load(value_weights) if value_weights else \
            NumpyValueNet.load()
        self._leaf = (ValueLeafEvaluator(net, StateEncoder(self._index,
                                                           self._effects),
                                         self.stats)
                      if net is not None else None)
        if net is None:
            logger.warning("value head unavailable; agent is its prior")

    # ------------------------------------------------------------------ #

    def _read_own_deck(self) -> list[int]:
        try:
            return [int(c) for c in read_deck_csv(self._deck_path)]
        except Exception:  # noqa: BLE001 — the prior still answers
            logger.warning("could not read own deck from %r", self._deck_path,
                           exc_info=True)
            return []

    # ------------------------------------------------------------------ #
    # Contract
    # ------------------------------------------------------------------ #

    def __call__(self, obs_dict: dict) -> list[int]:
        self.last_candidate_values = None
        if not isinstance(obs_dict, dict) or obs_dict.get("select") is None:
            self.estimator.reset()
            self.guard.reset()
            return list(self._own_deck)

        self.stats.decisions += 1
        answer = self._prior(copy.deepcopy(obs_dict))
        if not self._enable_search or self._leaf is None:
            self.stats.fallback_reasons["search-disabled"] += 1
            return answer

        scores = getattr(self._prior, "last_scores", None)
        estimate = self.estimator.observe(obs_dict)

        reason = eligibility_reason(obs_dict, answer, scores)
        if reason is not None:
            self.stats.fallback_reasons[reason] += 1
            return answer

        opp_deck = (list(self._override) if self._override is not None
                    else list(estimate.deck_ids))
        if not opp_deck or (self._override is None and not estimate.usable):
            self.stats.fallback_reasons[f"estimator:{estimate.reason}"] += 1
            return answer

        if not _is_contested(list(scores), self._contested_margin):
            self.stats.fallback_reasons["dominated"] += 1
            return answer

        tier = (self._fixed_tier if self._fixed_tier is not None
                else self.guard.choose(obs_dict))
        if tier is None:
            self.stats.fallback_reasons["budget"] += 1
            return answer

        nodes_before = self.stats.nodes
        t0 = time.perf_counter()
        best = None
        try:
            best = self._search(obs_dict, list(scores), answer[0], opp_deck,
                                tier)
        except Exception:  # noqa: BLE001 — counted, prior answer still legal
            self.stats.exceptions += 1
            logger.debug("value search failed; falling back", exc_info=True)
        finally:
            elapsed = time.perf_counter() - t0
            self.stats.search_time_s += elapsed
            # Charge the time either way; feed the cost model the nodes
            # actually expanded so a cheap early bail cannot teach the
            # guard that nodes are cheaper than they are.
            self.guard.record(tier, elapsed, self.stats.nodes - nodes_before)

        if best is None:
            return answer
        self.stats.searched += 1
        self.stats.tier_uses[tier.name] += 1
        if best != answer[0]:
            self.stats.changed += 1
        return [best]

    # ------------------------------------------------------------------ #

    def _search(self, obs_dict: dict, scores: list[float], prior_choice: int,
                opp_deck: list[int], tier: ValueTier) -> int | None:
        seat = obs_dict["current"]["yourIndex"]
        candidates = rank_candidates(scores, prior_choice, tier.n_candidates)
        if len(set(candidates)) < 2:
            self.stats.fallback_reasons["single-candidate"] += 1
            return None

        # The deepcopy stays. to_dataclass rebuilds nested DATACLASSES but
        # assigns plain lists straight through (cg/utils.py: the
        # non-dataclass branch is `d[key] = value`), so the Observation
        # it returns can alias lists inside the caller's dict. It costs
        # ~1.4ms once per decision, not per node — the hot loop was
        # never where this mattered.
        obs_cls = api.to_observation_class(copy.deepcopy(obs_dict))
        totals: dict[int, float] = {}
        counts: dict[int, int] = {}
        samples = 0
        for _ in range(tier.n_determinizations):
            det = runtime_determinize(obs_dict, seat, self._own_deck,
                                      opp_deck, self._rng)
            if det is None:
                self.stats.fallback_reasons["determinize"] += 1
                break
            root = api.search_begin(obs_cls, *det, [])
            try:
                values = beam_search_turn(root, seat, candidates, tier.depth,
                                          tier.beam, self._leaf, self.stats)
            finally:
                api.search_end()
            for cand, value in values.items():
                totals[cand] = totals.get(cand, 0.0) + value
                counts[cand] = counts.get(cand, 0) + 1
            samples += 1

        if samples == 0 or not totals:
            return None
        # A candidate the beam never priced under some determinization is
        # averaged over the ones it did -- not over `samples` -- so a
        # candidate is never penalised for lines the search declined to
        # open.
        means = {c: totals[c] / counts[c] for c in totals}
        self.last_candidate_values = means
        if prior_choice not in means:
            return None
        if self._select_worst:
            return min(means, key=lambda c: (means[c], -scores[c]))
        best = max(means, key=lambda c: (means[c], scores[c]))
        if best == prior_choice or self._override_margin <= 0.0:
            return best
        if means[best] - means[prior_choice] < self._override_margin:
            self.stats.fallback_reasons["below-margin"] += 1
            return prior_choice
        return best


__all__ = ["ValueSearchAgent"]
