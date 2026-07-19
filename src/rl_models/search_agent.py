"""SearchAgent (Camada 2): expectimax raso determinizado sobre um prior.

O prior (qualquer agente que exponha ``last_scores`` alinhado às options —
NetworkAgent, HeuristicAgent) propõe os K melhores candidatos da decisão;
cada candidato é avaliado sob D determinizações por MULTISET EXATO dos
ocultos (reuso de counterfactual.determinize — offline conhecemos as duas
decklists) jogando o ramo ATÉ O TERMINAL com políticas de rollout
configuráveis. Folha = resultado real do rollout (não-viesado) DE
PROPÓSITO: o value head do par 5D está descalibrado (crítico é da 5C,
ver CLAUDE.md), então ele NÃO entra na avaliação. As mesmas
determinizações servem todos os candidatos (pareado = menos variância).

Escopo: 1-ply (expectimax raso / ISMCTS determinizado sem árvore). PTCG
é informação imperfeita + estocástica e o custo é rollout-dominado —
orçamento extra vira mais determinizações/candidatos, não profundidade.

Ferramenta OFFLINE/dev (oponente de sparring, calibração de campo,
avaliação de candidato a piloto): sem restrição de latência de
submissão. O opponent-model dos rollouts é configurável e PODE ser o
oponente verdadeiro do matchup (avaliação interna) — reportar a escolha
junto com qualquer número produzido.

None-safety: inelegibilidade (sem search_begin_input, seleção múltipla,
``looking`` aberto, ativo do oponente oculto, multiset que não fecha)
ou qualquer exceção → resposta do prior, nunca crash; ``search_end()``
sempre em ``finally``. O prior é chamado ANTES da busca, então sempre
existe resposta legal para degradar.
"""

from __future__ import annotations

import copy
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Final

from cg import api

from ..analysis.counterfactual import _to_dict, determinize
from ..environment_wrapper.selfplay import RESULT_DRAW, Agent

AgentFactory = Callable[[int], Agent]

DRAW_VALUE: Final[float] = 0.5


@dataclass
class SearchStats:
    """Contadores agregáveis entre jogos (compartilhar a instância)."""

    decisions: int = 0
    searched: int = 0
    changed: int = 0            # buscas que trocaram a resposta do prior
    rollouts: int = 0
    rollout_caps: int = 0       # rollouts que bateram o teto (valem 0.5)
    exceptions: int = 0
    search_time_s: float = 0.0
    fallback_reasons: Counter = field(default_factory=Counter)

    @property
    def mean_search_ms(self) -> float:
        return (1000.0 * self.search_time_s / self.searched
                if self.searched else 0.0)

    def summary(self) -> str:
        share = self.searched / self.decisions if self.decisions else 0.0
        reasons = dict(self.fallback_reasons.most_common())
        return (f"decisões {self.decisions}, buscadas {self.searched} "
                f"({share:.0%}), trocas {self.changed}, "
                f"{self.mean_search_ms:.0f}ms/busca, "
                f"rollouts {self.rollouts} (cap: {self.rollout_caps}), "
                f"exceções {self.exceptions}, fallbacks {reasons}")


class SearchAgent:
    """Agente do contrato Kaggle: prior + busca rasa determinizada (dev).

    prior: agente que responde ``list[int]`` e expõe ``last_scores``
        (um float por option legal, na ordem das options).
    own_deck_ids/opp_deck_ids: as DUAS decklists (60 ids) — a
        determinização por multiset precisa de ambas (dev offline).
    rollout_self/rollout_opp: fábricas ``seed -> Agent`` que jogam o
        ramo até o fim (nosso lado / lado do oponente).
    n_candidates (K): options avaliadas por decisão (top-K do prior; a
        escolha do prior sempre participa).
    n_determinizations (D): amostras dos ocultos por decisão.
    """

    def __init__(
        self,
        prior: Agent,
        own_deck_ids: list[int],
        opp_deck_ids: list[int],
        rollout_self: AgentFactory,
        rollout_opp: AgentFactory,
        n_candidates: int = 4,
        n_determinizations: int = 8,
        rollout_max_selections: int = 600,
        seed: int = 0,
        stats: SearchStats | None = None,
    ) -> None:
        self._prior = prior
        self._own_deck = [int(c) for c in own_deck_ids]
        self._opp_deck = [int(c) for c in opp_deck_ids]
        self._rollout_self = rollout_self
        self._rollout_opp = rollout_opp
        self._k = max(2, n_candidates)
        self._d = max(1, n_determinizations)
        self._cap = rollout_max_selections
        self._rng = random.Random(seed)
        self.stats = stats if stats is not None else SearchStats()
        # soft targets (ExIt): após cada busca completa, os candidatos
        # avaliados e a taxa média de vitória de cada um (por
        # determinização). None quando a decisão não foi buscada.
        self.last_candidate_values: dict[int, float] | None = None

    # ------------------------------------------------------------------ #
    # Contrato
    # ------------------------------------------------------------------ #

    def __call__(self, obs_dict: dict) -> list[int]:
        self.stats.decisions += 1
        self.last_candidate_values = None
        if not isinstance(obs_dict, dict) or obs_dict.get("select") is None:
            return list(self._own_deck)  # seleção inicial de deck
        answer = self._prior(copy.deepcopy(obs_dict))
        scores = getattr(self._prior, "last_scores", None)
        reason = self._ineligible_reason(obs_dict, answer, scores)
        if reason is not None:
            self.stats.fallback_reasons[reason] += 1
            return answer
        try:
            t0 = time.perf_counter()
            best = self._search(obs_dict, list(scores), answer[0])
            elapsed = time.perf_counter() - t0
            if best is None:
                return answer
            self.stats.searched += 1
            self.stats.search_time_s += elapsed
            if best != answer[0]:
                self.stats.changed += 1
            return [best]
        except Exception:  # noqa: BLE001 — contado, resposta legal do prior
            self.stats.exceptions += 1
            return answer

    # ------------------------------------------------------------------ #
    # Elegibilidade (espelha o filtro validado do counterfactual)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _ineligible_reason(obs_dict: dict, answer: list[int],
                           scores: list[float] | None) -> str | None:
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
        your = state.get("yourIndex")
        players = state.get("players") or []
        if your not in (0, 1) or len(players) != 2:
            return "bad-state"
        opp_active = players[1 - your].get("active") or []
        if not opp_active or opp_active[0] is None:
            return "hidden-opp-active"
        return None

    # ------------------------------------------------------------------ #
    # Busca: 1-ply, determinizações pareadas, folha = rollout terminal
    # ------------------------------------------------------------------ #

    def _search(self, obs_dict: dict, scores: list[float],
                prior_choice: int) -> int | None:
        seat = obs_dict["current"]["yourIndex"]
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        candidates = order[:self._k]
        if prior_choice not in candidates:
            candidates[-1] = prior_choice
        if len(set(candidates)) < 2:
            return None  # nada a comparar — prior decide
        obs_cls = api.to_observation_class(copy.deepcopy(obs_dict))
        values = {i: 0.0 for i in candidates}
        for _ in range(self._d):
            det = determinize(obs_dict, seat, self._own_deck,
                              self._opp_deck, self._rng)
            if det is None:  # multiset não fecha: estrutural neste ponto
                self.stats.fallback_reasons["determinize"] += 1
                return None
            root = api.search_begin(obs_cls, *det, [])
            try:
                for i in candidates:
                    branch = api.search_step(root.searchId, [i])
                    values[i] += self._rollout(branch, seat)
            finally:
                api.search_end()
        self.last_candidate_values = {i: values[i] / self._d
                                      for i in candidates}
        # empate resolve pelo score do prior (busca só troca com evidência)
        return max(candidates, key=lambda i: (values[i], scores[i]))

    def _rollout(self, branch: api.SearchState, seat: int) -> float:
        self.stats.rollouts += 1
        rollout_seed = self._rng.randrange(1 << 30)
        ours = self._rollout_self(rollout_seed)
        theirs = self._rollout_opp(rollout_seed + 1)
        node = branch
        for _ in range(self._cap):
            current = node.observation.current
            if current is not None and current.result != -1:
                if current.result == seat:
                    return 1.0
                return DRAW_VALUE if current.result == RESULT_DRAW else 0.0
            node_dict = _to_dict(node.observation)
            acting = node_dict["current"]["yourIndex"]
            agent = ours if acting == seat else theirs
            node = api.search_step(node.searchId, agent(node_dict))
        self.stats.rollout_caps += 1
        return DRAW_VALUE


__all__ = ["SearchAgent", "SearchStats", "AgentFactory"]
