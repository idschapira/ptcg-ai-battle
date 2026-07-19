"""Expert Iteration (dev): coleta alvos da BUSCA para destilar na rede.

Fundamento (calibração da Camada 2): SearchAgent(prior) joga melhor que
o prior BC puro, mas ~2,6s/decisão não é submissível. O loop ExIt
destila: (1) a busca JOGA e grava a ação escolhida por decisão buscada;
(2) a policy re-treina imitando a busca (mix com os dados do líder);
(3) a rede rápida herda parte da força. Este módulo faz a etapa (1).

Coleta: SearchAgent(prior=BC alvo, orçamento calibrado) joga N partidas
contra um oponente-arm, com assentos alternados. A cada decisão BUSCADA
(single-select, elegível) grava, no MESMO schema dos replay datasets:

    states/options_flat/option_counts   encodings crus (encoding.py)
    labels                              ação escolhida PELA BUSCA
    prior_labels                        ação que o prior teria jogado
    values                              z do lado buscador (+1/-1/0)
    episode_ids                         900_000_000 + seed*1000 + jogo

A taxa de discordância labels != prior_labels mede quanta correção a
busca está injetando (as posições de discordância são o sinal do ExIt).

Rodar da raiz do repo (dev, ~2,6-3,2s/decisão buscada):
    python -m src.rl_models.expert_iter --games 25 --opponent ship \
        --seed 0 --out data/processed/exit_grimmsnarl/shard_ship_s0.npz
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
from typing import Callable, Final

import numpy as np

from ..agent_heuristics.crustle_agent import CrustleAgent
from ..agent_heuristics.heuristic_agent import HeuristicAgent
from ..deckbuilding.legality import read_deck_ids
from ..environment_wrapper.selfplay import RESULT_DRAW, Agent, play_one_game
from ..environment_wrapper.wrapper import EnvironmentWrapper
from ..ingestion.build_card_model import REPO_ROOT
from ..ingestion.build_effect_model import EffectIndex
from ..ingestion.card_index import CardIndex
from . import encoding
from .encoding import ENCODING_DIM, OPTION_DIM, OptionEncoder, StateEncoder
from .network_agent import NetworkAgent
from .search_agent import SearchAgent, SearchStats

EXIT_DIR: Final[Path] = REPO_ROOT / "data" / "processed" / "exit_grimmsnarl"
GRIMM_DECK: Final[Path] = REPO_ROOT / "data" / "decks" / "meta_grimmsnarl.csv"
GRIMM_WEIGHTS: Final[Path] = REPO_ROOT / "models" / "bc_grimmsnarl.npz"
STATS_PATH: Final[Path] = REPO_ROOT / "models" / "feature_stats.npz"
EPISODE_BASE: Final[int] = 900_000_000

# oponente-arm -> (deck path, factory de piloto plano). O piloto plano
# também é o opponent-model dos rollouts (proxy heuristic p/ networks,
# mesma convenção do search_matrix).
OPPONENTS: Final[dict[str, tuple[str, str]]] = {
    "ship": ("deck.csv", "crustle-v3"),
    "spidops": ("data/decks/meta_spidops.csv", "network:bc_spidops_v2"),
    "alakazam": ("data/decks/meta_alakazam.csv", "network:bc_alakazam"),
    "lucario": ("data/decks/seed_mega_lucario.csv", "heuristic"),
    "abomasnow": ("data/decks/placeholder_abomasnow.csv", "heuristic"),
    "dragapult": ("data/decks/seed_dragapult.csv", "heuristic"),
}


class CollectingSearchAgent:
    """Envolve o SearchAgent gravando (estado, opções, busca, prior)."""

    def __init__(self, search: SearchAgent, prior: NetworkAgent,
                 index: CardIndex, effects: EffectIndex) -> None:
        self._search = search
        self._prior = prior
        self._wrapper = EnvironmentWrapper(index)
        self._state_encoder = StateEncoder(index, effects)
        self._option_encoder = OptionEncoder(index, effects)
        self.states: list[np.ndarray] = []
        self.options: list[np.ndarray] = []
        self.counts: list[int] = []
        self.labels: list[int] = []
        self.prior_labels: list[int] = []

    def __call__(self, obs_dict: dict) -> list[int]:
        searched_before = self._search.stats.searched
        answer = self._search(obs_dict)
        if self._search.stats.searched == searched_before or len(answer) != 1:
            return answer  # decisão não buscada -> não vira alvo
        try:
            prior_answer = self._prior(copy.deepcopy(obs_dict))
            obs = self._wrapper.parse(obs_dict)
            state_vec = self._state_encoder.encode(obs)
            option_matrix, mask = encoding.build_action_mask(
                obs, self._state_encoder, self._option_encoder)
            legal = option_matrix[mask]
            if not (0 <= answer[0] < legal.shape[0]):
                return answer
            self.states.append(state_vec)
            self.options.append(legal)
            self.counts.append(legal.shape[0])
            self.labels.append(answer[0])
            self.prior_labels.append(prior_answer[0] if prior_answer else -1)
        except Exception:  # noqa: BLE001 — coleta nunca quebra o jogo
            pass
        return answer


def _plain_factory(spec: str, index: CardIndex,
                   effects: EffectIndex) -> Callable[[int], Agent]:
    if spec == "crustle-v3":
        return lambda s: CrustleAgent(seed=s, index=index, effects=effects,
                                      variant="v3")
    if spec.startswith("network:"):
        weights = REPO_ROOT / "models" / f"{spec.split(':', 1)[1]}.npz"
        agent = NetworkAgent(index=index, effects=effects,
                             weights_path=weights, stats_path=STATS_PATH)
        if agent._fallback is not None:
            raise SystemExit(f"pesos ausentes para arm {spec}")
        return lambda s: agent
    return lambda s: HeuristicAgent(seed=s, index=index, effects=effects)


def collect(games: int, opponent: str, seed: int, out_path: Path,
            budget: tuple[int, int] = (4, 8)) -> None:
    deck_text, plain_spec = OPPONENTS[opponent]
    index = CardIndex()
    effects = EffectIndex()
    own_ids = read_deck_ids(GRIMM_DECK)
    opp_ids = read_deck_ids(REPO_ROOT / deck_text)

    prior = NetworkAgent(index=index, effects=effects,
                         weights_path=GRIMM_WEIGHTS, stats_path=STATS_PATH)
    if prior._fallback is not None:
        raise SystemExit("pesos do prior BC-Grimmsnarl ausentes")
    opp_factory = _plain_factory(plain_spec, index, effects)
    # rollouts: self = heuristic, opp-model = política plana do oponente
    # (proxy heuristic para arms network), como calibrado
    rollout_opp = (opp_factory if plain_spec == "crustle-v3"
                   else _plain_factory("heuristic", index, effects))
    rollout_self = _plain_factory("heuristic", index, effects)

    sstats = SearchStats()
    all_states: list[np.ndarray] = []
    all_options: list[np.ndarray] = []
    all_counts: list[int] = []
    all_labels: list[int] = []
    all_prior: list[int] = []
    all_values: list[int] = []
    all_eps: list[int] = []
    wins = losses = draws = exceptions = 0
    t0 = time.perf_counter()

    for game_index in range(games):
        game_seed = seed + game_index
        search = SearchAgent(prior, own_ids, opp_ids, rollout_self,
                             rollout_opp, n_candidates=budget[0],
                             n_determinizations=budget[1], seed=game_seed,
                             stats=sstats)
        collector = CollectingSearchAgent(search, prior, index, effects)
        our_seat = game_index % 2
        agents = ((collector, opp_factory(game_seed + 10_000))
                  if our_seat == 0
                  else (opp_factory(game_seed + 10_000), collector))
        decks = ((own_ids, opp_ids) if our_seat == 0 else (opp_ids, own_ids))
        try:
            result, _turns = play_one_game(agents, list(decks[0]),
                                           list(decks[1]))
        except Exception:  # noqa: BLE001 — contado como exceção de jogo
            exceptions += 1
            continue
        z = (0 if result == RESULT_DRAW
             else (1 if result == our_seat else -1))
        wins += int(z == 1)
        losses += int(z == -1)
        draws += int(z == 0)
        episode_id = EPISODE_BASE + seed * 1000 + game_index
        n = len(collector.labels)
        all_states.extend(collector.states)
        all_options.extend(collector.options)
        all_counts.extend(collector.counts)
        all_labels.extend(collector.labels)
        all_prior.extend(collector.prior_labels)
        all_values.extend([z] * n)
        all_eps.extend([episode_id] * n)
        print(f"jogo {game_index + 1}/{games} ({opponent}): z={z:+d} "
              f"+{n} alvos ({time.perf_counter() - t0:.0f}s)", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        states=np.asarray(all_states, np.float32).reshape(-1, ENCODING_DIM),
        options_flat=(np.concatenate(all_options, axis=0)
                      if all_options else np.zeros((0, OPTION_DIM), np.float32)),
        option_counts=np.asarray(all_counts, np.uint16),
        labels=np.asarray(all_labels, np.uint16),
        prior_labels=np.asarray(all_prior, np.int32),
        values=np.asarray(all_values, np.int8),
        episode_ids=np.asarray(all_eps, np.int64),
    )
    n = len(all_labels)
    disagree = sum(1 for a, b in zip(all_labels, all_prior) if a != b)
    print(f"\n{opponent}: {games} jogos ({wins}V/{losses}D/{draws}E), "
          f"{n} alvos, discordância busca-vs-prior "
          f"{disagree}/{n} = {disagree / max(n, 1):.1%}")
    print(f"busca: {sstats.summary()}")
    print(f"exceções de jogo: {exceptions} (deve ser 0)")
    print(f"shard: {out_path} ({out_path.stat().st_size:,} bytes, "
          f"{time.perf_counter() - t0:.0f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=25)
    parser.add_argument("--opponent", choices=tuple(OPPONENTS), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    collect(args.games, args.opponent, args.seed, args.out)


if __name__ == "__main__":
    main()
