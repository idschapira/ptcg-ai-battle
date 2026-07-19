"""Expert Iteration (dev): coleta alvos da BUSCA para destilar na rede.

Fundamento (calibração da Camada 2 + PoC 19/Jul): SearchAgent(prior)
joga melhor que o prior BC puro (70% vs ship no PoC), mas ~2,6s/decisão
não é submissível. O loop ExIt destila a busca na rede. Rodada 2
conserta os 3 diagnósticos do PoC: SOFT TARGETS (valores por candidato,
não só argmax), ESCALA (coleta resiliente de dias, 20-50k alvos) e z por
posição para o value head.

Por decisão BUSCADA grava, no schema dos replay datasets:

    states/options_flat/option_counts   encodings crus (encoding.py)
    labels                              ação escolhida PELA BUSCA
    prior_labels                        ação que o prior teria jogado
    soft_targets [N, MAX_OPTIONS]       winrate médio por candidato
                                        avaliado (NaN = não avaliado)
    values                              z do lado buscador (+1/-1/0)
    episode_ids                         base + seed*1000 + jogo

RESILIÊNCIA (rodar dias em background):
  - chunks de --chunk-size jogos, um npz por chunk; retomável: chunks
    existentes são pulados (seeds determinísticas por índice de jogo);
  - arquivo STOP no diretório de saída -> encerra limpo no fim do jogo;
    arquivo PAUSE -> dorme até ser removido (usar durante builds/smokes
    de submissão — o flake de engine sob carga é documentado);
  - janela 21:00-21:30 local: pausa automática (jobs diários de
    replays/ELO/watch não podem competir por CPU);
  - status_<opponent>.json por processo (progresso/taxa) para monitorar.

Rodar da raiz do repo (dev):
    python -m src.rl_models.expert_iter --games 400 --opponent ship \
        --seed 0 --out-dir data/processed/exit_grimmsnarl/r2
Parar tudo:  criar o arquivo <out-dir>/STOP.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
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
from .encoding import (ENCODING_DIM, MAX_OPTIONS, OPTION_DIM, OptionEncoder,
                       StateEncoder)
from .network_agent import NetworkAgent
from .search_agent import SearchAgent, SearchStats

EXIT_DIR: Final[Path] = REPO_ROOT / "data" / "processed" / "exit_grimmsnarl"
GRIMM_DECK: Final[Path] = REPO_ROOT / "data" / "decks" / "meta_grimmsnarl.csv"
GRIMM_WEIGHTS: Final[Path] = REPO_ROOT / "models" / "bc_grimmsnarl.npz"
STATS_PATH: Final[Path] = REPO_ROOT / "models" / "feature_stats.npz"
EPISODE_BASE: Final[int] = 900_000_000

PAUSE_WINDOW: Final[tuple[datetime.time, datetime.time]] = (
    datetime.time(20, 58), datetime.time(21, 32))

OPPONENTS: Final[dict[str, tuple[str, str]]] = {
    "ship": ("deck.csv", "crustle-v3"),
    "spidops": ("data/decks/meta_spidops.csv", "network:bc_spidops_v2"),
    "alakazam": ("data/decks/meta_alakazam.csv", "network:bc_alakazam"),
    "lucario": ("data/decks/seed_mega_lucario.csv", "heuristic"),
    "abomasnow": ("data/decks/placeholder_abomasnow.csv", "heuristic"),
    "dragapult": ("data/decks/seed_dragapult.csv", "heuristic"),
}


class CollectingSearchAgent:
    """Envolve o SearchAgent gravando (estado, opções, busca, prior, soft)."""

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
        self.soft: list[np.ndarray] = []

    def __call__(self, obs_dict: dict) -> list[int]:
        answer = self._search(obs_dict)
        cand_values = self._search.last_candidate_values
        if cand_values is None or len(answer) != 1:
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
            soft = np.full(MAX_OPTIONS, np.nan, dtype=np.float32)
            for idx, value in cand_values.items():
                if 0 <= idx < MAX_OPTIONS:
                    soft[idx] = value
            self.states.append(state_vec)
            self.options.append(legal)
            self.counts.append(legal.shape[0])
            self.labels.append(answer[0])
            self.prior_labels.append(prior_answer[0] if prior_answer else -1)
            self.soft.append(soft)
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


def _wait_if_paused(out_dir: Path) -> bool:
    """True = STOP pedido. Bloqueia em PAUSE e na janela dos jobs diários."""
    while True:
        if (out_dir / "STOP").exists():
            return True
        now = datetime.datetime.now().time()
        if PAUSE_WINDOW[0] <= now <= PAUSE_WINDOW[1]:
            time.sleep(60)
            continue
        if (out_dir / "PAUSE").exists():
            time.sleep(30)
            continue
        return False


def _write_status(out_dir: Path, opponent: str, payload: dict) -> None:
    path = out_dir / f"status_{opponent}.json"
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def collect(games: int, opponent: str, seed: int, out_dir: Path,
            chunk_size: int = 5, budget: tuple[int, int] = (4, 8)) -> None:
    deck_text, plain_spec = OPPONENTS[opponent]
    out_dir.mkdir(parents=True, exist_ok=True)
    index = CardIndex()
    effects = EffectIndex()
    own_ids = read_deck_ids(GRIMM_DECK)
    opp_ids = read_deck_ids(REPO_ROOT / deck_text)

    prior = NetworkAgent(index=index, effects=effects,
                         weights_path=GRIMM_WEIGHTS, stats_path=STATS_PATH)
    if prior._fallback is not None:
        raise SystemExit("pesos do prior BC-Grimmsnarl ausentes")
    opp_factory = _plain_factory(plain_spec, index, effects)
    rollout_opp = (opp_factory if plain_spec == "crustle-v3"
                   else _plain_factory("heuristic", index, effects))
    rollout_self = _plain_factory("heuristic", index, effects)

    total_targets = 0
    total_games = 0
    wins = losses = exceptions = 0
    t0 = time.perf_counter()

    for chunk_start in range(0, games, chunk_size):
        chunk_id = chunk_start // chunk_size
        chunk_path = out_dir / f"shard_{opponent}_s{seed}_c{chunk_id:04d}.npz"
        chunk_games = min(chunk_size, games - chunk_start)
        if chunk_path.exists():
            with np.load(chunk_path) as data:
                total_targets += len(data["labels"])
            total_games += chunk_games
            continue  # retomada: chunk já coletado

        arrays: dict[str, list] = {k: [] for k in (
            "states", "options", "counts", "labels", "prior", "soft",
            "values", "eps")}
        sstats = SearchStats()
        for local in range(chunk_games):
            if _wait_if_paused(out_dir):
                print("STOP detectado — encerrando limpo", flush=True)
                return
            game_index = chunk_start + local
            game_seed = seed + game_index
            search = SearchAgent(prior, own_ids, opp_ids, rollout_self,
                                 rollout_opp, n_candidates=budget[0],
                                 n_determinizations=budget[1],
                                 seed=game_seed, stats=sstats)
            collector = CollectingSearchAgent(search, prior, index, effects)
            our_seat = game_index % 2
            agents = ((collector, opp_factory(game_seed + 10_000))
                      if our_seat == 0
                      else (opp_factory(game_seed + 10_000), collector))
            decks = ((own_ids, opp_ids) if our_seat == 0
                     else (opp_ids, own_ids))
            try:
                result, _ = play_one_game(agents, list(decks[0]),
                                          list(decks[1]))
            except Exception:  # noqa: BLE001
                exceptions += 1
                continue
            z = (0 if result == RESULT_DRAW
                 else (1 if result == our_seat else -1))
            wins += int(z == 1)
            losses += int(z == -1)
            n = len(collector.labels)
            episode_id = EPISODE_BASE + seed * 1000 + game_index
            arrays["states"].extend(collector.states)
            arrays["options"].extend(collector.options)
            arrays["counts"].extend(collector.counts)
            arrays["labels"].extend(collector.labels)
            arrays["prior"].extend(collector.prior_labels)
            arrays["soft"].extend(collector.soft)
            arrays["values"].extend([z] * n)
            arrays["eps"].extend([episode_id] * n)
            total_games += 1
            total_targets += n

        np.savez_compressed(
            chunk_path,
            states=np.asarray(arrays["states"], np.float32).reshape(
                -1, ENCODING_DIM),
            options_flat=(np.concatenate(arrays["options"], axis=0)
                          if arrays["options"]
                          else np.zeros((0, OPTION_DIM), np.float32)),
            option_counts=np.asarray(arrays["counts"], np.uint16),
            labels=np.asarray(arrays["labels"], np.uint16),
            prior_labels=np.asarray(arrays["prior"], np.int32),
            soft_targets=(np.stack(arrays["soft"])
                          if arrays["soft"]
                          else np.zeros((0, MAX_OPTIONS), np.float32)),
            values=np.asarray(arrays["values"], np.int8),
            episode_ids=np.asarray(arrays["eps"], np.int64),
        )
        elapsed = time.perf_counter() - t0
        rate = total_targets / max(elapsed / 3600, 1e-9)
        _write_status(out_dir, opponent, {
            "opponent": opponent, "seed": seed,
            "games_done": total_games, "games_planned": games,
            "targets": total_targets, "targets_per_hour": round(rate),
            "wins": wins, "losses": losses, "exceptions": exceptions,
            "searched": sstats.searched, "search_exceptions": sstats.exceptions,
            "updated": datetime.datetime.now().isoformat(timespec="seconds"),
        })
        print(f"chunk {chunk_id} salvo: {total_games}/{games} jogos, "
              f"{total_targets} alvos ({rate:.0f}/h) — {chunk_path.name}",
              flush=True)

    print(f"coleta completa: {total_games} jogos, {total_targets} alvos, "
          f"{exceptions} exceções ({(time.perf_counter() - t0) / 3600:.1f}h)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=400)
    parser.add_argument("--opponent", choices=tuple(OPPONENTS), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=EXIT_DIR / "r2")
    parser.add_argument("--chunk-size", type=int, default=5)
    args = parser.parse_args()
    collect(args.games, args.opponent, args.seed, args.out_dir,
            args.chunk_size)


if __name__ == "__main__":
    main()
