"""Matriz de matchups do 2º arquétipo com oponentes search-augmentados.

Destravada pela calibração da Camada 2 (search_calibration): no orçamento
4x8 o campo interno reproduz o ladder (63,0% [53,2–71,8] vs real 52%
[34–69]). Aqui cada lado é <deck.csv>@<arm> com arms:

    crustle-v3        CrustleAgent v3 (o ship)
    heuristic         HeuristicAgent genérico
    search:heuristic  SearchAgent com prior heurístico (deck sem clone)
    search:network    SearchAgent com prior BC (default: BC-Alakazam)

Opponent-model dos rollouts de um lado buscador = a política PLANA do
outro lado (crustle-v3/heuristic; para search:network o proxy é
heuristic — o prior BC custaria ~2x por rollout para ~55% de fidelidade,
escolha reportada). Rollout do próprio lado = heuristic, como calibrado.

Rodar da raiz do repo (dev offline):
    python -m src.analysis.search_matrix --games 30 --seed 0 \
        --a data/decks/meta_spidops.csv@search:heuristic \
        --b deck.csv@crustle-v3
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable, Final

from ..agent_heuristics.crustle_agent import CrustleAgent
from ..agent_heuristics.heuristic_agent import HeuristicAgent
from ..deckbuilding.gauntlet import run_pair
from ..environment_wrapper.ab_test import wilson_interval
from ..environment_wrapper.selfplay import Agent
from ..ingestion.build_effect_model import EffectIndex
from ..ingestion.card_index import CardIndex
from ..rl_models.network_agent import NetworkAgent
from ..rl_models.search_agent import SearchAgent, SearchStats
from .search_calibration import BC_STATS, BC_WEIGHTS, _load_deck, _parse_budget

ARMS: Final[tuple[str, ...]] = ("crustle-v3", "heuristic",
                                "search:heuristic", "search:network")


def _plain_factory(arm: str, index: CardIndex,
                   effects: EffectIndex) -> Callable[[int], Agent]:
    """Política plana de um arm (também o opponent-model de rollout)."""
    if arm == "crustle-v3":
        return lambda s: CrustleAgent(seed=s, index=index, effects=effects,
                                      variant="v3")
    # search:network -> proxy heuristic nos rollouts (ver docstring)
    return lambda s: HeuristicAgent(seed=s, index=index, effects=effects)


def _side_factory(arm: str, own_ids: list[int], opp_ids: list[int],
                  opp_arm: str, budget: tuple[int, int], index: CardIndex,
                  effects: EffectIndex, stats: SearchStats,
                  bc_prior: NetworkAgent | None,
                  ) -> Callable[[int], Agent]:
    if arm in ("crustle-v3", "heuristic"):
        return _plain_factory(arm, index, effects)
    k, d = budget
    rollout_self = lambda s: HeuristicAgent(seed=s, index=index,  # noqa: E731
                                            effects=effects)
    rollout_opp = _plain_factory(opp_arm, index, effects)
    if arm == "search:network":
        if bc_prior is None:
            raise SystemExit("prior BC indisponível (pesos ausentes?)")
        return lambda s: SearchAgent(bc_prior, own_ids, opp_ids,
                                     rollout_self, rollout_opp,
                                     n_candidates=k, n_determinizations=d,
                                     seed=s, stats=stats)
    return lambda s: SearchAgent(
        HeuristicAgent(seed=s + 500_000, index=index, effects=effects),
        own_ids, opp_ids, rollout_self, rollout_opp,
        n_candidates=k, n_determinizations=d, seed=s, stats=stats)


def _parse_side(text: str, flag: str) -> tuple[Path, str]:
    if "@" not in text:
        raise SystemExit(f"{flag} deve ser <deck.csv>@<arm>")
    deck_text, arm = text.split("@", 1)
    if arm not in ARMS:
        raise SystemExit(f"arm desconhecido '{arm}' (opções: {ARMS})")
    return Path(deck_text), arm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", required=True, help="<deck.csv>@<arm>")
    parser.add_argument("--b", required=True, help="<deck.csv>@<arm>")
    parser.add_argument("--games", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", default="4x8", metavar="KxD",
                        help="orçamento calibrado da busca (default 4x8)")
    parser.add_argument("--weights", type=Path, default=BC_WEIGHTS)
    parser.add_argument("--stats", type=Path, default=BC_STATS)
    args = parser.parse_args()

    index = CardIndex()
    effects = EffectIndex()
    path_a, arm_a = _parse_side(args.a, "--a")
    path_b, arm_b = _parse_side(args.b, "--b")
    ids_a = _load_deck(path_a, index)
    ids_b = _load_deck(path_b, index)
    budget = _parse_budget(args.budget)

    bc_prior: NetworkAgent | None = None
    if "search:network" in (arm_a, arm_b):
        bc_prior = NetworkAgent(index=index, effects=effects,
                                weights_path=args.weights,
                                stats_path=args.stats)
        if bc_prior._fallback is not None:
            bc_prior = None

    stats_a, stats_b = SearchStats(), SearchStats()
    make_a = _side_factory(arm_a, ids_a, ids_b, arm_b, budget, index,
                           effects, stats_a, bc_prior)
    make_b = _side_factory(arm_b, ids_b, ids_a, arm_a, budget, index,
                           effects, stats_b, bc_prior)

    label = f"{path_a.name}@{arm_a} vs {path_b.name}@{arm_b}"
    print(f"[matriz] {label} | busca {args.budget} | {args.games} jogos "
          f"| seed {args.seed}")
    t0 = time.perf_counter()
    res = run_pair(make_a, make_b, ids_a, ids_b, args.games, args.seed)
    wall = time.perf_counter() - t0
    decided = res.a_wins + res.b_wins
    lo, hi = wilson_interval(res.a_wins, decided)
    print(f"  A {res.a_wins} / B {res.b_wins} / draws {res.draws} "
          f"({wall:.0f}s, {wall / max(res.games, 1):.0f}s/jogo)")
    print(f"  winrate A = {res.winrate_a:.1%}  IC95 [{lo:.1%}, {hi:.1%}]")
    print(f"  engine exceptions {len(res.errors)} (deve ser 0)")
    for error in res.errors[:5]:
        print(f"    {error}")
    for side, sstats in (("A", stats_a), ("B", stats_b)):
        if sstats.decisions:
            print(f"  busca {side}: {sstats.summary()}")


if __name__ == "__main__":
    main()
