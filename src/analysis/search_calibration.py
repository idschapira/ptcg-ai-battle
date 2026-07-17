"""Sweep de calibração da Camada 2: SearchAgent(prior=BC) como oponente.

Curva do mecanismo: Crustle-e10 (CrustleAgent v3, o ship) vs
meta_alakazam pilotado por SearchAgent(prior = BC do Yushin Ito) em
orçamentos crescentes KxD (K candidatos do prior x D determinizações).
Referências: BC pura (busca 0) = 79,5% [73,4–84,5] nosso (16/Jul);
ladder real = 52% [34–69]. A busca fecha o gap se o nosso winrate
interno cair em direção ao IC real conforme o orçamento sobe.

Escolhas reportáveis (ver docstring do SearchAgent): folha = rollout
até o terminal (value head 5D descalibrado, não usado); opponent-model
dos rollouts default = o oponente VERDADEIRO do matchup (crustle-v3 —
avaliação interna de campo, não runtime); rollout do próprio lado
default = heuristic (barato; --rollout-self prior avalia a continuação
fiel ao clone, ~5x mais caro).

Rodar da raiz do repo (dev offline):
    python -m src.analysis.search_calibration --games 12 --budgets 3x4
    python -m src.analysis.search_calibration --games 100 \
        --budgets 3x4 4x8 5x12
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Final

from ..agent_heuristics.crustle_agent import CrustleAgent
from ..agent_heuristics.heuristic_agent import HeuristicAgent
from ..deckbuilding.gauntlet import run_pair
from ..deckbuilding.legality import read_deck_ids, validate_deck
from ..environment_wrapper.ab_test import wilson_interval
from ..ingestion.build_card_model import REPO_ROOT
from ..ingestion.build_effect_model import EffectIndex
from ..ingestion.card_index import CardIndex
from ..rl_models.network_agent import NetworkAgent
from ..rl_models.search_agent import SearchAgent, SearchStats

OUR_DECK: Final[Path] = REPO_ROOT / "deck.csv"
SEARCH_DECK: Final[Path] = REPO_ROOT / "data" / "decks" / "meta_alakazam.csv"
BC_WEIGHTS: Final[Path] = REPO_ROOT / "models" / "bc_alakazam.npz"
BC_STATS: Final[Path] = REPO_ROOT / "models" / "feature_stats.npz"
REAL_CI: Final[tuple[float, float]] = (0.34, 0.69)  # ladder vs Alakazam


def _parse_budget(text: str) -> tuple[int, int]:
    try:
        k, d = text.lower().split("x", 1)
        return int(k), int(d)
    except ValueError as exc:
        raise SystemExit(f"orçamento inválido '{text}' (formato KxD)") from exc


def _load_deck(path: Path, index: CardIndex) -> list[int]:
    ids = read_deck_ids(path)
    report = validate_deck(ids, index)
    if not report.ok:
        for error in report.errors:
            print(f"  - {error}")
        raise SystemExit(f"deck {path} é ILEGAL — abortando")
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=100,
                        help="jogos por ponto da curva")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budgets", nargs="+", default=["3x4", "4x8"],
                        metavar="KxD",
                        help="orçamentos de busca (K candidatos x D "
                             "determinizações), em ordem crescente")
    parser.add_argument("--weights", type=Path, default=BC_WEIGHTS,
                        help="prior .npz (default: BC-Alakazam)")
    parser.add_argument("--stats", type=Path, default=BC_STATS,
                        help="feature stats pareados com o prior")
    parser.add_argument("--search-deck", type=Path, default=SEARCH_DECK)
    parser.add_argument("--our-deck", type=Path, default=OUR_DECK)
    parser.add_argument("--rollout-self", choices=("heuristic", "prior"),
                        default="heuristic",
                        help="política do lado buscador nos rollouts")
    parser.add_argument("--rollout-opp", choices=("crustle-v3", "heuristic"),
                        default="crustle-v3",
                        help="opponent-model nos rollouts (default: o "
                             "oponente verdadeiro do matchup)")
    args = parser.parse_args()

    index = CardIndex()
    effects = EffectIndex()
    our_ids = _load_deck(args.our_deck, index)
    search_ids = _load_deck(args.search_deck, index)

    prior = NetworkAgent(index=index, effects=effects,
                         weights_path=args.weights, stats_path=args.stats)
    if prior._fallback is not None:
        raise SystemExit(f"pesos do prior ausentes: {args.weights}")
    print(f"prior: {args.weights.name} + {args.stats.name} | "
          f"rollout-self={args.rollout_self} rollout-opp={args.rollout_opp}")
    print(f"A = {args.our_deck.name}@crustle-v3 (ship)  vs  "
          f"B = {args.search_deck.name}@search(prior)")
    print(f"referências: BC pura 79,5% [73,4–84,5] | "
          f"ladder real 52% [{REAL_CI[0]:.0%}–{REAL_CI[1]:.0%}]\n")

    if args.rollout_self == "prior":
        rollout_self = lambda s: prior  # noqa: E731 — compartilhado, sem estado
    else:
        rollout_self = lambda s: HeuristicAgent(seed=s, index=index,  # noqa: E731
                                                effects=effects)
    if args.rollout_opp == "crustle-v3":
        rollout_opp = lambda s: CrustleAgent(seed=s, index=index,  # noqa: E731
                                             effects=effects, variant="v3")
    else:
        rollout_opp = lambda s: HeuristicAgent(seed=s, index=index,  # noqa: E731
                                               effects=effects)

    curve: list[tuple[str, float, float, float]] = []
    for budget_text in args.budgets:
        k, d = _parse_budget(budget_text)
        sstats = SearchStats()
        make_crustle = lambda s: CrustleAgent(  # noqa: E731
            seed=s, index=index, effects=effects, variant="v3")
        make_search = lambda s: SearchAgent(  # noqa: E731
            prior, search_ids, our_ids, rollout_self, rollout_opp,
            n_candidates=k, n_determinizations=d, seed=s, stats=sstats)
        t0 = time.perf_counter()
        res = run_pair(make_crustle, make_search, our_ids, search_ids,
                       args.games, args.seed)
        wall = time.perf_counter() - t0
        decided = res.a_wins + res.b_wins
        lo, hi = wilson_interval(res.a_wins, decided)
        inside = REAL_CI[0] <= res.winrate_a <= REAL_CI[1]
        curve.append((budget_text, res.winrate_a, lo, hi))
        print(f"=== busca {k}x{d} ({args.games} jogos, {wall:.0f}s, "
              f"{wall / max(res.games, 1):.1f}s/jogo) ===")
        print(f"  nosso winrate = {res.winrate_a:.1%}  "
              f"IC95 [{lo:.1%}, {hi:.1%}]  "
              f"(A {res.a_wins} / B {res.b_wins} / draws {res.draws})")
        print(f"  ponto dentro do IC real [34%,69%]? "
              f"{'SIM — CALIBRA' if inside else 'não'}")
        print(f"  engine exceptions {len(res.errors)} (deve ser 0)")
        for error in res.errors[:5]:
            print(f"    {error}")
        print(f"  busca: {sstats.summary()}\n")

    print("curva (orçamento -> nosso winrate):  busca 0 = 79,5%")
    for budget_text, rate, lo, hi in curve:
        print(f"  {budget_text:>6s}  {rate:.1%}  [{lo:.1%}, {hi:.1%}]")


if __name__ == "__main__":
    main()
