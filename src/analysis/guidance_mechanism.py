"""Mecanismo do motor de mill, medido junto do winrate.

Medido em 147 jogos REAIS (04-05/Ago): Land Collapse + Explorer's Guidance
no mesmo turno mila 5,00 cartas contra 1,98 sem o combo (2,5×), e a nossa
fatia da corrida de deck-out é só 36-43% — o resto os oponentes queimam
sozinhos. O diagnóstico da indisponibilidade (06/Ago) mostrou que em
84,9% dos turnos de Land Collapse sem combo o Guidance simplesmente NÃO
CHEGOU do deck (58,2% com parte já no descarte, 26,7% sem nenhuma cópia
vista) — só 1,3% tinha as 4 cópias gastas. Logo a alavanca é ACESSO, e o
que este módulo mede é se o acesso de fato subiu.

Métricas por jogo, do NOSSO assento (o assento vem do run_pair):

  guid_in_hand     fração das NOSSAS observações com >=1 Guidance na mão
                   — a disponibilidade bruta, que é o que o acesso ataca
  mill_our_turn    cartas que o deck DELES perde nos NOSSOS turnos
  mill_their_turn  o que eles queimam sozinhos (contexto da corrida)
  our_share        fatia NOSSA da queda total do deck deles
  combo_turns      fração dos nossos turnos que milaram >=4 cartas — a
                   assinatura do Land Collapse COM Ancient Supporter
                   (o combo rende 5, o solo rende ~2), medida por EFEITO
                   porque o observer vê estado, não a ação escolhida

Atribuição por FRONTEIRA DE TURNO: o ataque resolve depois da última
decisão do turno, então a queda só aparece na observação seguinte —
comparar dentro do turno mede zero (erro cometido e corrigido em 04/Ago).

Rodar da raiz do repo:
    python -m src.analysis.guidance_mechanism \
        --deck data/decks/variant_crustle_gb4.csv --cells Grimmsnarl --games 300
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from ..agent_heuristics.crustle_agent import EXPLORERS_GUIDANCE
from ..ingestion.build_card_model import REPO_ROOT
from .nz_mechanism import CELLS, CORRECTED_FIELD

#: queda que só o combo (Land Collapse + Ancient Supporter) explica
COMBO_MILL_FLOOR: Final[int] = 4


@dataclass
class GuidanceMechanism:
    """Uma partida, do NOSSO assento (None-safe em todo caminho)."""

    our_seat: int
    observations: int = 0
    guid_in_hand_obs: int = 0
    result: int | None = None
    turns: int = 0
    #: turno -> [maior deck deles visto no turno, dono do turno]
    _turn: dict[int, list[Any]] = field(default_factory=dict)

    def __call__(self, obs_dict: dict) -> None:
        current = obs_dict.get("current")
        if not isinstance(current, dict):
            return
        players = current.get("players") or []
        if len(players) != 2 or not 0 <= self.our_seat < 2:
            return
        acting = current.get("yourIndex")
        turn = current.get("turn")
        theirs = players[1 - self.our_seat]
        if isinstance(theirs, dict) and turn is not None:
            deck = theirs.get("deckCount")
            if deck is not None:
                slot = self._turn.setdefault(turn, [deck, acting])
                # início do turno = o MAIOR deck visto nele
                slot[0] = max(slot[0], deck)
        if acting != self.our_seat:
            return
        self.observations += 1
        ours = players[self.our_seat]
        hand = ours.get("hand") if isinstance(ours, dict) else None
        if isinstance(hand, list) and any(
                isinstance(c, dict) and c.get("id") == EXPLORERS_GUIDANCE
                for c in hand):
            self.guid_in_hand_obs += 1

    def finish(self, result: int, turns: int) -> None:
        self.result = result
        self.turns = turns

    def _drops(self) -> tuple[list[int], list[int]]:
        """(quedas nos NOSSOS turnos, quedas nos turnos DELES)."""
        ours: list[int] = []
        theirs: list[int] = []
        keys = sorted(self._turn)
        for turn, nxt in zip(keys, keys[1:]):
            drop = self._turn[turn][0] - self._turn[nxt][0]
            if drop < 0:
                continue
            (ours if self._turn[turn][1] == self.our_seat
             else theirs).append(drop)
        return ours, theirs

    @property
    def guid_in_hand(self) -> float:
        return (self.guid_in_hand_obs / self.observations
                if self.observations else 0.0)


def run_cell(deck_path: str, cell: str, games: int, seed: int,
             arm: str = "crustle-v3",
             opponent: str = CORRECTED_FIELD) -> dict[str, Any]:
    """`games` partidas contra uma célula: winrate E mecanismo do mill."""
    from ..deckbuilding.gauntlet import run_pair
    from ..deckbuilding.legality import read_deck_ids
    from ..environment_wrapper.ab_test import ArmMetrics, ArmSpec, arm_factory
    from ..ingestion.build_effect_model import EffectIndex
    from ..ingestion.card_index import CardIndex

    index, effects = CardIndex(), EffectIndex()
    our_deck = read_deck_ids(REPO_ROOT / deck_path)
    opp_deck = read_deck_ids(REPO_ROOT / CELLS[cell])
    spec_a, spec_b = ArmSpec.parse(arm), ArmSpec.parse(opponent)
    metrics_a, metrics_b = ArmMetrics(), ArmMetrics()
    probes: list[GuidanceMechanism] = []

    def on_game(_i: int, a_seat: int, _s: int) -> GuidanceMechanism:
        probe = GuidanceMechanism(our_seat=a_seat)
        probes.append(probe)
        return probe

    pair = run_pair(arm_factory(spec_a, index, effects, metrics_a, our_deck),
                    arm_factory(spec_b, index, effects, metrics_b, opp_deck),
                    our_deck, opp_deck, games, seed, on_game=on_game)

    done = [p for p in probes if p.observations]
    our_all: list[int] = []
    their_all: list[int] = []
    combo_share: list[float] = []
    for probe in done:
        ours, theirs = probe._drops()
        our_all.extend(ours)
        their_all.extend(theirs)
        if ours:
            combo_share.append(
                sum(1 for d in ours if d >= COMBO_MILL_FLOOR) / len(ours))
    total = sum(our_all) + sum(their_all)
    decided = pair.a_wins + pair.b_wins
    return {
        "cell": cell, "deck": deck_path, "games": games,
        "wins": pair.a_wins, "decided": decided,
        "winrate": pair.a_wins / decided if decided else 0.0,
        "exceptions": len(pair.errors),
        "guid_in_hand": (st.mean([p.guid_in_hand for p in done])
                         if done else 0.0),
        "mill_our_turn": st.mean(our_all) if our_all else 0.0,
        "mill_their_turn": st.mean(their_all) if their_all else 0.0,
        "our_share": sum(our_all) / total if total else 0.0,
        "combo_turns": st.mean(combo_share) if combo_share else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", type=str,
                        default="data/decks/variant_crustle_nzaccess.csv")
    parser.add_argument("--cells", nargs="+", default=["Grimmsnarl"])
    parser.add_argument("--games", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arm", type=str, default="crustle-v3")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    t0 = time.perf_counter()
    rows = []
    print(f"deck={args.deck}  piloto={args.arm}")
    print(f"  {'celula':14s}{'winrate':>9s}{'Guid na mao':>13s}"
          f"{'mill/turno':>12s}{'combo>=4':>10s}{'fatia nossa':>13s}{'exc':>5s}")
    for cell in args.cells:
        row = run_cell(args.deck, cell, args.games, args.seed, args.arm)
        rows.append(row)
        print(f"  {cell:14s}{row['winrate']:9.1%}{row['guid_in_hand']:13.1%}"
              f"{row['mill_our_turn']:12.2f}{row['combo_turns']:10.1%}"
              f"{row['our_share']:13.1%}{row['exceptions']:5d}")
    print(f"  wall {(time.perf_counter() - t0) / 60:.1f} min")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1)
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
