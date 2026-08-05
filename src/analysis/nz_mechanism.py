"""Mecanismo da proteção do motor de mill, medido junto do winrate.

A caça de misplay de 04/Ago mediu, em 147 jogos REAIS de ladder, que a
sobrevivência do Great Tusk é o preditor mais forte de vitória (vivo em
18/23 e 36/44 vitórias contra 9/52 e 3/28 derrotas) e que a Neutralization
Zone reduz o dano a ele 3,4× (Grimmsnarl) e 5,6× (Mega Lucario). Um A/B de
deck com N viável não consegue resolver 3-5pp de winrate, mas CONSEGUE
resolver o mecanismo — então o mecanismo é medido junto, e as duas coisas
são reportadas separadamente. Se o mecanismo anda e o winrate não, o
veredito é "direção certa, potência insuficiente", não "nulo".

Métricas por jogo, sempre do NOSSO assento (o assento vem do run_pair, não
é inferido do observation):

  nz_uptime        fração das observações com a Neutralization Zone em jogo
  charm_on_tusk    fração das observações com um Great Tusk NOSSO portando
                   um Pokémon Tool (mede se o piloto sequer anexa a carta —
                   o CrustleAgent v3 não tem regra de Tool, então isto é um
                   gate: uma variante que o piloto não usa é inerte)
  tusk_alive_end   Great Tusk em jogo na última observação
  tusk_damage      dano total sofrido por Great Tusks nossos, por turno

Rodar da raiz do repo:
    python -m src.analysis.nz_mechanism --deck data/decks/variant_crustle_charm4.csv \
        --cells Grimmsnarl --games 120
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from ..agent_heuristics.crustle_agent import GREAT_TUSK, NEUTRAL_ZONE
from ..ingestion.build_card_model import REPO_ROOT

#: células com decklist minerada (as mesmas do pilot_ab)
CELLS: Final[dict[str, str]] = {
    "Alakazam": "data/decks/meta_alakazam.csv",
    "Archaludon": "data/decks/meta_archaludon.csv",
    "Grimmsnarl": "data/decks/meta_grimmsnarl.csv",
    "Mega Lucario": "data/decks/meta_mega_lucario.csv",
    "Starmie": "data/decks/meta_starmie.csv",
    "Kangaskhan": "data/decks/meta_crustle_kangaskhan.csv",
    "Spidops": "data/decks/meta_spidops.csv",
}
CORRECTED_FIELD: Final[str] = "heuristic-tempo-scaled-routing-gust-conserve,8"


@dataclass
class GameMechanism:
    """Uma partida, do NOSSO assento (None-safe em todo caminho)."""

    our_seat: int
    observations: int = 0
    nz_up: int = 0
    tusk_with_tool: int = 0
    tusk_seen: int = 0
    tusk_alive_end: bool = False
    result: int | None = None
    turns: int = 0
    #: serial -> menor hp já visto, e maxHp, para somar dano sofrido
    _tusk_hp: dict[int, list[int]] = field(default_factory=dict)

    def __call__(self, obs_dict: dict) -> None:
        current = obs_dict.get("current")
        if not isinstance(current, dict):
            return
        players = current.get("players") or []
        if len(players) != 2 or not 0 <= self.our_seat < 2:
            return
        ours = players[self.our_seat]
        if not isinstance(ours, dict):
            return
        self.observations += 1

        stadium = current.get("stadium") or []
        top = stadium[0] if stadium else None
        if isinstance(top, dict) and top.get("id") == NEUTRAL_ZONE:
            self.nz_up += 1

        alive = False
        for area in ("active", "bench"):
            for pokemon in ours.get(area) or []:
                if not isinstance(pokemon, dict):
                    continue
                if pokemon.get("id") != GREAT_TUSK:
                    continue
                alive = True
                self.tusk_seen += 1
                if pokemon.get("tools"):
                    self.tusk_with_tool += 1
                serial, hp = pokemon.get("serial"), pokemon.get("hp")
                max_hp = pokemon.get("maxHp")
                if serial is not None and hp is not None and max_hp is not None:
                    slot = self._tusk_hp.setdefault(serial, [max_hp, hp])
                    slot[0] = max(slot[0], max_hp)
                    slot[1] = min(slot[1], hp)
        self.tusk_alive_end = alive

    def finish(self, result: int, turns: int) -> None:
        self.result = result
        self.turns = turns

    @property
    def tusk_damage(self) -> int:
        return sum(max(mx - lo, 0) for mx, lo in self._tusk_hp.values())

    @property
    def nz_uptime(self) -> float:
        return self.nz_up / self.observations if self.observations else 0.0

    @property
    def tool_uptime(self) -> float:
        return self.tusk_with_tool / self.tusk_seen if self.tusk_seen else 0.0


def run_cell(deck_path: str, cell: str, games: int, seed: int,
             arm: str = "crustle-v3",
             opponent: str = CORRECTED_FIELD) -> dict[str, Any]:
    """`games` partidas contra uma célula, medindo winrate E mecanismo."""
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

    probes: list[GameMechanism] = []

    def on_game(_game_index: int, a_seat: int, _seed: int) -> GameMechanism:
        probe = GameMechanism(our_seat=a_seat)
        probes.append(probe)
        return probe

    pair = run_pair(arm_factory(spec_a, index, effects, metrics_a, our_deck),
                    arm_factory(spec_b, index, effects, metrics_b, opp_deck),
                    our_deck, opp_deck, games, seed, on_game=on_game)

    decided = pair.a_wins + pair.b_wins
    done = [p for p in probes if p.observations]
    wins = [p for p in done if p.result == p.our_seat]
    return {
        "cell": cell, "deck": deck_path, "games": games,
        "wins": pair.a_wins, "decided": decided,
        "winrate": pair.a_wins / decided if decided else 0.0,
        "exceptions": len(pair.errors),
        "nz_uptime": st.mean([p.nz_uptime for p in done]) if done else 0.0,
        "tool_on_tusk": st.mean([p.tool_uptime for p in done]) if done else 0.0,
        "tusk_alive_end": (sum(p.tusk_alive_end for p in done) / len(done)
                           if done else 0.0),
        "tusk_alive_end_wins": (sum(p.tusk_alive_end for p in wins) / len(wins)
                                if wins else 0.0),
        "tusk_damage_per_turn": (st.mean([p.tusk_damage / max(p.turns, 1)
                                          for p in done]) if done else 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", type=str, default="deck.csv")
    parser.add_argument("--cells", nargs="+", default=["Grimmsnarl"])
    parser.add_argument("--games", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arm", type=str, default="crustle-v3")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    t0 = time.perf_counter()
    rows = []
    print(f"deck={args.deck}  piloto={args.arm}")
    print(f"  {'celula':14s}{'winrate':>9s}{'NZ up':>8s}{'tool/Tusk':>11s}"
          f"{'Tusk vivo':>11s}{'dano/turno':>12s}{'exc':>5s}")
    for cell in args.cells:
        row = run_cell(args.deck, cell, args.games, args.seed, args.arm)
        rows.append(row)
        print(f"  {cell:14s}{row['winrate']:9.1%}{row['nz_uptime']:8.1%}"
              f"{row['tool_on_tusk']:11.1%}{row['tusk_alive_end']:11.1%}"
              f"{row['tusk_damage_per_turn']:12.1f}{row['exceptions']:5d}")
    print(f"  wall {(time.perf_counter() - t0) / 60:.1f} min")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1)
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
