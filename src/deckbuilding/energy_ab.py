"""A/B de densidade de energia (Epic 4.5): braços de deck, piloto FIXO.

Pergunta do experimento: subir a densidade de energia do Crustle
LibraryOut (8/60) QUEBRA o torno diagnosticado na rodada 2 de misplays
(muro morre cedo E o mill mal completa) ou só o desloca (salva o muro
mas rouba consistência do mill)?

Desenho: cada braço (csv em data/decks/) enfrenta o campo inteiro de
seeds com o MESMO piloto nosso — CrustleAgent(variant="v3"), o ship — e
o oponente de histórico (HeuristicAgent, como no gate de campo do
ab_test). Assentos alternam; o loop de jogo é selfplay.play_one_game
(nunca reimplementado). Um recorder-tap guarda o último estado de
decisão de cada jogo e o mecanismo de derrota/vitória é classificado
pelo MESMO estimador da análise de ladder (episode_review._classify,
inferência do golpe final incluída) — números comparáveis aos da caça
de misplays.

Motor não-reproduzível (ver ab_test): todo winrate sai com IC de Wilson
95%. Uma linha JSONL por jogo em data/processed/energy_ab/ (gitignored)
para reanálise sem re-simular.

Rodar da raiz do repo:
    python -m src.deckbuilding.energy_ab --games 100 --seed 0
    python -m src.deckbuilding.energy_ab --arms data/decks/seed_crustle.csv
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from ..agent_heuristics.crustle_agent import CrustleAgent
from ..agent_heuristics.heuristic_agent import HeuristicAgent
from ..analysis.episode_review import _classify
from ..environment_wrapper.selfplay import RESULT_DRAW, play_one_game
from ..ingestion.build_card_model import REPO_ROOT
from ..ingestion.build_effect_model import EffectIndex
from ..ingestion.card_index import CardIndex
from .legality import read_deck_ids, validate_deck

DECKS_DIR: Final[Path] = REPO_ROOT / "data" / "decks"
OUT_DIR: Final[Path] = REPO_ROOT / "data" / "processed" / "energy_ab"

DEFAULT_ARMS: Final[tuple[str, ...]] = (
    "data/decks/seed_crustle.csv",          # baseline: ship, 8/60 energia
    "data/decks/candidate_crustle_e10.csv",  # -2 Pokégear, +2 {F}: 10/60
    "data/decks/candidate_crustle_e12.csv",  # -2 Pokégear -2 Xerosic, +4 {F}
)

# Campo de oponentes: os seeds curados MENOS o próprio crustle (os braços
# são variações dele) e MENOS candidatos (candidate_*) de rodadas futuras.
FIELD_PREFIXES: Final[tuple[str, ...]] = ("seed_", "placeholder_")


class _StateTap:
    """Recorder mínimo: retém o último estado de decisão do jogo.

    play_one_game não expõe o estado terminal (o loop retorna ao vê-lo);
    o tap fica 1 ação atrás do fim — exatamente o cenário que a
    inferência de golpe final de episode_review._classify cobre.
    """

    __slots__ = ("last",)

    def __init__(self) -> None:
        self.last: dict | None = None

    def record_step(self, obs_dict: dict, answer: list[int],
                    scores: object = None) -> None:
        current = obs_dict.get("current")
        if isinstance(current, dict):
            self.last = current


@dataclass
class ArmStats:
    """Agregado de um braço contra o campo inteiro."""

    games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    exceptions: list[str] = field(default_factory=list)
    turns: list[int] = field(default_factory=list)
    loss_mechanisms: Counter = field(default_factory=Counter)
    win_mechanisms: Counter = field(default_factory=Counter)
    opp_deck_at_loss: list[int] = field(default_factory=list)
    our_deck_at_deckout_win: list[int] = field(default_factory=list)

    @property
    def decided(self) -> int:
        return self.wins + self.losses


def field_decks(arms: set[Path]) -> dict[str, Path]:
    """Seeds/placeholders de data/decks/, exceto os braços e o crustle."""
    field_: dict[str, Path] = {}
    for path in sorted(DECKS_DIR.glob("*.csv")):
        if not path.name.startswith(FIELD_PREFIXES):
            continue
        if path.resolve() in arms or "crustle" in path.stem:
            continue
        name = path.stem
        for prefix in FIELD_PREFIXES:
            name = name.removeprefix(prefix)
        field_[name] = path
    return field_


def _load(path: Path, index: CardIndex) -> list[int]:
    ids = read_deck_ids(path)
    report = validate_deck(ids, index)
    if not report.ok:
        for error in report.errors:
            print(f"  - {error}")
        raise SystemExit(f"deck {path} ILEGAL — abortando")
    return ids


def run_arm(arm_name: str, arm_deck: list[int],
            field_: dict[str, list[int]], n_games: int, seed: int,
            index: CardIndex, effects: EffectIndex,
            out_path: Path) -> ArmStats:
    """n_games por pareamento contra cada deck do campo, assentos alternando."""
    stats = ArmStats()
    with open(out_path, "w", encoding="utf-8") as sink:
        for opp_name, opp_deck in field_.items():
            for game_index in range(n_games):
                ours = CrustleAgent(seed=seed + game_index, index=index,
                                    effects=effects, variant="v3")
                theirs = HeuristicAgent(seed=seed + 10_000 + game_index,
                                        index=index, effects=effects)
                our_seat = game_index % 2
                agents = (ours, theirs) if our_seat == 0 else (theirs, ours)
                decks = ((arm_deck, opp_deck) if our_seat == 0
                         else (opp_deck, arm_deck))
                tap = _StateTap()
                stats.games += 1
                try:
                    winner, turns = play_one_game(
                        agents, list(decks[0]), list(decks[1]), tap)
                except Exception as exc:  # noqa: BLE001 — gate de exceções
                    stats.exceptions.append(
                        f"{opp_name} game {game_index}: "
                        f"{type(exc).__name__}: {exc}")
                    continue
                stats.turns.append(turns)
                winner_index = None if winner == RESULT_DRAW else winner
                result, mechanism = _classify(tap.last, our_seat,
                                              winner_index)
                players = (tap.last or {}).get("players") or [{}, {}]
                us = players[our_seat] if our_seat < len(players) else {}
                them = (players[1 - our_seat]
                        if 1 - our_seat < len(players) else {})
                if result == "win":
                    stats.wins += 1
                    stats.win_mechanisms[mechanism] += 1
                    # vitória por deck-out inclui a corrida apertada, que
                    # _classify rotula "out-milled (control mirror)"
                    if "decked out" in mechanism or "out-milled" in mechanism:
                        stats.our_deck_at_deckout_win.append(
                            us.get("deckCount") or 0)
                elif result == "loss":
                    stats.losses += 1
                    stats.loss_mechanisms[mechanism] += 1
                    stats.opp_deck_at_loss.append(them.get("deckCount") or 0)
                else:
                    stats.draws += 1
                sink.write(json.dumps({
                    "arm": arm_name, "opponent": opp_name,
                    "game": game_index, "our_seat": our_seat,
                    "result": result, "mechanism": mechanism,
                    "turns": turns,
                    "our_deck": us.get("deckCount"),
                    "opp_deck": them.get("deckCount"),
                    "our_prizes": len(us.get("prize") or []),
                    "opp_prizes": len(them.get("prize") or []),
                }) + "\n")
    return stats


def _median(values: list[int]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    return (ordered[mid] if len(ordered) % 2
            else (ordered[mid - 1] + ordered[mid]) / 2)


def print_arm(name: str, s: ArmStats) -> None:
    from ..environment_wrapper.ab_test import wilson_interval
    lo, hi = wilson_interval(s.wins, s.decided)
    print(f"\n=== braço {name} ===")
    print(f"  jogos {s.games}  decididos {s.decided}  empates {s.draws}  "
          f"exceções {len(s.exceptions)} (deve ser 0)")
    print(f"  winrate {s.wins / max(s.decided, 1):.1%}  "
          f"IC95 [{lo:.1%}, {hi:.1%}]  "
          f"turnos médios {sum(s.turns) / max(len(s.turns), 1):.1f}")
    total_losses = max(s.losses, 1)
    print(f"  derrotas ({s.losses}):")
    for mech, count in s.loss_mechanisms.most_common():
        print(f"    {mech:32s} {count:3d}  ({count / total_losses:.0%})")
    print(f"  vitórias ({s.wins}):")
    for mech, count in s.win_mechanisms.most_common():
        print(f"    {mech:32s} {count:3d}")
    print(f"  margem de mill — deck do oponente nas derrotas: "
          f"mediana {_median(s.opp_deck_at_loss)}  "
          f"(≤5 em {sum(1 for v in s.opp_deck_at_loss if v <= 5)} "
          f"de {s.losses})")
    print(f"  buffer nosso nas vitórias por deck-out: "
          f"mediana {_median(s.our_deck_at_deckout_win)}")
    for error in s.exceptions[:5]:
        print(f"  EXCEÇÃO {error}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=100,
                        help="jogos por pareamento braço×oponente")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arms", type=Path, nargs="+",
                        default=[Path(p) for p in DEFAULT_ARMS])
    args = parser.parse_args()

    index = CardIndex()
    effects = EffectIndex()
    arm_paths = {p.resolve() for p in args.arms}
    field_paths = field_decks(arm_paths)
    field_ = {name: _load(path, index)
              for name, path in field_paths.items()}
    print(f"campo: {sorted(field_)}  ({args.games} jogos/pareamento)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, ArmStats] = {}
    for arm_path in args.arms:
        deck = _load(arm_path, index)
        name = arm_path.stem
        energies = sum(1 for cid in deck
                       if (index.get_card(cid) or object())
                       and getattr(index.get_card(cid), "stage_code", 0)
                       in (1, 2))
        print(f"\nbraço {name}: LEGAL, {energies}/60 energia")
        t0 = time.perf_counter()
        stats = run_arm(name, deck, field_, args.games, args.seed,
                        index, effects, OUT_DIR / f"{name}.jsonl")
        print(f"  ({stats.games} jogos em {time.perf_counter() - t0:.0f}s)")
        results[name] = stats

    for name, stats in results.items():
        print_arm(name, stats)


if __name__ == "__main__":
    main()
