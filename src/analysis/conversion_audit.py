"""The two clocks of the race: their damage->KO pipeline, and our mill.

Every INPUT variable now matches the ladder (attacks per game, Powerful
Hand share and damage, Rare Candy, Enhanced Hammer, gust targeting). Two
OUTPUT variables still do not, and they are the two clocks that decide
the matchup:

  (a) they knock out 5.49 of our bodies per game on the ladder and 4.02
      internally, taking 3.51 prizes against 2.13;
  (b) our mill runs at 3.00 cards/turn on the ladder and 3.50 internally.

This module decomposes both, and it does so from OBSERVED BOARD STATE
rather than from a damage model. An attack's damage is read as the HP
delta the engine actually applied to the body that was in front — so
"prevented" is not inferred from a rules reading, it is the attacks that
landed and moved nothing.

Sample discipline is the usual one and the reason for it is fresh: the
prize metric in cell_telemetry read ~6.0 for every game because it took a
minimum over the SETUP observations, where the prize list is still empty
(found and fixed 01/Ago). So every counter here is anchored to an event
the engine actually emitted, and the per-side denominators are printed
next to the rates.

Run from the repo root:
    python -m src.analysis.conversion_audit --games 250
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from cg.api import OptionType, SelectContext

from ..deckbuilding.archetype_rules import label_archetype
from ..deckbuilding.legality import read_deck_ids
from ..environment_wrapper.selfplay import play_one_game
from ..ingestion.build_card_model import REPO_ROOT
from ..ingestion.build_effect_model import EffectIndex
from ..ingestion.card_index import CardIndex
from ..ingestion.replays_parse import _iter_decisions
from .cell_telemetry import OUR_DECK_LABEL, OUR_TEAM, SUB_INDEX
from .deadweight_audit import _fighting_card_ids
from .fetch_my_episodes import EPISODES_DIR
from .gust_telemetry import is_protected
from .meta_radar import observed_serials as _observed_serials

GREAT_TUSK: Final[int] = 58
JUMBO_ICE_CREAM: Final[int] = 1147
LAND_COLLAPSE: Final[int] = 62      # Great Tusk's mill (verified: attacks_of)
GIANT_TUSK: Final[int] = 63         # its damage attack, for the split


@dataclass
class ConversionTelemetry:
    """Poolable by addition; one shape for both sides."""

    games: int = 0
    our_turns: int = 0

    # --- their damage -> KO pipeline (attacks aimed at our active) ---
    attacks_on_us: int = 0
    attacks_on_covered: int = 0        # our active held live cover
    damage_dealt: int = 0              # HP the engine actually removed
    attacks_zero_damage: int = 0
    attacks_zero_on_covered: int = 0
    kos: int = 0
    kos_from_attack: int = 0           # body vanished right after an attack
    our_heals: int = 0                 # Jumbo Ice Cream plays by us
    active_covered_at_attack: Counter = field(default_factory=Counter)

    # --- our mill clock ---
    tusk_active_turns: int = 0
    tusk_deaths: int = 0
    tusk_first_death_turn: list[int] = field(default_factory=list)
    # THE CLEAN MEASURE: how many times our mill attack actually fired.
    # A deck-count delta cannot be attributed without care — the window
    # between two of OUR turns contains THEIR draw, so a per-turn "cards
    # milled" figure silently counts the opponent drawing (this is what
    # cell_telemetry's mill_per_turn does, and it is why its 3.00 vs 3.50
    # is not a pure mill rate). Counting the ATTACK is unambiguous.
    land_collapse: int = 0
    giant_tusk: int = 0
    # kept as a cycle-consumption figure, honestly labelled: both sides
    # use the same estimator, so the comparison holds even though the
    # quantity is "their deck consumed per cycle", not "our mill".
    cycle_with_tusk: int = 0
    cycle_without_tusk: int = 0
    tusk_turns_counted: int = 0
    non_tusk_turns_counted: int = 0

    def merge(self, other: "ConversionTelemetry") -> None:
        for name in ("games", "our_turns", "attacks_on_us",
                     "attacks_on_covered", "damage_dealt",
                     "attacks_zero_damage", "attacks_zero_on_covered",
                     "kos", "kos_from_attack", "our_heals",
                     "tusk_active_turns", "tusk_deaths", "land_collapse",
                     "giant_tusk", "cycle_with_tusk",
                     "cycle_without_tusk", "tusk_turns_counted",
                     "non_tusk_turns_counted"):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.tusk_first_death_turn.extend(other.tusk_first_death_turn)
        self.active_covered_at_attack.update(other.active_covered_at_attack)


class Collector:
    """Consumes (agent_index, obs_dict, action); source-agnostic."""

    def __init__(self, index: CardIndex, fighting: frozenset[int],
                 wrapper=None) -> None:
        self._index = index
        self._fighting = fighting
        self._wrapper = wrapper
        self.total = ConversionTelemetry()
        self._reset()

    def _reset(self) -> None:
        self._our_seat: int | None = None
        self._hp: dict[int, int] = {}        # our serial -> last seen hp
        self._pending: dict | None = None    # attack awaiting its HP delta
        self._turns_seen: set[int] = set()
        self._tusk_serials: set[int] = set()
        self._tusk_dead = False
        self._opp_deck: int | None = None
        self._turn_start_deck: dict[int, int] = {}
        self._turn_tusk: dict[int, bool] = {}
        self._game = ConversionTelemetry()

    def begin_game(self, our_seat: int) -> None:
        self._reset()
        self._our_seat = our_seat

    # ------------------------------------------------------------------ #

    def _our_board(self, players: list) -> list[dict]:
        try:
            ours = players[self._our_seat]
        except (IndexError, TypeError):
            return []
        board = list(ours.get("active") or []) + list(ours.get("bench") or [])
        return [p for p in board if isinstance(p, dict)]

    def _our_active(self, players: list) -> dict | None:
        try:
            active = players[self._our_seat].get("active") or []
        except (IndexError, TypeError):
            return None
        return active[0] if active and isinstance(active[0], dict) else None

    def observe(self, agent_index: int, obs_dict: dict,
                action: list[int]) -> None:
        if self._our_seat is None:
            return
        state = obs_dict.get("current") or {}
        players = state.get("players") or []
        if len(players) < 2:
            return
        turn = state.get("turn")
        turn = turn if isinstance(turn, int) else 0

        self._resolve_pending(players)
        self._track_bodies(players, turn)
        self._track_mill(players, turn, agent_index)

        select = obs_dict.get("select") or {}
        if select.get("context") != int(SelectContext.MAIN) or not action:
            return
        options = select.get("option") or []
        picked_index = action[0]
        if not 0 <= picked_index < len(options):
            return
        picked = options[picked_index]
        if not isinstance(picked, dict):
            return

        if agent_index == self._our_seat:
            if picked.get("type") == int(OptionType.PLAY):
                card_id = self._resolve_play(obs_dict, picked)
                if card_id == JUMBO_ICE_CREAM:
                    self._game.our_heals += 1
            elif picked.get("type") == int(OptionType.ATTACK):
                if picked.get("attackId") == LAND_COLLAPSE:
                    self._game.land_collapse += 1
                elif picked.get("attackId") == GIANT_TUSK:
                    self._game.giant_tusk += 1
            return

        if picked.get("type") != int(OptionType.ATTACK):
            return
        active = self._our_active(players)
        if active is None:
            return
        covered = is_protected(active, self._fighting)
        self._game.attacks_on_us += 1
        self._game.attacks_on_covered += int(covered)
        self._game.active_covered_at_attack[covered] += 1
        self._pending = {"serial": active.get("serial"),
                         "hp": active.get("hp"), "covered": covered}

    def _resolve_pending(self, players: list) -> None:
        """Read the HP the engine actually removed, one observation later."""
        pending = self._pending
        if pending is None:
            return
        serial = pending["serial"]
        before = pending["hp"]
        if not isinstance(serial, int) or not isinstance(before, int):
            self._pending = None
            return
        now = None
        for pokemon in self._our_board(players):
            if pokemon.get("serial") == serial:
                now = pokemon.get("hp")
                break
        if now is None:                       # the body is gone: a knockout
            self._game.kos_from_attack += 1
            self._game.damage_dealt += before
            self._pending = None
            return
        if not isinstance(now, int):
            self._pending = None
            return
        dealt = max(0, before - now)
        self._game.damage_dealt += dealt
        if dealt == 0:
            self._game.attacks_zero_damage += 1
            self._game.attacks_zero_on_covered += int(pending["covered"])
        self._pending = None

    def _resolve_play(self, obs_dict: dict, option: dict) -> int | None:
        if self._wrapper is None:
            return None
        try:
            obs = self._wrapper.parse(obs_dict)
            for candidate in (obs.select.option or []):
                if (candidate.type == OptionType.PLAY
                        and candidate.index == option.get("index")
                        and candidate.area == option.get("area")):
                    return self._wrapper.resolve_card_id(obs, candidate)
        except Exception:
            return None
        return None

    def _track_bodies(self, players: list, turn: int) -> None:
        present: dict[int, int] = {}
        tusk_now: set[int] = set()
        for pokemon in self._our_board(players):
            serial = pokemon.get("serial")
            if not isinstance(serial, int):
                continue
            present[serial] = pokemon.get("hp") or 0
            if pokemon.get("id") == GREAT_TUSK:
                tusk_now.add(serial)
        for serial in set(self._hp) - set(present):
            self._game.kos += 1
            if serial in self._tusk_serials:
                self._game.tusk_deaths += 1
                if not self._tusk_dead:
                    self._tusk_dead = True
                    self._game.tusk_first_death_turn.append(turn)
        self._hp = present
        self._tusk_serials = tusk_now

    def _track_mill(self, players: list, turn: int, agent_index: int) -> None:
        """Opponent deck consumed during OUR turns is our mill: they do not
        draw on our turn, so the delta over a turn we acted in is ours."""
        try:
            opp_deck = players[1 - self._our_seat].get("deckCount")
        except (IndexError, TypeError):
            return
        if not isinstance(opp_deck, int):
            return
        if agent_index != self._our_seat:
            self._opp_deck = opp_deck
            return
        active = self._our_active(players)
        tusk = active is not None and active.get("id") == GREAT_TUSK
        if turn not in self._turn_start_deck:
            self._turn_start_deck[turn] = opp_deck
            self._turn_tusk[turn] = tusk
            self._game.our_turns += 1
            self._game.tusk_active_turns += int(tusk)
        else:
            # tusk counts for the turn if it was ever the active in it
            self._turn_tusk[turn] = self._turn_tusk[turn] or tusk
        self._turn_start_deck[turn] = max(self._turn_start_deck[turn],
                                          opp_deck)
        self._opp_deck = opp_deck

    def end_game(self, final_players: list | None = None) -> None:
        turns = sorted(self._turn_start_deck)
        for i, turn in enumerate(turns):
            start = self._turn_start_deck[turn]
            end = (self._turn_start_deck[turns[i + 1]]
                   if i + 1 < len(turns) else self._opp_deck)
            if end is None:
                continue
            milled = max(0, start - end)
            if self._turn_tusk.get(turn):
                self._game.cycle_with_tusk += milled
                self._game.tusk_turns_counted += 1
            else:
                self._game.cycle_without_tusk += milled
                self._game.non_tusk_turns_counted += 1
        self._game.games = 1
        self.total.merge(self._game)
        self._reset()


# --------------------------------------------------------------------------
# drivers
# --------------------------------------------------------------------------

def from_replays(episodes_dir: Path, archetype: str, team: str,
                 allowed: set[int] | None, index: CardIndex,
                 fighting: frozenset[int], wrapper) -> ConversionTelemetry:
    collector = Collector(index, fighting, wrapper)
    for path in sorted(episodes_dir.glob("*.json")):
        stem = os.path.splitext(path.name)[0]
        if allowed is not None and stem.isdigit() and int(stem) not in allowed:
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                replay = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        names = ((replay.get("info") or {}).get("TeamNames")) or []
        if team not in names:
            continue
        seat = names.index(team)
        revealed: dict[int, Counter] = {0: Counter(), 1: Counter()}
        for (player, _s), card_id in _observed_serials(replay).items():
            card = index.get_card(card_id)
            if card is not None and player in revealed:
                revealed[player][card.card_name] += 1
        if label_archetype(revealed[seat]) != OUR_DECK_LABEL:
            continue
        if label_archetype(revealed[1 - seat]) != archetype:
            continue
        collector.begin_game(seat)
        for agent_index, obs_dict, action in _iter_decisions(replay):
            collector.observe(agent_index, obs_dict, action)
        collector.end_game()
    return collector.total


class _Recording:
    def __init__(self, inner, collector: Collector, seat: int) -> None:
        self._inner, self._collector, self._seat = inner, collector, seat

    def __call__(self, obs_dict: dict) -> list[int]:
        answer = self._inner(obs_dict)
        try:
            self._collector.observe(self._seat, obs_dict, answer)
        except Exception:
            pass
        return answer


def from_selfplay(our_deck: list[int], opp_deck: list[int], our_arm: str,
                  opp_arm: str, games: int, seed: int, index: CardIndex,
                  effects: EffectIndex, fighting: frozenset[int],
                  wrapper) -> ConversionTelemetry:
    from ..environment_wrapper.ab_test import ArmMetrics, ArmSpec, arm_factory

    collector = Collector(index, fighting, wrapper)
    make_us = arm_factory(ArmSpec.parse(our_arm), index, effects,
                          ArmMetrics(), our_deck)
    make_them = arm_factory(ArmSpec.parse(opp_arm), index, effects,
                            ArmMetrics(), opp_deck)
    for game_index in range(games):
        our_seat = game_index % 2
        collector.begin_game(our_seat)
        us = _Recording(make_us(seed + game_index), collector, our_seat)
        them = _Recording(make_them(seed + 10_000 + game_index), collector,
                          1 - our_seat)
        agents = (us, them) if our_seat == 0 else (them, us)
        decks = ((our_deck, opp_deck) if our_seat == 0
                 else (opp_deck, our_deck))
        play_one_game(agents, list(decks[0]), list(decks[1]))
        collector.end_game()
    return collector.total


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def _mean(values: list) -> float | None:
    return sum(values) / len(values) if values else None


def _div(a: float, b: float) -> float | None:
    return a / b if b else None


def _fmt(value, spec: str = ".2f") -> str:
    return format(value, spec) if value is not None else "-"


def compare(rows: dict[str, ConversionTelemetry]) -> dict:
    labels = list(rows)

    def line(name: str, values: list, spec: str = ".2f") -> None:
        cells = "".join(f"{_fmt(v, spec):>16s}" for v in values)
        print(f"  {name:44s}{cells}")

    print("\n" + "=" * 108)
    print("CONVERSAO — o funil dano->KO deles, e o relogio do nosso mill")
    print("=" * 108)
    print(f"  {'métrica':44s}" + "".join(f"{l:>16s}" for l in labels))
    print("  " + "-" * 104)
    line("jogos", [float(rows[l].games) for l in labels], ".0f")
    line("turnos em que fomos consultados/jogo",
         [_div(rows[l].our_turns, rows[l].games) for l in labels])
    print("  --- funil: ataques deles -> KOs nossos " + "-" * 64)
    line("ataques no nosso ativo/jogo",
         [_div(rows[l].attacks_on_us, rows[l].games) for l in labels])
    line("...com o nosso ativo COBERTO",
         [_div(rows[l].attacks_on_covered, rows[l].attacks_on_us)
          for l in labels], ".1%")
    line("dano efetivo por ataque (HP medido)",
         [_div(rows[l].damage_dealt, rows[l].attacks_on_us) for l in labels])
    line("dano efetivo/jogo",
         [_div(rows[l].damage_dealt, rows[l].games) for l in labels], ".0f")
    line("ataques que moveram ZERO",
         [_div(rows[l].attacks_zero_damage, rows[l].attacks_on_us)
          for l in labels], ".1%")
    line("...e o alvo estava coberto",
         [_div(rows[l].attacks_zero_on_covered, rows[l].attacks_zero_damage)
          for l in labels], ".1%")
    line("KOs nossos/jogo",
         [_div(rows[l].kos, rows[l].games) for l in labels])
    line(">> KOs por ataque (CONVERSAO)",
         [_div(rows[l].kos, rows[l].attacks_on_us) for l in labels], ".1%")
    line("Jumbo Ice Cream (nossas curas)/jogo",
         [_div(rows[l].our_heals, rows[l].games) for l in labels])
    print("  --- relogio do mill: uptime do Great Tusk " + "-" * 61)
    line("share dos NOSSOS turnos com Great Tusk ativo",
         [_div(rows[l].tusk_active_turns, rows[l].our_turns)
          for l in labels], ".1%")
    line("mortes do Great Tusk/jogo",
         [_div(rows[l].tusk_deaths, rows[l].games) for l in labels])
    line("turno da 1a morte do Great Tusk",
         [_mean(rows[l].tusk_first_death_turn) for l in labels])
    line("jogos em que ele chega a morrer",
         [_div(len(rows[l].tusk_first_death_turn), rows[l].games)
          for l in labels], ".1%")
    line(">> Land Collapse (mill) usados/jogo",
         [_div(rows[l].land_collapse, rows[l].games) for l in labels])
    line(">> ...por turno consultado",
         [_div(rows[l].land_collapse, rows[l].our_turns) for l in labels],
         ".1%")
    line("Giant Tusk (dano) usados/jogo",
         [_div(rows[l].giant_tusk, rows[l].games) for l in labels])
    line("deck deles consumido/ciclo, COM Tusk*",
         [_div(rows[l].cycle_with_tusk, rows[l].tusk_turns_counted)
          for l in labels])
    line(">> deck deles consumido/ciclo, SEM Tusk**",
         [_div(rows[l].cycle_without_tusk, rows[l].non_tusk_turns_counted)
          for l in labels])
    print("  * ciclo = nosso turno + o turno deles, entao inclui o SAQUE "
          "deles.")
    print("  ** sem Great Tusk no ativo o Land Collapse NAO pode disparar, "
          "entao essa linha e")
    print("     o consumo que o oponente faz do PROPRIO deck.")
    print("    Mesmo estimador nos dois lados: a comparacao vale, o valor "
          "absoluto NAO e mill puro.")

    payload = {}
    for label in labels:
        t = rows[label]
        payload[label] = {
            "games": t.games,
            "our_turns_per_game": _div(t.our_turns, t.games),
            "attacks_on_us_per_game": _div(t.attacks_on_us, t.games),
            "covered_share": _div(t.attacks_on_covered, t.attacks_on_us),
            "damage_per_attack": _div(t.damage_dealt, t.attacks_on_us),
            "damage_per_game": _div(t.damage_dealt, t.games),
            "zero_damage_share": _div(t.attacks_zero_damage, t.attacks_on_us),
            "kos_per_game": _div(t.kos, t.games),
            "kos_per_attack": _div(t.kos, t.attacks_on_us),
            "heals_per_game": _div(t.our_heals, t.games),
            "tusk_uptime": _div(t.tusk_active_turns, t.our_turns),
            "tusk_deaths_per_game": _div(t.tusk_deaths, t.games),
            "tusk_first_death_turn": _mean(t.tusk_first_death_turn),
            "land_collapse_per_game": _div(t.land_collapse, t.games),
            "land_collapse_per_our_turn": _div(t.land_collapse, t.our_turns),
            "giant_tusk_per_game": _div(t.giant_tusk, t.games),
            "cycle_consumed_with_tusk": _div(t.cycle_with_tusk,
                                             t.tusk_turns_counted),
            "cycle_consumed_without_tusk": _div(t.cycle_without_tusk,
                                                t.non_tusk_turns_counted),
        }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archetype", type=str,
                        default="Alakazam box (non-ex)")
    parser.add_argument("--opp-deck", type=Path,
                        default=Path("data/decks/meta_alakazam.csv"))
    parser.add_argument("--our-deck", type=Path, default=Path("deck.csv"))
    parser.add_argument("--our-arm", type=str, default="crustle-v3")
    parser.add_argument("--opp-arms", nargs="+",
                        default=["heuristic-tempo-scaled"])
    parser.add_argument("--games", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes-dir", type=Path, default=EPISODES_DIR)
    parser.add_argument("--team", type=str, default=OUR_TEAM)
    parser.add_argument("--submission-index", type=Path, default=SUB_INDEX)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    from ..environment_wrapper.wrapper import EnvironmentWrapper
    index, effects = CardIndex(), EffectIndex()
    wrapper = EnvironmentWrapper(index)
    fighting = _fighting_card_ids(index)
    allowed = None
    if args.submission_index.exists():
        with open(args.submission_index, encoding="utf-8") as fh:
            allowed = {int(i) for i in
                       (json.load(fh).get("episode_ids") or [])}

    rows: dict[str, ConversionTelemetry] = {}
    rows["REAL"] = from_replays(args.episodes_dir, args.archetype, args.team,
                                allowed, index, fighting, wrapper)
    our_deck = read_deck_ids(args.our_deck)
    opp_deck = read_deck_ids(args.opp_deck)
    for arm in args.opp_arms:
        # keep the WHOLE spec in the label: two -conserve arms differ only
        # in the floor after the comma, and cutting there silently made
        # them overwrite each other in this dict
        label = arm.replace("heuristic-tempo-scaled", "h-t-s")
        rows[label] = from_selfplay(
            our_deck, opp_deck, args.our_arm, arm, args.games, args.seed,
            index, effects, fighting, wrapper)

    payload = compare(rows)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        print(f"\njson: {args.json}")


if __name__ == "__main__":
    main()
