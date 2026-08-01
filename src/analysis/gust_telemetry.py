"""The BYPASS hypothesis: does the real Alakazam drag an UNPROTECTED body?

Three rounds have now matched an attack variable to the ladder (tempo,
damage valuation, attack band) and the Alakazam cell still reads ~+44pp.
Powerful Hand only reaches the ACTIVE spot, and Mist/Rock nullify it
outright (damage counters are an EFFECT — verified against the engine in
tests/test_effect_prevention_contract.py). So the remaining way a real
opponent wins is not attacking harder, it is attacking someone ELSE:
gust the one body that has no cover into the Active Spot, then hit it.

That makes a sharp, falsifiable measurement. When the engine hands the
opponent a SWITCH over OUR bench, it is offering a CHOICE, and the
choice set usually contains both covered and bare bodies. The statistic
that decides the hypothesis is therefore conditional:

    given a choice set with BOTH a protected and an unprotected body,
    how often is the unprotected one taken?

A pilot that ignores cover answers ~50% (or worse — HeuristicAgent
scores promotion by HP and printed damage, so it drags our BEST body).
A pilot that plays the bypass answers ~100%.

Everything is measured by ONE collector fed by two drivers (real replays
and internal games), exactly like cell_telemetry: measuring the two
sides with two pieces of code would put the comparison at the mercy of
its own bugs. Sample discipline on the real side is the usual one —
submission-id filter, archetype sentinel on BOTH decks, our seat read
from the replay's own team names.

Protection is read from the observation that OFFERS the switch, which is
the only moment that can matter and is public information (the engine
exposes energyCards for both boards — verified in the replays).

Run from the repo root:
    python -m src.analysis.gust_telemetry --games 300
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import cg.api as api
from cg.api import AreaType, SelectContext

from ..deckbuilding.archetype_rules import label_archetype
from ..deckbuilding.legality import read_deck_ids
from ..environment_wrapper.selfplay import play_one_game
from ..environment_wrapper.wrapper import EnvironmentWrapper
from ..ingestion.build_card_model import REPO_ROOT
from ..ingestion.build_effect_model import EffectIndex
from ..ingestion.card_index import CardIndex
from ..ingestion.replays_parse import _iter_decisions
from .cell_telemetry import OUR_DECK_LABEL, OUR_TEAM, SUB_INDEX
from .deadweight_audit import (FIGHTING_TYPE, MIST_ENERGY, ROCK_FIGHTING,
                               _fighting_card_ids)
from .fetch_my_episodes import EPISODES_DIR
from .meta_radar import observed_serials as _observed_serials


def covers(energy_id: int, host_is_fighting: bool) -> bool:
    """Engine-verified prevention matrix (same clause as the audits):
    Mist covers any host, Rock Fighting only a {F} one."""
    if energy_id == MIST_ENERGY:
        return True
    if energy_id == ROCK_FIGHTING:
        return host_is_fighting
    return False


def is_protected(pokemon: dict, fighting_ids: frozenset[int]) -> bool:
    """Would an effect-based attack (Powerful Hand) be nullified here?"""
    if not isinstance(pokemon, dict):
        return False
    host_is_fighting = pokemon.get("id") in fighting_ids
    for card in pokemon.get("energyCards") or []:
        if isinstance(card, dict) and isinstance(card.get("id"), int):
            if covers(card["id"], host_is_fighting):
                return True
    return False


@dataclass
class GustTelemetry:
    """Everything poolable by addition, so both sides share one shape."""

    games: int = 0
    our_losses: int = 0

    gusts: int = 0                       # opponent chose OUR active
    gust_turns: list[int] = field(default_factory=list)
    gust_forced: int = 0                 # only one legal option: no choice
    # the conditional that decides the hypothesis
    real_choices: int = 0                # both kinds were on the table
    chose_unprotected: int = 0
    # unconditional colour of the target
    target_protected: int = 0
    target_unprotected: int = 0
    target_hp: list[int] = field(default_factory=list)
    target_name: Counter = field(default_factory=Counter)

    # THE OTHER BYPASS: strip the cover before attacking. Enhanced
    # Hammer discards a SPECIAL energy, so the engine only ever offers
    # specials — which makes the interesting question not "did they take
    # a special" but "did they take one that was actually COVERING its
    # host". A Rock Fighting sitting on a non-{F} body covers nothing,
    # and burning the Hammer on it is a wasted card.
    hammers: int = 0
    hammer_forced: int = 0
    hammer_real_choices: int = 0         # live cover AND dead card offered
    hammer_took_live_cover: int = 0
    hammer_took_cover: int = 0           # unconditional

    # consequence
    our_kos: int = 0
    kos_on_gusted: int = 0               # victim had been gusted in
    kos_same_turn_as_gust: int = 0
    our_kos_in_losses: int = 0
    kos_on_gusted_in_losses: int = 0

    def merge(self, other: "GustTelemetry") -> None:
        for name in ("games", "our_losses", "gusts", "gust_forced",
                     "real_choices", "chose_unprotected", "target_protected",
                     "target_unprotected", "our_kos", "kos_on_gusted",
                     "kos_same_turn_as_gust", "our_kos_in_losses",
                     "kos_on_gusted_in_losses", "hammers", "hammer_forced",
                     "hammer_real_choices", "hammer_took_live_cover",
                     "hammer_took_cover"):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.gust_turns.extend(other.gust_turns)
        self.target_hp.extend(other.target_hp)
        self.target_name.update(other.target_name)


class Collector:
    """Consumes (agent_index, obs_dict, action); source-agnostic."""

    def __init__(self, index: CardIndex,
                 fighting_ids: frozenset[int]) -> None:
        self._index = index
        self._fighting = fighting_ids
        self.total = GustTelemetry()
        self._reset()

    def _reset(self) -> None:
        self._our_seat: int | None = None
        self._alive: dict[int, int] = {}      # our serial -> last seen turn
        self._gusted: dict[int, int] = {}     # our serial -> turn gusted in
        self._game = GustTelemetry()

    def begin_game(self, our_seat: int) -> None:
        self._reset()
        self._our_seat = our_seat

    # -- helpers -------------------------------------------------------- #

    def _our_board(self, players: list) -> list[dict]:
        try:
            ours = players[self._our_seat]
        except (IndexError, TypeError):
            return []
        board = list(ours.get("active") or []) + list(ours.get("bench") or [])
        return [p for p in board if isinstance(p, dict)]

    def _pokemon_at(self, players: list, option: dict) -> dict | None:
        try:
            player = players[option.get("playerIndex")]
        except (IndexError, TypeError):
            return None
        area = option.get("area")
        index = option.get("index")
        if not isinstance(index, int):
            return None
        key = ("active" if area == int(AreaType.ACTIVE)
               else "bench" if area == int(AreaType.BENCH) else None)
        if key is None:
            return None
        try:
            entry = (player.get(key) or [])[index]
        except (IndexError, TypeError):
            return None
        return entry if isinstance(entry, dict) else None

    # -- the stream ----------------------------------------------------- #

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

        self._track_our_bodies(players, turn)
        if agent_index == self._our_seat:
            return

        select = obs_dict.get("select") or {}
        context = select.get("context")
        if context == int(SelectContext.DISCARD_ENERGY):
            self._observe_hammer(players, select, action)
            return
        if context != int(SelectContext.SWITCH):
            return
        options = select.get("option") or []
        ours = [o for o in options
                if isinstance(o, dict)
                and o.get("playerIndex") == self._our_seat]
        if not ours or not action:
            return           # a SWITCH over their OWN board is not a gust
        picked_index = action[0]
        if not 0 <= picked_index < len(options):
            return
        picked = options[picked_index]
        if picked.get("playerIndex") != self._our_seat:
            return

        target = self._pokemon_at(players, picked)
        if target is None:
            return
        self._game.gusts += 1
        self._game.gust_turns.append(turn)
        serial = target.get("serial")
        if isinstance(serial, int):
            self._gusted[serial] = turn

        protected = is_protected(target, self._fighting)
        self._game.target_protected += int(protected)
        self._game.target_unprotected += int(not protected)
        if isinstance(target.get("hp"), int):
            self._game.target_hp.append(target["hp"])
        card = self._index.get_card(target.get("id"))
        self._game.target_name[card.card_name if card else "?"] += 1

        if len(ours) < 2:
            self._game.gust_forced += 1
            return
        colours = {is_protected(self._pokemon_at(players, o), self._fighting)
                   for o in ours
                   if self._pokemon_at(players, o) is not None}
        if len(colours) > 1:              # a REAL choice was on the table
            self._game.real_choices += 1
            self._game.chose_unprotected += int(not protected)

    def _live_cover(self, players: list, option: dict) -> bool | None:
        """Is THIS attached energy actually providing cover right now?

        None when the option does not resolve to one of our energies.
        """
        host = self._pokemon_at(players, option)
        if host is None:
            return None
        cards = host.get("energyCards") or []
        index = option.get("energyIndex")
        if not isinstance(index, int) or not 0 <= index < len(cards):
            return None
        card = cards[index]
        if not isinstance(card, dict) or not isinstance(card.get("id"), int):
            return None
        return covers(card["id"], host.get("id") in self._fighting)

    def _observe_hammer(self, players: list, select: dict,
                        action: list[int]) -> None:
        options = select.get("option") or []
        ours = [o for o in options
                if isinstance(o, dict)
                and o.get("playerIndex") == self._our_seat]
        if not ours or not action:
            return
        picked_index = action[0]
        if not 0 <= picked_index < len(options):
            return
        picked = options[picked_index]
        if picked.get("playerIndex") != self._our_seat:
            return
        taken = self._live_cover(players, picked)
        if taken is None:
            return
        self._game.hammers += 1
        self._game.hammer_took_cover += int(taken)
        if len(ours) < 2:
            self._game.hammer_forced += 1
            return
        colours = {c for c in (self._live_cover(players, o) for o in ours)
                   if c is not None}
        if len(colours) > 1:      # a live cover AND a dead card on offer
            self._game.hammer_real_choices += 1
            self._game.hammer_took_live_cover += int(taken)

    def _track_our_bodies(self, players: list, turn: int) -> None:
        """Our Pokémon leaving the board is a KO (they do not retreat off
        it); a victim carrying a gust mark is a post-gust knockout."""
        present = {}
        for pokemon in self._our_board(players):
            serial = pokemon.get("serial")
            if isinstance(serial, int):
                present[serial] = turn
        for serial in set(self._alive) - set(present):
            self._game.our_kos += 1
            gust_turn = self._gusted.pop(serial, None)
            if gust_turn is not None:
                self._game.kos_on_gusted += 1
                if gust_turn == self._alive[serial]:
                    self._game.kos_same_turn_as_gust += 1
        self._alive = present

    def end_game(self, our_result: bool | None) -> None:
        self._game.games = 1
        if our_result is False:
            self._game.our_losses = 1
            self._game.our_kos_in_losses = self._game.our_kos
            self._game.kos_on_gusted_in_losses = self._game.kos_on_gusted
        self.total.merge(self._game)
        self._reset()


# --------------------------------------------------------------------------
# driver 1: real replays
# --------------------------------------------------------------------------

def from_replays(episodes_dir: Path, archetype: str, team: str,
                 allowed: set[int] | None, index: CardIndex,
                 fighting_ids: frozenset[int]) -> GustTelemetry:
    collector = Collector(index, fighting_ids)
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
        rewards = replay.get("rewards") or []
        won = None
        if len(rewards) == 2 and rewards[0] is not None:
            won = rewards[seat] > rewards[1 - seat]
        collector.end_game(won)
    return collector.total


# --------------------------------------------------------------------------
# driver 2: internal games
# --------------------------------------------------------------------------

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
                  effects: EffectIndex,
                  fighting_ids: frozenset[int]) -> GustTelemetry:
    from ..environment_wrapper.ab_test import ArmMetrics, ArmSpec, arm_factory

    collector = Collector(index, fighting_ids)
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
        result, _turns = play_one_game(agents, list(decks[0]), list(decks[1]))
        collector.end_game(result == our_seat if result in (0, 1) else None)
    return collector.total


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def _mean(values: list) -> float | None:
    return sum(values) / len(values) if values else None


def _fmt(value, spec: str = ".2f") -> str:
    return format(value, spec) if value is not None else "-"


def compare(rows: dict[str, GustTelemetry]) -> dict:
    labels = list(rows)
    payload: dict = {}

    def line(name: str, values: list, spec: str = ".2f") -> None:
        cells = "".join(f"{_fmt(v, spec):>16s}" for v in values)
        print(f"  {name:40s}{cells}")

    print("\n" + "=" * 100)
    print("GUSTING — o oponente arrasta QUEM para o nosso ativo?")
    print("=" * 100)
    print(f"  {'métrica':40s}" + "".join(f"{l:>16s}" for l in labels))
    print("  " + "-" * 96)
    line("jogos", [float(rows[l].games) for l in labels], ".0f")
    line("gusts/jogo",
         [rows[l].gusts / rows[l].games if rows[l].games else None
          for l in labels])
    line("turno médio do gust",
         [_mean(rows[l].gust_turns) for l in labels])
    line("gusts SEM escolha (1 opção)",
         [rows[l].gust_forced / rows[l].gusts if rows[l].gusts else None
          for l in labels], ".1%")
    print("  " + "-" * 96)
    line("alvo DESPROTEGIDO (todos os gusts)",
         [rows[l].target_unprotected / rows[l].gusts if rows[l].gusts else None
          for l in labels], ".1%")
    line(">> ESCOLHAS REAIS (protegido E desprotegido na mesa)",
         [float(rows[l].real_choices) for l in labels], ".0f")
    line(">> ...e puxaram o DESPROTEGIDO",
         [rows[l].chose_unprotected / rows[l].real_choices
          if rows[l].real_choices else None for l in labels], ".1%")
    line("HP médio do alvo puxado",
         [_mean(rows[l].target_hp) for l in labels])
    print("  " + "-" * 96)
    line("Enhanced Hammer: descartes/jogo",
         [rows[l].hammers / rows[l].games if rows[l].games else None
          for l in labels])
    line("...levou energia que COBRIA de fato",
         [rows[l].hammer_took_cover / rows[l].hammers if rows[l].hammers
          else None for l in labels], ".1%")
    line(">> ESCOLHAS REAIS (cobertura viva E carta morta)",
         [float(rows[l].hammer_real_choices) for l in labels], ".0f")
    line(">> ...e levaram a COBERTURA VIVA",
         [rows[l].hammer_took_live_cover / rows[l].hammer_real_choices
          if rows[l].hammer_real_choices else None for l in labels], ".1%")
    print("  " + "-" * 96)
    line("KOs nossos/jogo",
         [rows[l].our_kos / rows[l].games if rows[l].games else None
          for l in labels])
    line("share dos KOs num corpo ARRASTADO",
         [rows[l].kos_on_gusted / rows[l].our_kos if rows[l].our_kos else None
          for l in labels], ".1%")
    line("...no MESMO turno do gust",
         [rows[l].kos_same_turn_as_gust / rows[l].kos_on_gusted
          if rows[l].kos_on_gusted else None for l in labels], ".1%")
    line("share nas nossas DERROTAS",
         [rows[l].kos_on_gusted_in_losses / rows[l].our_kos_in_losses
          if rows[l].our_kos_in_losses else None for l in labels], ".1%")

    for label in labels:
        t = rows[label]
        print(f"\n  [{label}] alvos mais puxados: "
              f"{', '.join(f'{n} x{c}' for n, c in t.target_name.most_common(5))}")
        payload[label] = {
            "games": t.games, "gusts": t.gusts,
            "gusts_per_game": t.gusts / t.games if t.games else None,
            "gust_turn_mean": _mean(t.gust_turns),
            "forced_share": t.gust_forced / t.gusts if t.gusts else None,
            "target_unprotected_share": (t.target_unprotected / t.gusts
                                         if t.gusts else None),
            "real_choices": t.real_choices,
            "chose_unprotected_share": (t.chose_unprotected / t.real_choices
                                        if t.real_choices else None),
            "target_hp_mean": _mean(t.target_hp),
            "our_kos": t.our_kos,
            "kos_on_gusted_share": (t.kos_on_gusted / t.our_kos
                                    if t.our_kos else None),
            "kos_same_turn_share": (t.kos_same_turn_as_gust / t.kos_on_gusted
                                    if t.kos_on_gusted else None),
            "kos_on_gusted_in_losses_share": (
                t.kos_on_gusted_in_losses / t.our_kos_in_losses
                if t.our_kos_in_losses else None),
            "hammers": t.hammers,
            "hammers_per_game": t.hammers / t.games if t.games else None,
            "hammer_took_cover_share": (t.hammer_took_cover / t.hammers
                                        if t.hammers else None),
            "hammer_real_choices": t.hammer_real_choices,
            "hammer_took_live_cover_share": (
                t.hammer_took_live_cover / t.hammer_real_choices
                if t.hammer_real_choices else None),
            "targets": dict(t.target_name.most_common(10)),
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
    parser.add_argument("--games", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes-dir", type=Path, default=EPISODES_DIR)
    parser.add_argument("--team", type=str, default=OUR_TEAM)
    parser.add_argument("--submission-index", type=Path, default=SUB_INDEX)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    index, effects = CardIndex(), EffectIndex()
    fighting_ids = _fighting_card_ids(index)
    allowed = None
    if args.submission_index.exists():
        with open(args.submission_index, encoding="utf-8") as fh:
            allowed = {int(i) for i in
                       (json.load(fh).get("episode_ids") or [])}

    rows: dict[str, GustTelemetry] = {}
    rows["REAL"] = from_replays(args.episodes_dir, args.archetype, args.team,
                                allowed, index, fighting_ids)
    our_deck = read_deck_ids(args.our_deck)
    opp_deck = read_deck_ids(args.opp_deck)
    for arm in args.opp_arms:
        rows[arm.split(",")[0]] = from_selfplay(
            our_deck, opp_deck, args.our_arm, arm, args.games, args.seed,
            index, effects, fighting_ids)

    payload = compare(rows)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        print(f"\njson: {args.json}")


if __name__ == "__main__":
    main()
