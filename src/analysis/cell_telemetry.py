"""Why does an internal cell lie? Measure the SAME behaviours both sides.

The Alakazam cell reads 83.5% internally and 35.0% on the real ladder —
+48.5pp. A winrate gap that large is not a statistics problem, it is a
behaviour problem: the simulated opponent is not playing the deck the way
real opponents play it. This module finds out which behaviours differ.

The whole point is ONE code path. A collector consumes a stream of
(agent_index, obs_dict, action) and never learns where the stream came
from; real replays and internal games are just two drivers feeding it.
Measuring the two sides with two different pieces of code would put the
comparison at the mercy of the comparison's own bugs.

Behaviours tracked, chosen because each has a plausible mechanical path
to the winrate in THIS matchup:

  enhanced hammer   Enhanced Hammer (1081) discards a SPECIAL Energy from
                    one of our Pokémon — that is precisely our Mist/Rock.
                    If simulated opponents underplay it, our prevention
                    survives artificially and the cell flatters us.
  energy stripped   the outcome that matters, measured directly on our
                    board: special energies that left a host WHICH
                    SURVIVED (an effect removed them) versus those that
                    died with their host (a KO, not disruption).
  powerful hand     Alakazam's win condition, and its damage is 2 counters
                    per card in the opponent's HAND, so the hand size at
                    the moment of the attack is part of the damage roll.
  online turn       when Alakazam (a Stage 2) first reaches the board.
  disruption        Boss's Orders / Xerosic's Machinations / Judge plays.
  prizes            how many prizes the opponent actually takes per game.

Sample discipline for the real side: submission-id filter, deck sentinel,
and our board is read only from OUR OWN seat's observations.

Run from the repo root:
    python -m src.analysis.cell_telemetry --games 300
    python -m src.analysis.cell_telemetry --json out.json
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
from cg.api import OptionType

from ..deckbuilding.archetype_rules import label_archetype
from ..deckbuilding.legality import read_deck_ids
from ..environment_wrapper.selfplay import play_one_game
from ..environment_wrapper.wrapper import EnvironmentWrapper
from ..ingestion.build_card_model import REPO_ROOT
from ..ingestion.build_effect_model import EffectIndex
from ..ingestion.card_index import CardIndex
from ..ingestion.replays_parse import _iter_decisions
from .fetch_my_episodes import EPISODES_DIR
from .meta_radar import observed_serials as _observed_serials

OUR_TEAM: Final[str] = "Ilan Schapira"
OUR_DECK_LABEL: Final[str] = "Crustle mill (ours)"
SUB_INDEX: Final[Path] = (REPO_ROOT / "data" / "processed" /
                          "episodes_index" / "sub_54917180.json")

MIST: Final[int] = 11
ROCK: Final[int] = 20
OUR_SPECIAL: Final[frozenset[int]] = frozenset({MIST, ROCK})

ENHANCED_HAMMER: Final[int] = 1081
BOSS_ORDERS: Final[int] = 1182
XEROSIC: Final[int] = 1197
JUDGE: Final[int] = 1226        # resolved at runtime if the name differs
ALAKAZAM: Final[int] = 743
POWERFUL_HAND: Final[int] = 1072

# Cards worth counting on the opponent's side, in two groups.
# DISRUPTION attacks our resources; RECURSION attacks our WIN CONDITION by
# putting cards back into the deck we are trying to empty (Sacred Ash
# alone returns up to 5 Pokémon, and Judge shuffles both hands back in).
# An internal opponent that underplays recursion decks itself out and
# hands us a win the real ladder never gives.
DISRUPTION: Final[dict[int, str]] = {
    ENHANCED_HAMMER: "Enhanced Hammer",
    BOSS_ORDERS: "Boss's Orders",
    XEROSIC: "Xerosic's Machinations",
}
RECURSION: Final[dict[int, str]] = {
    1129: "Sacred Ash (5 de volta)",
    1184: "Lana's Aid (3 de volta)",
    1097: "Night Stretcher",
    1213: "Judge (mao -> deck)",
    1079: "Rare Candy",
}
TRACKED: Final[dict[int, str]] = {**DISRUPTION, **RECURSION}


@dataclass
class Telemetry:
    """Everything poolable by addition, so real and internal use one shape."""

    games: int = 0
    turns: list[int] = field(default_factory=list)
    our_wins: int = 0

    # opponent card plays, per card id
    plays: Counter = field(default_factory=Counter)
    play_turns: dict[int, list[int]] = field(default_factory=dict)

    # opponent attacks
    attacks: Counter = field(default_factory=Counter)
    powerful_hand_handsize: list[int] = field(default_factory=list)

    # our special energies leaving the board, by cause
    stripped_survivor: int = 0      # host lived -> an effect removed it
    lost_with_host: int = 0         # host was KO'd
    our_kos: int = 0                # our bodies that left the board
    attached_special: int = 0       # how many we managed to attach at all

    alakazam_online_turn: list[int] = field(default_factory=list)
    opp_prizes_taken: list[int] = field(default_factory=list)

    # THE MILL RACE. Our win condition is decking them out, so how far
    # the mill got — and how fast — is the other half of every game the
    # prize count alone cannot explain.
    opp_deck_low: list[int] = field(default_factory=list)
    our_deck_low: list[int] = field(default_factory=list)
    opp_deck_end: list[int] = field(default_factory=list)
    mill_per_turn: list[float] = field(default_factory=list)

    def merge(self, other: "Telemetry") -> None:
        self.games += other.games
        self.turns.extend(other.turns)
        self.our_wins += other.our_wins
        self.plays.update(other.plays)
        for card_id, turns in other.play_turns.items():
            self.play_turns.setdefault(card_id, []).extend(turns)
        self.attacks.update(other.attacks)
        self.powerful_hand_handsize.extend(other.powerful_hand_handsize)
        self.stripped_survivor += other.stripped_survivor
        self.lost_with_host += other.lost_with_host
        self.our_kos += other.our_kos
        self.attached_special += other.attached_special
        self.alakazam_online_turn.extend(other.alakazam_online_turn)
        self.opp_prizes_taken.extend(other.opp_prizes_taken)
        self.opp_deck_low.extend(other.opp_deck_low)
        self.our_deck_low.extend(other.our_deck_low)
        self.opp_deck_end.extend(other.opp_deck_end)
        self.mill_per_turn.extend(other.mill_per_turn)


class Collector:
    """Consumes (agent_index, obs_dict, action); source-agnostic."""

    def __init__(self, wrapper: EnvironmentWrapper) -> None:
        self._wrapper = wrapper
        self.total = Telemetry()
        self._reset()

    def _reset(self) -> None:
        self._our_seat: int | None = None
        self._prizes_dealt = False
        self._special: dict[int, list[int]] = {}   # our serial -> specials
        self._alive: set[int] = set()
        self._online: int | None = None
        self._opp_prize: int | None = None
        self._opp_deck_low: int | None = None
        self._our_deck_low: int | None = None
        self._opp_deck_end: int | None = None
        self._game = Telemetry()

    def begin_game(self, our_seat: int) -> None:
        self._reset()
        self._our_seat = our_seat

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

        if agent_index == self._our_seat:
            self._track_our_board(players[self._our_seat])
            opponent = players[1 - self._our_seat]
            # Prizes are only DEALT after setup, and before that the list
            # is empty — which is indistinguishable from "took all six"
            # unless the tracking starts at the first non-empty reading.
            # Taking min over every observation instead made this metric
            # report ~6.0 taken in essentially every game, real and
            # internal alike: an artifact of the setup phase, not a fact
            # about the opponent. (Found 01/Ago; the number it produced
            # was quoted in the 31/Jul and 01/Ago rounds.)
            prize = len(opponent.get("prize") or [])
            if prize > 0:
                self._prizes_dealt = True
            if self._prizes_dealt:
                self._opp_prize = (prize if self._opp_prize is None
                                   else min(self._opp_prize, prize))
            opp_deck = opponent.get("deckCount")
            if isinstance(opp_deck, int):
                self._opp_deck_end = opp_deck
                self._opp_deck_low = (opp_deck if self._opp_deck_low is None
                                      else min(self._opp_deck_low, opp_deck))
            our_deck = players[self._our_seat].get("deckCount")
            if isinstance(our_deck, int):
                self._our_deck_low = (our_deck if self._our_deck_low is None
                                      else min(self._our_deck_low, our_deck))
            if self._online is None:
                for pokemon in _board(opponent):
                    if pokemon.get("id") == ALAKAZAM:
                        self._online = turn
            return

        # --- opponent decision ---
        opponent = players[agent_index]
        if self._online is None:
            for pokemon in _board(opponent):
                if pokemon.get("id") == ALAKAZAM:
                    self._online = turn
        try:
            obs = api.to_observation_class(dict(obs_dict))
        except Exception:
            return
        select = getattr(obs, "select", None)
        options = getattr(select, "option", None) if select else None
        if not options or not action:
            return
        picked = action[0]
        if not 0 <= picked < len(options):
            return
        option = options[picked]
        if option.type == OptionType.ATTACK:
            attack_id = getattr(option, "attackId", None)
            if isinstance(attack_id, int):
                self._game.attacks[attack_id] += 1
                if attack_id == POWERFUL_HAND:
                    hand = opponent.get("handCount")
                    if isinstance(hand, int):
                        self._game.powerful_hand_handsize.append(hand)
        elif option.type == OptionType.PLAY:
            card_id = self._wrapper.resolve_card_id(obs, option)
            if isinstance(card_id, int):
                self._game.plays[card_id] += 1
                self._game.play_turns.setdefault(card_id, []).append(turn)

    def _track_our_board(self, ours: dict) -> None:
        """Special energies leaving our board, split by cause."""
        present: dict[int, list[int]] = {}
        for pokemon in _board(ours):
            serial = pokemon.get("serial")
            if not isinstance(serial, int):
                continue
            specials = sorted(
                c.get("id") for c in (pokemon.get("energyCards") or [])
                if isinstance(c, dict) and c.get("id") in OUR_SPECIAL)
            present[serial] = specials
            previous = self._special.get(serial)
            if previous is None:
                self._game.attached_special += len(specials)
            elif len(specials) > len(previous):
                self._game.attached_special += len(specials) - len(previous)
            elif len(specials) < len(previous):
                # host is still here, so something REMOVED the energy
                self._game.stripped_survivor += len(previous) - len(specials)
        for serial in set(self._special) - set(present):
            self._game.lost_with_host += len(self._special[serial])
            # our bodies do not leave the board except by knockout, so a
            # vanished serial IS a KO — an independent read on the same
            # question the prize count answers, and one that does not
            # depend on a field that is empty during setup
            self._game.our_kos += 1
        self._special = present

    def end_game(self, our_result: bool | None, turns: int) -> None:
        self._game.games = 1
        self._game.turns.append(turns)
        if our_result:
            self._game.our_wins += 1
        if self._online is not None:
            self._game.alakazam_online_turn.append(self._online)
        if self._opp_prize is not None:
            self._game.opp_prizes_taken.append(6 - self._opp_prize)
        if self._opp_deck_low is not None:
            self._game.opp_deck_low.append(self._opp_deck_low)
            # 60 minus what is left, over the turns it took: the mill's
            # actual throughput, which is what the race is decided on
            if turns > 0:
                self._game.mill_per_turn.append(
                    (60 - self._opp_deck_low) / turns)
        if self._our_deck_low is not None:
            self._game.our_deck_low.append(self._our_deck_low)
        if self._opp_deck_end is not None:
            self._game.opp_deck_end.append(self._opp_deck_end)
        self.total.merge(self._game)
        self._reset()


def _board(player: dict) -> list[dict]:
    active = player.get("active") or []
    bench = player.get("bench") or []
    return [p for p in list(active) + list(bench) if isinstance(p, dict)]


# --------------------------------------------------------------------------
# driver 1: real replays
# --------------------------------------------------------------------------

def from_replays(episodes_dir: Path, archetype: str, team: str,
                 allowed: set[int] | None, index: CardIndex,
                 wrapper: EnvironmentWrapper) -> Telemetry:
    collector = Collector(wrapper)
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
        last_turn = 0
        for agent_index, obs_dict, action in _iter_decisions(replay):
            collector.observe(agent_index, obs_dict, action)
            turn = (obs_dict.get("current") or {}).get("turn")
            if isinstance(turn, int):
                last_turn = max(last_turn, turn)
        rewards = replay.get("rewards") or []
        won = None
        if len(rewards) == 2 and rewards[0] is not None:
            won = rewards[seat] > rewards[1 - seat]
        collector.end_game(won, last_turn)
    return collector.total


# --------------------------------------------------------------------------
# driver 2: internal games
# --------------------------------------------------------------------------

class _Recording:
    """Wraps an agent so its (observation, answer) pairs reach the collector."""

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
                  wrapper: EnvironmentWrapper) -> Telemetry:
    from ..environment_wrapper.ab_test import ArmMetrics, ArmSpec, arm_factory

    collector = Collector(wrapper)
    make_us = arm_factory(ArmSpec.parse(our_arm), index, effects,
                          ArmMetrics(), our_deck)
    make_them = arm_factory(ArmSpec.parse(opp_arm), index, effects,
                            ArmMetrics(), opp_deck)
    for game_index in range(games):
        our_seat = game_index % 2          # alternate seats, as always
        collector.begin_game(our_seat)
        us = _Recording(make_us(seed + game_index), collector, our_seat)
        them = _Recording(make_them(seed + 10_000 + game_index), collector,
                          1 - our_seat)
        agents = (us, them) if our_seat == 0 else (them, us)
        decks = ((our_deck, opp_deck) if our_seat == 0
                 else (opp_deck, our_deck))
        result, turns = play_one_game(agents, list(decks[0]), list(decks[1]))
        collector.end_game(result == our_seat if result in (0, 1) else None,
                           turns)
    return collector.total


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def _mean(values: list) -> float | None:
    return sum(values) / len(values) if values else None


def _median(values: list) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2


def _fmt(value: float | None, spec: str = ".2f") -> str:
    return format(value, spec) if value is not None else "-"


def _per_game(count: int, games: int) -> float | None:
    return count / games if games else None


def compare(rows: dict[str, Telemetry], index: CardIndex) -> dict:
    labels = list(rows)
    payload: dict = {}

    def line(name: str, values: list[float | None], spec: str = ".2f") -> None:
        cells = "".join(f"{_fmt(v, spec):>16s}" for v in values)
        print(f"  {name:34s}{cells}")

    print("\n" + "=" * 90)
    print("CÉLULA ALAKAZAM — mesmo coletor nos dois lados")
    print("=" * 90)
    print(f"  {'métrica':34s}" + "".join(f"{l:>16s}" for l in labels))
    print("  " + "-" * 86)

    line("jogos", [float(rows[l].games) for l in labels], ".0f")
    line("NOSSO winrate",
         [rows[l].our_wins / rows[l].games if rows[l].games else None
          for l in labels], ".1%")
    line("turnos por jogo (média)", [_mean(rows[l].turns) for l in labels])
    line("prêmios que ELES tiram/jogo",
         [_mean(rows[l].opp_prizes_taken) for l in labels])
    line("KOs que ELES nos dão/jogo",
         [_per_game(rows[l].our_kos, rows[l].games) for l in labels])
    line("turno em que Alakazam entra",
         [_mean(rows[l].alakazam_online_turn) for l in labels])

    print("  " + "-" * 86)
    line("deck DELES no mínimo (mediana)",
         [_median(rows[l].opp_deck_low) for l in labels], ".1f")
    line("deck DELES no fim (média)",
         [_mean(rows[l].opp_deck_end) for l in labels])
    line("cartas milladas por turno",
         [_mean(rows[l].mill_per_turn) for l in labels])
    line("share de jogos em que eles chegam a <=5",
         [sum(1 for d in rows[l].opp_deck_low if d <= 5) / len(rows[l].opp_deck_low)
          if rows[l].opp_deck_low else None for l in labels], ".1%")
    line("share em que eles DECKAM (chegam a 0)",
         [sum(1 for d in rows[l].opp_deck_low if d == 0) / len(rows[l].opp_deck_low)
          if rows[l].opp_deck_low else None for l in labels], ".1%")
    line("NOSSO deck no mínimo (mediana)",
         [_median(rows[l].our_deck_low) for l in labels], ".1f")

    print("  " + "-" * 86)
    for card_id, name in TRACKED.items():
        line(f"{name}/jogo",
             [_per_game(rows[l].plays.get(card_id, 0), rows[l].games)
              for l in labels])
        line(f"  ...turno médio",
             [_mean(rows[l].play_turns.get(card_id, [])) for l in labels])

    print("  " + "-" * 86)
    line("energias especiais NOSSAS anexadas/jogo",
         [_per_game(rows[l].attached_special, rows[l].games)
          for l in labels])
    line("...ARRANCADAS (host sobreviveu)/jogo",
         [_per_game(rows[l].stripped_survivor, rows[l].games)
          for l in labels])
    line("...perdidas junto com o host (KO)/jogo",
         [_per_game(rows[l].lost_with_host, rows[l].games) for l in labels])
    line("share das anexadas que foi ARRANCADA",
         [rows[l].stripped_survivor / rows[l].attached_special
          if rows[l].attached_special else None for l in labels], ".1%")

    print("  " + "-" * 86)
    line("ataques deles/jogo",
         [_per_game(sum(rows[l].attacks.values()), rows[l].games)
          for l in labels])
    line("share Powerful Hand",
         [rows[l].attacks.get(POWERFUL_HAND, 0) / sum(rows[l].attacks.values())
          if sum(rows[l].attacks.values()) else None for l in labels], ".1%")
    line("mão deles ao usar Powerful Hand",
         [_mean(rows[l].powerful_hand_handsize) for l in labels])
    line("=> POWERFUL HANDS ACERTADOS/jogo",
         [_per_game(rows[l].attacks.get(POWERFUL_HAND, 0), rows[l].games)
          for l in labels])
    line("=> dano médio do Powerful Hand",
         [(_mean(rows[l].powerful_hand_handsize) or 0) * 20
          for l in labels], ".0f")

    for label in labels:
        telemetry = rows[label]
        payload[label] = {
            "games": telemetry.games,
            "our_winrate": (telemetry.our_wins / telemetry.games
                            if telemetry.games else None),
            "mean_turns": _mean(telemetry.turns),
            "opp_prizes_taken": _mean(telemetry.opp_prizes_taken),
            "our_kos_per_game": _per_game(telemetry.our_kos, telemetry.games),
            "alakazam_online_turn": _mean(telemetry.alakazam_online_turn),
            "plays_per_game": {
                TRACKED[c]: _per_game(telemetry.plays.get(c, 0),
                                      telemetry.games)
                for c in TRACKED},
            "special_attached_per_game": _per_game(
                telemetry.attached_special, telemetry.games),
            "special_stripped_per_game": _per_game(
                telemetry.stripped_survivor, telemetry.games),
            "special_lost_with_host_per_game": _per_game(
                telemetry.lost_with_host, telemetry.games),
            "stripped_share": (telemetry.stripped_survivor
                               / telemetry.attached_special
                               if telemetry.attached_special else None),
            "attacks_per_game": _per_game(sum(telemetry.attacks.values()),
                                          telemetry.games),
            "powerful_hand_share": (
                telemetry.attacks.get(POWERFUL_HAND, 0)
                / sum(telemetry.attacks.values())
                if sum(telemetry.attacks.values()) else None),
            "powerful_hand_handsize": _mean(telemetry.powerful_hand_handsize),
            "opp_deck_low_median": _median(telemetry.opp_deck_low),
            "opp_deck_end_mean": _mean(telemetry.opp_deck_end),
            "mill_per_turn": _mean(telemetry.mill_per_turn),
            "opp_deckout_share": (
                sum(1 for d in telemetry.opp_deck_low if d == 0)
                / len(telemetry.opp_deck_low)
                if telemetry.opp_deck_low else None),
            "our_deck_low_median": _median(telemetry.our_deck_low),
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
                        default=["heuristic",
                                 "network,models/bc_majkel.npz,"
                                 "models/feature_stats.npz"])
    parser.add_argument("--games", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes-dir", type=Path, default=EPISODES_DIR)
    parser.add_argument("--team", type=str, default=OUR_TEAM)
    parser.add_argument("--submission-index", type=Path, default=SUB_INDEX)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    index, effects = CardIndex(), EffectIndex()
    wrapper = EnvironmentWrapper(index)
    allowed = None
    if args.submission_index.exists():
        with open(args.submission_index, encoding="utf-8") as fh:
            allowed = {int(i) for i in
                       (json.load(fh).get("episode_ids") or [])}

    rows: dict[str, Telemetry] = {}
    rows["REAL"] = from_replays(args.episodes_dir, args.archetype, args.team,
                                allowed, index, wrapper)
    our_deck = read_deck_ids(args.our_deck)
    opp_deck = read_deck_ids(args.opp_deck)
    for arm in args.opp_arms:
        label = arm.split(",")[0]
        if "bc_" in arm:
            label = "BC-" + arm.split("bc_")[1].split(".")[0]
        rows[label] = from_selfplay(our_deck, opp_deck, args.our_arm, arm,
                                    args.games, args.seed, index, effects,
                                    wrapper)

    payload = compare(rows, index)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        print(f"\njson: {args.json}")


if __name__ == "__main__":
    main()
