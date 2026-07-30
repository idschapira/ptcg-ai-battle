"""Quantify DEAD WEIGHT in the shipped Crustle list, from real games.

The thesis under test: the 4 Rock Fighting Energy are close to dead
weight, so their slots could buy TURNS instead (more bodies = more
prizes the opponent must take = more turns for the mill to land).

What the engine actually says (`cg.api`, not our copy of it):
  * Rock Fighting Energy (20) "Prevent all effects of attacks used by
    your opponent's Pokémon done to the **{F} Pokémon** this card is
    attached to. (Damage is not an effect.)" -> provides {F}; the
    protection clause is LIVE only on a {F} host, and never reduces
    damage.
  * Mist Energy (11) has the SAME clause with no type restriction ->
    provides {C}, protects ANY host. It is the strict superset.
  * Our {F} Pokémon are Great Tusk (58) and Terrakion (607). The wall,
    Crustle (345), and Dwebble (344) are {G} -> Rock Fighting's clause
    is structurally dead on them.

So the protection clause can only ever pay out when (i) our active is
{F}, (ii) it holds a Rock Fighting, and (iii) the incoming attack HAS an
effect. This module measures each of those three factors separately on
our real ladder episodes, plus the mechanism baselines the deck change
is supposed to move (turn of death, how close the opponent got to
decking out).

Two sample traps this module refuses to fall into:

  * viewer/episodes/ holds episodes from EVERY submission we ever ran —
    Grimmsnarl and Abomasnow lists included. Auditing the Crustle deck
    over that directory measures the wrong decks (Marnie's Grimmsnarl
    shows up as "our active"). ``--our-archetype`` filters to episodes
    where WE played the list under audit; it defaults to Crustle.
  * Within one step, BOTH agents' observations carry a `current`, and the
    two views differ (hidden info masked, one lagging the other by an
    action). Sweeping all of them double-counts and invents HP swings.
    We sweep ONLY our own seat's views, which are authoritative about our
    own board.

Sample caveat printed with every report: the fetch defaults to
loss-first, so a win rate here is NOT a ladder estimate unless the whole
submission was pulled.

Run from the repo root:
    python -m src.analysis.deadweight_audit
    python -m src.analysis.deadweight_audit --archetype "Alakazam box (non-ex)"
    python -m src.analysis.deadweight_audit --json out.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Iterator

import cg.api as api
from cg.api import OptionType

from ..deckbuilding.archetype_rules import label_archetype
from ..environment_wrapper.wrapper import EnvironmentWrapper
from ..ingestion.build_card_model import REPO_ROOT
from ..ingestion.build_effect_model import (EffectIndex, EffectTarget,
                                            EffectType)
from ..ingestion.card_index import CardIndex
from ..ingestion.replays_parse import _iter_decisions
from .fetch_my_episodes import EPISODES_DIR
from .meta_radar import observed_serials as _observed_serials

OUR_TEAM: Final[str] = "Ilan Schapira"

ROCK_FIGHTING: Final[int] = 20
MIST_ENERGY: Final[int] = 11
BASIC_FIGHTING: Final[int] = 6
FIGHTING_TYPE: Final[int] = 6          # engine energy namespace
LOW_DECK_THRESHOLD: Final[int] = 5     # "1-2 turns from deck-out"


def _fighting_card_ids(index: CardIndex) -> frozenset[int]:
    """Card ids whose energyType is {F} (Rock Fighting's clause is live)."""
    out = set()
    for card in api.all_card_data():
        if getattr(card, "energyType", None) == FIGHTING_TYPE and card.hp:
            out.add(card.cardId)
    return frozenset(out)


@dataclass
class AttackEvent:
    """One attack the OPPONENT used against us."""

    turn: int | None
    attack_id: int
    has_effect: bool          # loose: any rules text
    preventable: bool         # strict: effect done to OUR active
    our_active_id: int | None
    our_active_is_fighting: bool
    our_active_had_rock: bool
    our_active_had_mist: bool


@dataclass
class EpisodeAudit:
    episode_id: int
    our_seat: int
    opponent: str
    archetype: str
    our_archetype: str
    result: str
    final_turn: int | None = None
    our_deck_left: int | None = None
    opp_deck_left: int | None = None
    opp_min_deck: int | None = None        # closest the opponent got to 0
    our_min_deck: int | None = None

    # active-slot occupancy over OUR decision points
    active_states: Counter[int] = field(default_factory=Counter)
    active_states_damaged: Counter[int] = field(default_factory=Counter)
    active_states_threatened: Counter[int] = field(default_factory=Counter)

    # attachments: (energy card serial, host serial) -> host card id
    rock_targets: Counter[int] = field(default_factory=Counter)
    mist_targets: Counter[int] = field(default_factory=Counter)

    attacks_on_us: list[AttackEvent] = field(default_factory=list)
    damage_absorbed: Counter[int] = field(default_factory=Counter)
    kos_suffered: Counter[int] = field(default_factory=Counter)


def _our_seat(replay: dict, team: str) -> int | None:
    names = ((replay.get("info") or {}).get("TeamNames")) or []
    for i, name in enumerate(names):
        if name == team:
            return i
    return None


def _result(replay: dict, our_seat: int) -> str:
    rewards = replay.get("rewards")
    if not isinstance(rewards, list) or len(rewards) != 2:
        return "unknown"
    if rewards[0] is None or rewards[1] is None:
        return "unknown"
    if rewards[0] == rewards[1]:
        return "draw"
    return "win" if rewards[our_seat] > rewards[1 - our_seat] else "loss"


def _iter_states(replay: dict, seat: int | None = None) -> Iterator[dict]:
    """Engine `current` dicts in order; one per step when `seat` is given.

    Both agents publish a `current` in the same step and the two views
    disagree (each masks the other's hidden zones, and one lags the other
    by an action). Any ledger that accumulates deltas MUST pin a single
    perspective or it double-counts; pass `seat` for that. `seat=None`
    keeps every view, which is what card-revelation mining wants.
    """
    for step in replay.get("steps") or []:
        if not isinstance(step, list):
            continue
        entries = (list(step) if seat is None
                   else ([step[seat]] if seat < len(step) else []))
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            current = (entry.get("observation") or {}).get("current")
            if isinstance(current, dict) and current.get("players"):
                yield current


def _in_play(player: dict) -> list[dict]:
    active = player.get("active") or []
    bench = player.get("bench") or []
    return [p for p in list(active) + list(bench) if isinstance(p, dict)]


def _energy_card_ids(pokemon: dict) -> list[int]:
    out = []
    for card in pokemon.get("energyCards") or []:
        if isinstance(card, dict) and isinstance(card.get("id"), int):
            out.append(card["id"])
    return out


def _damaging_attacks(index: CardIndex) -> dict[int, int]:
    """card_id -> minimum energy count among its damaging attacks."""
    attacks = api.all_attack()
    out: dict[int, int] = {}
    for card in api.all_card_data():
        best = None
        for aid in (card.attacks or []):
            if not isinstance(aid, int) or not 1 <= aid <= len(attacks):
                continue
            attack = attacks[aid - 1]
            if (attack.damage or 0) <= 0:
                continue
            cost = len(attack.energies or [])
            best = cost if best is None else min(best, cost)
        if best is not None:
            out[card.cardId] = best
    return out


class _EffectTable:
    """Which attacks a prevention energy on OUR ACTIVE could actually stop.

    Two different questions, so two different flags:

    ``has_text`` — the attack has any rules text at all. This is the LOOSE
    upper bound and it badly overstates the clause's reach: most text is a
    damage modifier ("does 30 more damage for each…"), a self-effect
    ("also does 70 damage to itself") or an effect on the attacker's own
    board ("attach 3 Energy to your Benched Pokémon"). None of that is an
    effect done to the Pokémon holding the energy.

    ``preventable`` — the attack carries an effect row whose type is an
    effect DONE TO one of our Pokémon and whose target includes our
    active. Damage-flavoured rows are deliberately excluded: the cards say
    "Damage is not an effect", and BENCH_DAMAGE/SNIPE are damage, so an
    energy on the active neither blocks them nor would help if it did.
    COUNTERS is the important one and it is NOT damage — verified against
    the engine in tests/test_effect_prevention_contract.py.
    """

    PREVENTABLE_TYPES: Final[frozenset[int]] = frozenset({
        int(EffectType.STATUS),
        int(EffectType.COUNTERS),
        int(EffectType.ENERGY_DISCARD_OPP),
        int(EffectType.GUST),
        int(EffectType.DISRUPT_RESOURCE),
    })
    ACTIVE_TARGETS: Final[frozenset[int]] = frozenset({
        int(EffectTarget.OPP_ACTIVE),
        int(EffectTarget.OPP_ANY),
    })

    def __init__(self, effects: EffectIndex) -> None:
        self._has_text = {
            a.attackId: bool((a.text or "").strip()) for a in api.all_attack()
        }
        self._effects = effects

    def has_text(self, attack_id: int | None) -> bool:
        if attack_id is None:
            return False
        return self._has_text.get(attack_id, False)

    def preventable(self, attack_id: int | None) -> bool:
        if attack_id is None:
            return False
        for row in self._effects.effects_of(attack_id):
            if (row.effect_type in self.PREVENTABLE_TYPES
                    and row.target in self.ACTIVE_TARGETS):
                return True
        return False


def audit_episode(path: Path, index: CardIndex, wrapper: EnvironmentWrapper,
                  fighting_ids: frozenset[int], effects: _EffectTable,
                  min_cost: dict[int, int],
                  team: str) -> EpisodeAudit | None:
    try:
        with open(path, encoding="utf-8") as fh:
            replay = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    seat = _our_seat(replay, team)
    if seat is None:
        return None
    names = ((replay.get("info") or {}).get("TeamNames")) or ["?", "?"]
    episode_id = (replay.get("info") or {}).get("EpisodeId")
    if not isinstance(episode_id, int):
        episode_id = int(path.stem) if path.stem.isdigit() else 0

    # --- archetype labels from every card either side ever revealed ---
    # Same rules the runtime estimator uses (archetype_rules is the single
    # source of truth); ours matters because viewer/episodes/ mixes decks.
    revealed: dict[int, Counter[str]] = {0: Counter(), 1: Counter()}
    for (player, _serial), card_id in _observed_serials(replay).items():
        card = index.get_card(card_id)
        if card is not None and player in revealed:
            revealed[player][card.card_name] += 1

    audit = EpisodeAudit(
        episode_id=episode_id,
        our_seat=seat,
        opponent=str(names[1 - seat]) if 1 - seat < len(names) else "?",
        archetype=label_archetype(revealed[1 - seat]),
        our_archetype=label_archetype(revealed[seat]),
        result=_result(replay, seat),
    )

    # --- state sweep: attachments, deck lows, damage/KO ledger ---
    # ONE view per step (ours) so deltas are not double-counted.
    seen_attach: set[tuple[int, int]] = set()
    hp_by_serial: dict[int, int] = {}
    id_by_serial: dict[int, int] = {}
    alive: set[int] = set()
    last_turn = None
    for state in _iter_states(replay, seat=seat):
        players = state.get("players") or []
        if len(players) < 2:
            continue
        ours, theirs = players[seat], players[1 - seat]
        turn = state.get("turn")
        if isinstance(turn, int):
            last_turn = turn if last_turn is None else max(last_turn, turn)

        for player, attr in ((ours, "our_min_deck"), (theirs, "opp_min_deck")):
            count = player.get("deckCount")
            if isinstance(count, int):
                current = getattr(audit, attr)
                setattr(audit, attr,
                        count if current is None else min(current, count))
        audit.our_deck_left = ours.get("deckCount")
        audit.opp_deck_left = theirs.get("deckCount")

        present: set[int] = set()
        evolved_under: set[int] = set()
        for pokemon in _in_play(ours):
            serial = pokemon.get("serial")
            card_id = pokemon.get("id")
            if not isinstance(serial, int) or not isinstance(card_id, int):
                continue
            present.add(serial)
            id_by_serial[serial] = card_id
            for under in pokemon.get("preEvolution") or []:
                if isinstance(under, dict) and isinstance(under.get("serial"),
                                                          int):
                    evolved_under.add(under["serial"])
            hp = pokemon.get("hp")
            if isinstance(hp, int):
                previous = hp_by_serial.get(serial)
                if previous is not None and hp < previous:
                    audit.damage_absorbed[card_id] += previous - hp
                hp_by_serial[serial] = hp
            for energy_serial, energy_id in _attached(pokemon):
                key = (energy_serial, serial)
                if key in seen_attach:
                    continue
                seen_attach.add(key)
                if energy_id == ROCK_FIGHTING:
                    audit.rock_targets[card_id] += 1
                elif energy_id == MIST_ENERGY:
                    audit.mist_targets[card_id] += 1
        # A serial that left the board was knocked out UNLESS it is now
        # sitting under an evolution (evolving mints a new serial and
        # files the old card in the new one's preEvolution). Only the top
        # card is charged with the KO, so a KO'd Crustle counts once even
        # though its Dwebble left the board turns earlier.
        for serial in alive - present:
            if serial in evolved_under:
                continue
            card_id = id_by_serial.get(serial)
            if card_id is None:
                continue
            audit.kos_suffered[card_id] += 1
            # the lethal hit lands inside the opponent's turn, which our
            # own view never shows; charge the last known HP so the
            # ledger stays a LOWER BOUND instead of dropping the hit.
            audit.damage_absorbed[card_id] += hp_by_serial.get(serial, 0)
        alive = present

    audit.final_turn = last_turn

    # --- decision sweep: active occupancy + attacks aimed at us ---
    for agent_index, obs_dict, action in _iter_decisions(replay):
        state = obs_dict.get("current") or {}
        players = state.get("players") or []
        if len(players) < 2:
            continue
        ours = players[seat] if seat < len(players) else {}
        theirs = players[1 - seat] if 1 - seat < len(players) else {}
        our_active = next(iter(ours.get("active") or []), None)
        our_active = our_active if isinstance(our_active, dict) else None
        turn = state.get("turn")

        if agent_index == seat and our_active is not None:
            card_id = our_active.get("id")
            if isinstance(card_id, int):
                audit.active_states[card_id] += 1
                hp, max_hp = our_active.get("hp"), our_active.get("maxHp")
                if (isinstance(hp, int) and isinstance(max_hp, int)
                        and hp < max_hp):
                    audit.active_states_damaged[card_id] += 1
                if _threatened(theirs, min_cost):
                    audit.active_states_threatened[card_id] += 1

        if agent_index != seat:
            attack_id = _chosen_attack(obs_dict, action)
            if attack_id is None:
                continue
            rock = mist = False
            active_id = None
            fighting = False
            if our_active is not None:
                active_id = our_active.get("id")
                fighting = active_id in fighting_ids
                attached = _energy_card_ids(our_active)
                rock = ROCK_FIGHTING in attached
                mist = MIST_ENERGY in attached
            audit.attacks_on_us.append(AttackEvent(
                turn=turn if isinstance(turn, int) else None,
                attack_id=attack_id,
                has_effect=effects.has_text(attack_id),
                preventable=effects.preventable(attack_id),
                our_active_id=active_id,
                our_active_is_fighting=fighting,
                our_active_had_rock=rock,
                our_active_had_mist=mist,
            ))
    return audit


def _attached(pokemon: dict) -> list[tuple[int, int]]:
    """(serial, card_id) for every energy card attached to `pokemon`."""
    out = []
    for card in pokemon.get("energyCards") or []:
        if (isinstance(card, dict) and isinstance(card.get("id"), int)
                and isinstance(card.get("serial"), int)):
            out.append((card["serial"], card["id"]))
    return out


def _threatened(opponent: dict, min_cost: dict[int, int]) -> bool:
    """Can the opponent's active pay for a damaging attack right now?"""
    active = next(iter(opponent.get("active") or []), None)
    if not isinstance(active, dict):
        return False
    card_id = active.get("id")
    if not isinstance(card_id, int) or card_id not in min_cost:
        return False
    return len(active.get("energies") or []) >= min_cost[card_id]


def _chosen_attack(obs_dict: dict, action: list[int]) -> int | None:
    """attackId of the ATTACK the recorded action picked, if it is one."""
    try:
        obs = api.to_observation_class(dict(obs_dict))
    except Exception:
        return None
    select = getattr(obs, "select", None)
    options = getattr(select, "option", None) if select is not None else None
    if not options or not action:
        return None
    index = action[0]
    if not 0 <= index < len(options):
        return None
    option = options[index]
    if option.type != OptionType.ATTACK:
        return None
    attack_id = getattr(option, "attackId", None)
    return attack_id if isinstance(attack_id, int) else None


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def _pct(num: int, den: int) -> str:
    return f"{num/den:6.1%}" if den else "     -"


def report(audits: list[EpisodeAudit], index: CardIndex,
           fighting_ids: frozenset[int]) -> dict[str, Any]:
    def name(card_id: int | None) -> str:
        card = index.get_card(card_id) if card_id is not None else None
        return card.card_name if card is not None else f"id={card_id}"

    print(f"\n{'='*74}\nAMOSTRA: {len(audits)} episódios reais "
          f"(viewer/episodes/, fetch é LOSS-FIRST -> winrate aqui não "
          f"estima ladder)\n{'='*74}")
    results = Counter(a.result for a in audits)
    print(f"resultados: {dict(results)}")

    by_arch: dict[str, list[EpisodeAudit]] = defaultdict(list)
    for audit in audits:
        by_arch[audit.archetype].append(audit)
    print("\n-- oponentes por arquétipo --")
    for arch, group in sorted(by_arch.items(), key=lambda kv: -len(kv[1])):
        wins = sum(1 for a in group if a.result == "win")
        print(f"  {arch:34s} n={len(group):4d} ({len(group)/len(audits):5.1%})"
              f"  W{wins}/L{len(group)-wins}")

    # ---- factor 1: is a {F} pokémon ever the active under threat? ----
    print("\n-- FATOR 1: ocupação do slot ATIVO nas NOSSAS decisões --")
    total = Counter()
    damaged = Counter()
    threatened = Counter()
    for audit in audits:
        total += audit.active_states
        damaged += audit.active_states_damaged
        threatened += audit.active_states_threatened
    n_total = sum(total.values())
    print(f"  {'ativo':22s} {'decisões':>10s} {'share':>7s} "
          f"{'c/ dano':>9s} {'sob ameaça':>11s}")
    for card_id, count in total.most_common(10):
        tag = " {F}" if card_id in fighting_ids else ""
        print(f"  {name(card_id)[:20]+tag:22s} {count:10d} "
              f"{_pct(count, n_total)} {damaged[card_id]:9d} "
              f"{threatened[card_id]:11d}")
    fight_total = sum(c for cid, c in total.items() if cid in fighting_ids)
    fight_threat = sum(c for cid, c in threatened.items()
                       if cid in fighting_ids)
    all_threat = sum(threatened.values())
    print(f"\n  ativo é {{F}} (Rock Fighting VIVA): {fight_total}/{n_total} "
          f"= {_pct(fight_total, n_total)} das decisões")
    print(f"  ativo é {{F}} E sob ameaça:          {fight_threat}/{all_threat}"
          f" = {_pct(fight_threat, all_threat)} das decisões sob ameaça")

    # ---- factor 2: where did the Rock Fightings actually land? ----
    print("\n-- FATOR 2: alvos REAIS das anexações de energia --")
    rock = Counter()
    mist = Counter()
    for audit in audits:
        rock += audit.rock_targets
        mist += audit.mist_targets
    for label, counter in (("Rock Fighting", rock), ("Mist", mist)):
        n = sum(counter.values())
        live = sum(c for cid, c in counter.items() if cid in fighting_ids)
        print(f"  {label} — {n} anexações em {len(audits)} jogos "
              f"({n/len(audits):.2f}/jogo)")
        for card_id, count in counter.most_common(6):
            tag = "{F} cláusula VIVA" if card_id in fighting_ids else \
                  "não-{F} cláusula MORTA"
            print(f"      {name(card_id)[:20]:22s} {count:5d} "
                  f"{_pct(count, n)}  {tag}")
        if label == "Rock Fighting":
            print(f"      => cláusula viva em {live}/{n} = "
                  f"{_pct(live, n)} das anexações")

    # ---- factor 3: did the incoming attacks even HAVE effects? ----
    print("\n-- FATOR 3: ataques do oponente contra nós --")
    events = [e for a in audits for e in a.attacks_on_us]
    with_text = [e for e in events if e.has_effect]
    prevent = [e for e in events if e.preventable]
    print(f"  {len(events)} ataques observados")
    print(f"    com QUALQUER texto de regra:            {len(with_text):5d} "
          f"{_pct(len(with_text), len(events))}  (limite superior FROUXO)")
    print(f"    com efeito DIRIGIDO ao nosso ativo:     {len(prevent):5d} "
          f"{_pct(len(prevent), len(events))}  <- o que a cláusula pode parar")
    fight = [e for e in prevent if e.our_active_is_fighting]
    rock = [e for e in prevent
            if e.our_active_is_fighting and e.our_active_had_rock]
    mist = [e for e in prevent if e.our_active_had_mist]
    covered = [e for e in prevent
               if e.our_active_had_mist
               or (e.our_active_is_fighting and e.our_active_had_rock)]
    print(f"  dos preveníveis, ativo era {{F}}:           {len(fight):5d} "
          f"{_pct(len(fight), len(prevent))}")
    print(f"  JÁ prevenido por Rock em ativo {{F}}:       {len(rock):5d} "
          f"{_pct(len(rock), len(prevent))}")
    print(f"  JÁ prevenido por Mist (qualquer ativo):    {len(mist):5d} "
          f"{_pct(len(mist), len(prevent))}")
    print(f"  JÁ prevenido por alguma das duas:          {len(covered):5d} "
          f"{_pct(len(covered), len(prevent))}")
    print(f"  NÃO prevenido (dano gratuito recebido):    "
          f"{len(prevent)-len(covered):5d} "
          f"{_pct(len(prevent)-len(covered), len(prevent))}")
    top = Counter(e.attack_id for e in prevent)
    attacks = api.all_attack()
    print("  ataques preveníveis mais frequentes:")
    for attack_id, count in top.most_common(5):
        attack = attacks[attack_id - 1] if 1 <= attack_id <= len(attacks) \
            else None
        label = attack.name if attack is not None else f"aid={attack_id}"
        print(f"      {label[:28]:28s} {count:5d} "
              f"{_pct(count, len(prevent))}")

    # ---- damage / KO ledger ----
    print("\n-- quem de fato absorve dano e morre --")
    dmg = Counter()
    kos = Counter()
    for audit in audits:
        dmg += audit.damage_absorbed
        kos += audit.kos_suffered
    n_dmg = sum(dmg.values())
    for card_id, amount in dmg.most_common(8):
        tag = " {F}" if card_id in fighting_ids else ""
        print(f"  {name(card_id)[:20]+tag:22s} dano={amount:7d} "
              f"{_pct(amount, n_dmg)}  KOs={kos[card_id]}")

    # ---- mechanism baselines ----
    print("\n-- MECANISMO (linha de base a bater) --")
    _mechanism_block(audits)
    print("\n-- MECANISMO por arquétipo --")
    for arch, group in sorted(by_arch.items(), key=lambda kv: -len(kv[1])):
        if len(group) < 5:
            continue
        print(f"  [{arch}] n={len(group)}")
        _mechanism_block(group, indent="    ")

    return {
        "n_episodes": len(audits),
        "results": dict(results),
        "active_states": {name(k): v for k, v in total.items()},
        "active_states_threatened": {name(k): v for k, v in threatened.items()},
        "fighting_active_share": (fight_total / n_total) if n_total else None,
        "fighting_threatened_share": ((fight_threat / all_threat)
                                      if all_threat else None),
        "rock_targets": {name(k): v for k, v in rock.items()},
        "rock_clause_live_share": (
            sum(c for cid, c in rock.items() if cid in fighting_ids)
            / sum(rock.values())) if sum(rock.values()) else None,
        "mist_targets": {name(k): v for k, v in mist.items()},
        "attacks_on_us": len(events),
        "attacks_with_any_text": len(with_text),
        "attacks_preventable": len(prevent),
        "preventable_vs_fighting_active": len(fight),
        "preventable_already_stopped_by_rock": len(rock),
        "preventable_already_stopped_by_mist": len(mist),
        "preventable_uncovered": len(prevent) - len(covered),
        "damage_absorbed": {name(k): v for k, v in dmg.items()},
        "kos_suffered": {name(k): v for k, v in kos.items()},
        "by_archetype": {
            arch: _mechanism_dict(group) for arch, group in by_arch.items()
        },
        "overall_mechanism": _mechanism_dict(audits),
    }


def _mechanism_dict(audits: list[EpisodeAudit]) -> dict[str, Any]:
    losses = [a for a in audits if a.result == "loss"]
    turns = [a.final_turn for a in losses if a.final_turn is not None]
    near = [a for a in audits
            if a.opp_min_deck is not None
            and a.opp_min_deck <= LOW_DECK_THRESHOLD]
    near_loss = [a for a in losses
                 if a.opp_min_deck is not None
                 and a.opp_min_deck <= LOW_DECK_THRESHOLD]
    opp_left = [a.opp_min_deck for a in losses if a.opp_min_deck is not None]
    return {
        "n": len(audits),
        "wins": sum(1 for a in audits if a.result == "win"),
        "losses": len(losses),
        "mean_turn_of_death": (sum(turns) / len(turns)) if turns else None,
        "median_turn_of_death": _median(turns),
        "opp_near_deckout_share": (len(near) / len(audits)) if audits else None,
        "opp_near_deckout_share_in_losses": ((len(near_loss) / len(losses))
                                             if losses else None),
        "median_opp_deck_low_in_losses": _median(opp_left),
    }


def _mechanism_block(audits: list[EpisodeAudit], indent: str = "  ") -> None:
    stats = _mechanism_dict(audits)
    turn = stats["median_turn_of_death"]
    mean = stats["mean_turn_of_death"]
    print(f"{indent}turno da morte (derrotas): mediana="
          f"{turn if turn is not None else '-'} "
          f"média={mean:.1f}" if mean is not None else
          f"{indent}turno da morte (derrotas): sem dados")
    share = stats["opp_near_deckout_share_in_losses"]
    print(f"{indent}derrotas em que o oponente chegou a <="
          f"{LOW_DECK_THRESHOLD} cartas: "
          f"{share:.1%}" if share is not None else
          f"{indent}derrotas: sem dados de deck")
    low = stats["median_opp_deck_low_in_losses"]
    print(f"{indent}mínimo do deck do oponente nas derrotas (mediana): {low}")


def _median(values: list[int]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-dir", type=Path, default=EPISODES_DIR)
    parser.add_argument("--team", type=str, default=OUR_TEAM)
    parser.add_argument("--archetype", type=str, default=None,
                        help="restrict the report to one opponent archetype")
    parser.add_argument("--our-archetype", type=str,
                        default="Crustle mill (ours)",
                        help="only audit episodes where WE played this list "
                             "(viewer/episodes/ mixes every submission we "
                             "ever ran); pass 'any' to disable")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    index = CardIndex()
    wrapper = EnvironmentWrapper(index)
    fighting_ids = _fighting_card_ids(index)
    effects = _EffectTable(EffectIndex())
    min_cost = _damaging_attacks(index)

    paths = sorted(args.episodes_dir.glob("*.json"))
    if not paths:
        raise SystemExit(f"nenhum episódio em {args.episodes_dir}")

    audits: list[EpisodeAudit] = []
    for path in paths:
        audit = audit_episode(path, index, wrapper, fighting_ids, effects,
                              min_cost, args.team)
        if audit is not None:
            audits.append(audit)
    if args.our_archetype and args.our_archetype != "any":
        before = len(audits)
        ours_only = [a for a in audits
                     if a.our_archetype == args.our_archetype]
        dropped = Counter(a.our_archetype for a in audits
                          if a.our_archetype != args.our_archetype)
        print(f"filtro NOSSO deck = {args.our_archetype!r}: "
              f"{len(ours_only)}/{before} episódios; descartados "
              f"{dict(dropped)}")
        audits = ours_only
    if args.archetype:
        audits = [a for a in audits if a.archetype == args.archetype]
        print(f"(filtrado para arquétipo do OPONENTE {args.archetype!r})")
    if not audits:
        raise SystemExit("nenhum episódio utilizável após o filtro")

    payload = report(audits, index, fighting_ids)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        print(f"\njson: {args.json}")


if __name__ == "__main__":
    main()
