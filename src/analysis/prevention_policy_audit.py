"""POLICY or ECONOMY: why 69% of preventable effects landed unprevented.

The 30/Jul audit found that Rock Fighting's clause is live most of the
game (Great Tusk is our active 52.4% of decisions) and that Alakazam's
`Powerful Hand` — 89.2% of the opponent's attacks in that cell — places
damage COUNTERS, which are an effect, not damage, so Mist/Rock null it
outright. It also found that 69% of the preventable effects we received
arrived with no prevention on the victim.

That 69% has two very different explanations and they imply opposite
work:

  ECONOMY   we never held a protective energy that could legally cover
            the victim. Then the deck is the lever — and the deck lever
            was already measured and failed (variants V1-V4, N=600/cell),
            so the hole is STRUCTURAL and the file is closed.
  POLICY    we held one and put it somewhere else, or did not attach at
            all. Then a pilot rule can recover it for zero deck slots.

This module decides between them by replaying our real games and asking,
for every unprevented preventable hit, whether a legal protective attach
onto THAT victim was available in the window since the opponent's last
attack.

Eligibility follows the engine-verified matrix
(tests/test_effect_prevention_contract.py):
  Mist Energy (11)          -> covers ANY host
  Rock Fighting Energy (20) -> covers a {F} host only

Sample discipline (the traps this project has already paid for):
  * SUBMISSION FILTER — viewer/episodes/ mixes every submission we ever
    ran. Episodes are intersected with the cached id list for the
    submission under audit, AND re-checked against a deck sentinel, so a
    stale download cannot leak a Grimmsnarl game into a Crustle audit.
  * FIXED PERSPECTIVE — only our own seat's observations are swept.
    Summing both agents' views double-counts and invents phantom KOs.
  * The victim is whoever was ACTIVE WHEN THE HIT LANDED, not whoever we
    ended our turn with: the opponent can gust a different Pokémon up,
    and protecting the wrong one would not have helped.

Where the energy actually went is read from the OBSERVED next state, not
inferred from option order — with several Pokémon in play the engine
emits one ATTACH option per (card, target) with no target field, so the
ordering is an assumption and the outcome is a measurement.

Run from the repo root:
    python -m src.analysis.prevention_policy_audit
    python -m src.analysis.prevention_policy_audit --json out.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import cg.api as api
from cg.api import AreaType, OptionType

from ..deckbuilding.archetype_rules import label_archetype
from ..ingestion.build_card_model import REPO_ROOT
from ..ingestion.build_effect_model import EffectIndex
from ..ingestion.card_index import CardIndex
from ..ingestion.replays_parse import _iter_decisions
from .deadweight_audit import (MIST_ENERGY, ROCK_FIGHTING, _EffectTable,
                               _chosen_attack, _fighting_card_ids,
                               _in_play)
from .fetch_my_episodes import EPISODES_DIR
from .meta_radar import observed_serials as _observed_serials

OUR_TEAM: Final[str] = "Ilan Schapira"
SUB_INDEX: Final[Path] = (REPO_ROOT / "data" / "processed" /
                          "episodes_index" / "sub_54917180.json")
OUR_DECK_LABEL: Final[str] = "Crustle mill (ours)"

# classification of one unprevented preventable hit
AVOIDABLE_NO_ATTACH: Final[str] = "EVITÁVEL: tinha na mão, não anexou nada"
AVOIDABLE_ELSEWHERE: Final[str] = "EVITÁVEL: tinha na mão, anexou em outro"
AVOIDABLE_WRONG_CARD: Final[str] = "EVITÁVEL: anexou energia SEM proteção"
INEVITABLE_NOT_HELD: Final[str] = "INEVITÁVEL: nenhuma protetora elegível"
UNKNOWN_WINDOW: Final[str] = "sem janela de decisão nossa"

AVOIDABLE = frozenset({AVOIDABLE_NO_ATTACH, AVOIDABLE_ELSEWHERE,
                       AVOIDABLE_WRONG_CARD})


@dataclass
class Hit:
    """One unprevented preventable attack, with its verdict."""

    episode_id: int
    turn: int | None
    attack_id: int
    victim_id: int | None
    victim_is_fighting: bool
    verdict: str
    held_mist: bool = False
    held_rock: bool = False
    attach_made: bool = False
    attach_target_id: int | None = None
    attach_card_id: int | None = None
    protective_on_bench: bool = False
    # repositioning: the victim could not be covered from hand, but a
    # DIFFERENT body of ours was already carrying cover and a switch was
    # legal. That is still policy, not economy — the resource existed, it
    # was in the wrong slot.
    cover_parked_and_switchable: bool = False
    decisions_in_window: int = 0


@dataclass
class EpisodeAudit:
    episode_id: int
    opponent_archetype: str
    result: str
    hits: list[Hit] = field(default_factory=list)
    our_decisions: int = 0
    # volume: decisions where a protective attach was legal AND the
    # active was not yet covered (the population a new rule would touch)
    coverable_decisions: int = 0
    covered_already: int = 0
    # the engine re-offers MAIN after every non-terminal action, so one
    # turn produces many decisions. A rule fires at most ONCE per turn
    # (one energy attachment per turn), so distinct turns is the honest
    # denominator for "how often would this rule actually act".
    coverable_turns: set[int] = field(default_factory=set)
    our_turns: set[int] = field(default_factory=set)
    # THE candidate rule: Mist covers ANY host, Rock covers only {F}.
    # Great Tusk is {F}, so a Rock does the same job there and keeps the
    # scarce universal card for the {G} wall, which NOTHING else in the
    # pool can protect. Count how often we spent a Mist on a {F} host
    # while a Rock was equally attachable — that is the rule's exact
    # population, and it must be big enough to be worth an A/B.
    mist_wasted_on_fighting: int = 0
    mist_on_fighting_total: int = 0


def _our_seat(replay: dict, team: str) -> int | None:
    names = ((replay.get("info") or {}).get("TeamNames")) or []
    for i, name in enumerate(names):
        if name == team:
            return i
    return None


def _result(replay: dict, seat: int) -> str:
    rewards = replay.get("rewards")
    if (not isinstance(rewards, list) or len(rewards) != 2
            or rewards[0] is None or rewards[1] is None):
        return "unknown"
    if rewards[0] == rewards[1]:
        return "draw"
    return "win" if rewards[seat] > rewards[1 - seat] else "loss"


def _hand_ids(player: dict) -> list[int]:
    return [c.get("id") for c in (player.get("hand") or [])
            if isinstance(c, dict) and isinstance(c.get("id"), int)]


def _attachable_hand_indexes(options: list[dict]) -> set[int]:
    """Hand indexes the engine is offering as ATTACH right now."""
    out: set[int] = set()
    for option in options:
        if (option.get("type") == int(OptionType.ATTACH)
                and option.get("area") == int(AreaType.HAND)
                and isinstance(option.get("index"), int)):
            out.add(option["index"])
    return out


def _energy_on(pokemon: dict) -> list[int]:
    return [c.get("id") for c in (pokemon.get("energyCards") or [])
            if isinstance(c, dict)]


def _covers(energy_id: int, host_is_fighting: bool) -> bool:
    """Engine-verified prevention matrix."""
    if energy_id == MIST_ENERGY:
        return True
    if energy_id == ROCK_FIGHTING:
        return host_is_fighting
    return False


SWITCH_CARD: Final[int] = 1123


def _switch_available(obs_dict: dict, ours: dict) -> bool:
    """Could we have put a different body in the Active Spot right now?

    Either the engine is offering a RETREAT, or a Switch (1123) sits in
    hand as a playable option. Both change which Pokémon eats the next
    attack, which is what repositioning cover requires.
    """
    select = obs_dict.get("select")
    options = (select or {}).get("option") or []
    hand = _hand_ids(ours)
    for option in options:
        if option.get("type") == int(OptionType.RETREAT):
            return True
        if option.get("type") == int(OptionType.PLAY):
            index = option.get("index")
            if (isinstance(index, int) and 0 <= index < len(hand)
                    and hand[index] == SWITCH_CARD):
                return True
    return False


def _find_pokemon(player: dict, serial: int | None) -> dict | None:
    if serial is None:
        return None
    for pokemon in _in_play(player):
        if pokemon.get("serial") == serial:
            return pokemon
    return None


def audit_episode(path: Path, index: CardIndex, effects: _EffectTable,
                  fighting_ids: frozenset[int],
                  team: str) -> EpisodeAudit | None:
    try:
        with open(path, encoding="utf-8") as fh:
            replay = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    seat = _our_seat(replay, team)
    if seat is None:
        return None

    revealed: dict[int, Counter[str]] = {0: Counter(), 1: Counter()}
    for (player, _serial), card_id in _observed_serials(replay).items():
        card = index.get_card(card_id)
        if card is not None and player in revealed:
            revealed[player][card.card_name] += 1
    if label_archetype(revealed[seat]) != OUR_DECK_LABEL:
        return None                      # deck sentinel

    episode_id = (replay.get("info") or {}).get("EpisodeId")
    if not isinstance(episode_id, int):
        episode_id = int(path.stem) if path.stem.isdigit() else 0
    audit = EpisodeAudit(
        episode_id=episode_id,
        opponent_archetype=label_archetype(revealed[1 - seat]),
        result=_result(replay, seat),
    )

    # our decisions since the opponent's last attack = the window in
    # which we could have prepared for the next one
    window: list[dict] = []

    for agent_index, obs_dict, action in _iter_decisions(replay):
        state = obs_dict.get("current") or {}
        players = state.get("players") or []
        if len(players) < 2:
            continue

        if agent_index == seat:
            audit.our_decisions += 1
            _tally_mist_waste(audit, players[seat], obs_dict, action,
                              window, fighting_ids)
            window.append({"obs": obs_dict, "action": action})
            _tally_volume(audit, players[seat], obs_dict, fighting_ids)
            continue

        # opponent decision: is it an attack that our energies could stop?
        attack_id = _chosen_attack(obs_dict, action)
        if attack_id is None or not effects.preventable(attack_id):
            continue
        ours = players[seat]
        victim = next((p for p in (ours.get("active") or [])
                       if isinstance(p, dict)), None)
        attached = _energy_on(victim) if victim else []
        victim_id = victim.get("id") if victim else None
        victim_fighting = victim_id in fighting_ids
        if any(_covers(e, victim_fighting) for e in attached):
            window = []
            continue                     # already prevented: not a hit

        hit = _classify(audit.episode_id, state.get("turn"), attack_id,
                        victim, victim_id, victim_fighting, window,
                        seat, fighting_ids)
        audit.hits.append(hit)
        window = []
    return audit


def _tally_mist_waste(audit: EpisodeAudit, ours: dict, obs_dict: dict,
                      action: list[int], window: list[dict],
                      fighting_ids: frozenset[int]) -> None:
    """Did we just spend a Mist on a {F} host with a Rock also in hand?

    Read from the OBSERVED transition (previous decision -> this one),
    not from option order, for the reason given in the module docstring.
    """
    if not window:
        return
    previous = window[-1]["obs"]
    prev_players = (previous.get("current") or {}).get("players") or []
    seat = (previous.get("current") or {}).get("yourIndex")
    if not isinstance(seat, int) or seat >= len(prev_players):
        return
    landed = _attach_delta(prev_players[seat], ours)
    if landed is None:
        return
    energy_id, host = landed
    if energy_id != MIST_ENERGY or host.get("id") not in fighting_ids:
        return
    audit.mist_on_fighting_total += 1
    # was a Rock Fighting equally attachable at that moment?
    options = (previous.get("select") or {}).get("option") or []
    hand = _hand_ids(prev_players[seat])
    for hand_index in _attachable_hand_indexes(options):
        if 0 <= hand_index < len(hand) and hand[hand_index] == ROCK_FIGHTING:
            audit.mist_wasted_on_fighting += 1
            return


def _tally_volume(audit: EpisodeAudit, ours: dict, obs_dict: dict,
                  fighting_ids: frozenset[int]) -> None:
    """Count decisions where a protective attach onto the active was legal.

    This is the VOLUME a new pilot rule would act on. A rule that fires on
    0.6% of decisions cannot move a winrate no matter how right it is, so
    the population size is reported before any rule is proposed.
    """
    turn = (obs_dict.get("current") or {}).get("turn")
    if isinstance(turn, int):
        audit.our_turns.add(turn)
    active = next((p for p in (ours.get("active") or [])
                   if isinstance(p, dict)), None)
    if active is None:
        return
    active_fighting = active.get("id") in fighting_ids
    if any(_covers(e, active_fighting) for e in _energy_on(active)):
        audit.covered_already += 1
        return
    select = obs_dict.get("select")
    options = (select or {}).get("option") or []
    attachable = _attachable_hand_indexes(options)
    if not attachable:
        return
    hand = _hand_ids(ours)
    for hand_index in attachable:
        if 0 <= hand_index < len(hand) and _covers(hand[hand_index],
                                                   active_fighting):
            audit.coverable_decisions += 1
            if isinstance(turn, int):
                audit.coverable_turns.add(turn)
            return


def _classify(episode_id: int, turn: Any, attack_id: int,
              victim: dict | None, victim_id: int | None,
              victim_fighting: bool, window: list[dict], seat: int,
              fighting_ids: frozenset[int]) -> Hit:
    hit = Hit(episode_id=episode_id,
              turn=turn if isinstance(turn, int) else None,
              attack_id=attack_id, victim_id=victim_id,
              victim_is_fighting=victim_fighting,
              verdict=UNKNOWN_WINDOW,
              decisions_in_window=len(window))
    if not window:
        return hit

    victim_serial = victim.get("serial") if victim else None
    held_eligible = False
    attach_made = False
    attach_target: dict | None = None
    attach_card: int | None = None
    protective_on_bench = False

    for step_index, step in enumerate(window):
        obs_dict = step["obs"]
        state = obs_dict.get("current") or {}
        players = state.get("players") or []
        if len(players) < 2:
            continue
        ours = players[seat]

        # (a) did we hold an energy that could legally cover the victim,
        #     with the engine offering it as an attach?
        select = obs_dict.get("select")
        options = (select or {}).get("option") or []
        attachable = _attachable_hand_indexes(options)
        hand = _hand_ids(ours)
        for hand_index in attachable:
            if (0 <= hand_index < len(hand)
                    and _covers(hand[hand_index], victim_fighting)):
                held_eligible = True
                hit.held_mist |= hand[hand_index] == MIST_ENERGY
                hit.held_rock |= hand[hand_index] == ROCK_FIGHTING

        # (b) a protective energy already parked on a NON-victim body
        for pokemon in _in_play(ours):
            if pokemon.get("serial") == victim_serial:
                continue
            host_fighting = pokemon.get("id") in fighting_ids
            if any(_covers(e, host_fighting) for e in _energy_on(pokemon)):
                protective_on_bench = True
                if _switch_available(obs_dict, ours):
                    hit.cover_parked_and_switchable = True

        # (c) where an attach actually landed: compare this state with
        #     the next one in the window (observed, not inferred)
        if step_index + 1 < len(window):
            after = ((window[step_index + 1]["obs"].get("current") or {})
                     .get("players") or [])
            if len(after) > seat:
                landed = _attach_delta(ours, after[seat])
                if landed is not None:
                    attach_made = True
                    attach_card, attach_target = landed

    hit.attach_made = attach_made
    hit.attach_card_id = attach_card
    hit.attach_target_id = (attach_target.get("id") if attach_target
                            else None)
    hit.protective_on_bench = protective_on_bench

    if not held_eligible:
        hit.verdict = INEVITABLE_NOT_HELD
    elif not attach_made:
        hit.verdict = AVOIDABLE_NO_ATTACH
    elif attach_target is not None and \
            attach_target.get("serial") == victim_serial:
        # attached to the victim but with a card that does not cover it
        hit.verdict = AVOIDABLE_WRONG_CARD
    else:
        hit.verdict = AVOIDABLE_ELSEWHERE
    return hit


def _attach_delta(before: dict, after: dict) -> tuple[int, dict] | None:
    """(energy card id, host) for an energy that appeared between states."""
    seen: dict[int, list[int]] = {}
    for pokemon in _in_play(before):
        serial = pokemon.get("serial")
        if isinstance(serial, int):
            seen[serial] = sorted(
                e for e in _energy_on(pokemon) if isinstance(e, int))
    for pokemon in _in_play(after):
        serial = pokemon.get("serial")
        if not isinstance(serial, int):
            continue
        now = sorted(e for e in _energy_on(pokemon) if isinstance(e, int))
        was = seen.get(serial)
        if was is None or len(now) <= len(was):
            continue
        remaining = list(was)
        for energy in now:
            if energy in remaining:
                remaining.remove(energy)
            else:
                return energy, pokemon
    return None


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def _pct(num: int, den: int) -> str:
    return f"{num/den:6.1%}" if den else "     -"


def report(audits: list[EpisodeAudit], index: CardIndex) -> dict:
    def name(card_id: int | None) -> str:
        card = index.get_card(card_id) if card_id is not None else None
        return card.card_name if card is not None else f"id={card_id}"

    hits = [h for a in audits for h in a.hits]
    decisions = sum(a.our_decisions for a in audits)
    coverable = sum(a.coverable_decisions for a in audits)
    covered = sum(a.covered_already for a in audits)

    print("=" * 78)
    print(f"AMOSTRA: {len(audits)} episódios (submissão 54917180 + sentinela "
          f"de deck), {decisions} decisões nossas")
    print(f"efeitos preveníveis que chegaram SEM prevenção: {len(hits)}")
    print("=" * 78)

    verdicts = Counter(h.verdict for h in hits)
    print("\n-- VEREDITO: política ou economia? --")
    for verdict, count in verdicts.most_common():
        print(f"  {verdict:44s} {count:5d} {_pct(count, len(hits))}")
    avoidable = sum(c for v, c in verdicts.items() if v in AVOIDABLE)
    inevitable = verdicts.get(INEVITABLE_NOT_HELD, 0)
    decided = avoidable + inevitable
    print(f"\n  EVITÁVEL   {avoidable:5d} / {decided} = "
          f"{_pct(avoidable, decided)}")
    print(f"  INEVITÁVEL {inevitable:5d} / {decided} = "
          f"{_pct(inevitable, decided)}")

    print("\n-- por arquétipo do oponente --")
    by_arch: dict[str, list[Hit]] = defaultdict(list)
    for audit in audits:
        for hit in audit.hits:
            by_arch[audit.opponent_archetype].append(hit)
    for arch, group in sorted(by_arch.items(), key=lambda kv: -len(kv[1])):
        av = sum(1 for h in group if h.verdict in AVOIDABLE)
        inev = sum(1 for h in group if h.verdict == INEVITABLE_NOT_HELD)
        print(f"  {arch:34s} hits={len(group):5d}  evitável "
              f"{_pct(av, av + inev)}  inevitável {_pct(inev, av + inev)}")

    print("\n-- padrão dos EVITÁVEIS (para onde foi a energia) --")
    avoid = [h for h in hits if h.verdict in AVOIDABLE]
    targets = Counter(name(h.attach_target_id) for h in avoid if h.attach_made)
    cards = Counter(name(h.attach_card_id) for h in avoid if h.attach_made)
    print(f"  anexou algo na janela: {sum(1 for h in avoid if h.attach_made)}"
          f"/{len(avoid)}")
    for label, counter in (("alvo da anexação", targets),
                           ("carta anexada", cards)):
        print(f"  {label}:")
        for key, count in counter.most_common(5):
            print(f"      {key[:26]:26s} {count:5d} "
                  f"{_pct(count, sum(counter.values()))}")
    parked = sum(1 for h in avoid if h.protective_on_bench)
    print(f"  protetora JÁ estava em outro corpo nosso: {parked}/{len(avoid)}"
          f" = {_pct(parked, len(avoid))}")
    print(f"  tinha Mist elegível: "
          f"{sum(1 for h in avoid if h.held_mist)}  |  "
          f"tinha Rock elegível: {sum(1 for h in avoid if h.held_rock)}")

    print("\n-- vítimas --")
    victims = Counter(name(h.victim_id) for h in hits)
    for key, count in victims.most_common(6):
        print(f"  {key[:26]:26s} {count:5d} {_pct(count, len(hits))}")

    print("\n-- INEVITÁVEIS: por que não havia energia elegível --")
    inev_hits = [h for h in hits if h.verdict == INEVITABLE_NOT_HELD]
    by_victim: Counter[str] = Counter(name(h.victim_id) for h in inev_hits)
    for key, count in by_victim.most_common(6):
        print(f"  vítima {key[:22]:22s} {count:5d} "
              f"{_pct(count, len(inev_hits))}")
    print("  (a Crustle/Dwebble são {G}: SÓ a Mist as cobre — 4 cartas em "
          "60. O Great Tusk aceita Mist OU Rock — 8 em 60.)")
    repositionable = sum(1 for h in inev_hits
                         if h.cover_parked_and_switchable)
    print(f"\n  destes, tinham cobertura ESTACIONADA em outro corpo E "
          f"switch/retreat legal: {repositionable}/{len(inev_hits)} = "
          f"{_pct(repositionable, len(inev_hits))}")
    print("  (esses são política por REPOSICIONAMENTO: o recurso existia, "
          "estava no slot errado)")

    print("\n-- VOLUME (o que uma regra nova tocaria) --")
    turns = sum(len(a.our_turns) for a in audits)
    coverable_turns = sum(len(a.coverable_turns) for a in audits)
    print(f"  decisões nossas totais:                       {decisions:6d}")
    print(f"  ativo JÁ coberto por protetora:               {covered:6d} "
          f"{_pct(covered, decisions)}")
    print(f"  ativo DESCOBERTO e attach protetora LEGAL:    {coverable:6d} "
          f"{_pct(coverable, decisions)}")
    print(f"  TURNOS nossos (denominador honesto):          {turns:6d}")
    print(f"  TURNOS em que a regra poderia agir:           "
          f"{coverable_turns:6d} {_pct(coverable_turns, turns)}"
          f"  <- população da regra")

    waste = sum(a.mist_wasted_on_fighting for a in audits)
    waste_total = sum(a.mist_on_fighting_total for a in audits)
    print("\n-- REGRA CANDIDATA: 'não gaste Mist num host {F} se há Rock' --")
    print(f"  Mist anexada a host {{F}}:                     "
          f"{waste_total:6d}")
    print(f"  ...com uma Rock IGUALMENTE anexável na mão:   {waste:6d} "
          f"{_pct(waste, waste_total)}  <- trocas que a regra faria")
    print(f"  = {waste/len(audits):.2f} por jogo em {len(audits)} jogos")
    print("  (a Rock cobre o Great Tusk tão bem quanto a Mist; a Mist é a "
          "ÚNICA carta do pool que cobre a Crustle {G}, e está no teto de 4)")

    return {
        "episodes": len(audits),
        "our_decisions": decisions,
        "hits": len(hits),
        "verdicts": dict(verdicts),
        "avoidable": avoidable,
        "inevitable": inevitable,
        "avoidable_share": (avoidable / decided) if decided else None,
        "by_archetype": {
            arch: {
                "hits": len(group),
                "avoidable": sum(1 for h in group if h.verdict in AVOIDABLE),
                "inevitable": sum(1 for h in group
                                  if h.verdict == INEVITABLE_NOT_HELD),
            } for arch, group in by_arch.items()},
        "attach_targets": {k: v for k, v in targets.items()},
        "attach_cards": {k: v for k, v in cards.items()},
        "protective_parked_elsewhere": parked,
        "inevitable_by_victim": dict(by_victim),
        "inevitable_repositionable": repositionable,
        "mist_wasted_on_fighting": sum(a.mist_wasted_on_fighting for a in audits),
        "mist_on_fighting_total": sum(a.mist_on_fighting_total for a in audits),
        "volume": {
            "decisions": decisions,
            "already_covered": covered,
            "coverable_uncovered": coverable,
            "coverable_share": coverable / decisions if decisions else None,
            "our_turns": turns,
            "coverable_turns": coverable_turns,
            "coverable_turn_share": (coverable_turns / turns
                                     if turns else None),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-dir", type=Path, default=EPISODES_DIR)
    parser.add_argument("--team", type=str, default=OUR_TEAM)
    parser.add_argument("--submission-index", type=Path, default=SUB_INDEX)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    index = CardIndex()
    effects = _EffectTable(EffectIndex())
    fighting_ids = _fighting_card_ids(index)

    allowed: set[int] | None = None
    if args.submission_index.exists():
        with open(args.submission_index, encoding="utf-8") as fh:
            payload = json.load(fh)
        allowed = {int(i) for i in payload.get("episode_ids") or []}
        print(f"filtro de submissão {payload.get('submission_id')}: "
              f"{len(allowed)} episódios elegíveis")

    audits: list[EpisodeAudit] = []
    skipped_submission = 0
    for path in sorted(args.episodes_dir.glob("*.json")):
        if allowed is not None and path.stem.isdigit() \
                and int(path.stem) not in allowed:
            skipped_submission += 1
            continue
        audit = audit_episode(path, index, effects, fighting_ids, args.team)
        if audit is not None:
            audits.append(audit)
    print(f"descartados por submissão: {skipped_submission}; "
          f"auditados: {len(audits)}\n")
    if not audits:
        raise SystemExit("nenhum episódio utilizável")

    payload = report(audits, index)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        print(f"\njson: {args.json}")


if __name__ == "__main__":
    main()
