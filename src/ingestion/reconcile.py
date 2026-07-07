"""Reconcile the CardIndex against the official engine's card database.

Compares cg.api.all_card_data() / cg.api.all_attack() with the ids in the
processed star schema and reports:

    missing  = engine ids absent from the index  (CRITICAL: the engine can
               reference these ids in-game and enrich() would return None)
    extra    = index ids unknown to the engine   (logged only)

Attack names are also compared id-by-id, since an id that exists on both
sides but points at a different attack is worse than a missing one.

Exit code is non-zero when anything critical is found, so this can gate CI.

Run from the repo root:  python -m src.ingestion.reconcile
"""

from __future__ import annotations

import sys
from typing import NamedTuple

from cg import api

from .card_index import CardIndex


class Report(NamedTuple):
    label: str
    engine_count: int
    index_count: int
    missing: frozenset[int]
    extra: frozenset[int]
    name_mismatches: tuple[tuple[int, str, str], ...]  # (id, engine name, index name)
    extra_is_critical: bool = False  # e.g. a false is_tera flag corrupts features

    @property
    def critical(self) -> bool:
        if self.extra_is_critical and self.extra:
            return True
        return bool(self.missing) or bool(self.name_mismatches)


def _norm(name: str) -> str:
    """Names differ only by stray whitespace between CSV and engine."""
    return " ".join(name.split())


def reconcile_cards(index: CardIndex) -> Report:
    engine_cards = api.all_card_data()
    engine_ids = {c.cardId for c in engine_cards}
    index_ids = set(index.cards.keys())

    mismatches = tuple(
        (c.cardId, c.name, index.get_card(c.cardId).card_name)  # type: ignore[union-attr]
        for c in engine_cards
        if c.cardId in index_ids
        and _norm(c.name) != _norm(index.get_card(c.cardId).card_name)  # type: ignore[union-attr]
    )
    return Report(
        label="cards (all_card_data vs dim_card)",
        engine_count=len(engine_ids),
        index_count=len(index_ids),
        missing=frozenset(engine_ids - index_ids),
        extra=frozenset(index_ids - engine_ids),
        name_mismatches=mismatches,
    )


def reconcile_tera(index: CardIndex) -> Report:
    """dim_card.is_tera must equal the engine's CardData.tera flag exactly."""
    engine_cards = api.all_card_data()
    engine_ids = {c.cardId for c in engine_cards if c.tera}
    index_ids = {card_id for card_id, card in index.cards.items() if card.is_tera}
    return Report(
        label="tera flag (CardData.tera vs dim_card.is_tera)",
        engine_count=len(engine_ids),
        index_count=len(index_ids),
        missing=frozenset(engine_ids - index_ids),
        extra=frozenset(index_ids - engine_ids),
        name_mismatches=(),
        extra_is_critical=True,
    )


def reconcile_skills(index: CardIndex) -> Report:
    """Our dim_skill must cover the engine's Pokémon ability skills by name.

    The engine also registers Trainer/Energy card texts as CardData.skills;
    those live in dim_card/dim_effect territory, so only POKEMON cards with
    a skill whose name matches no ability row count as missing.
    """
    engine_pairs = {
        (c.cardId, _norm(s.name))
        for c in api.all_card_data()
        if c.cardType == api.CardType.POKEMON
        for s in c.skills
    }
    index_pairs = {
        (skill.card_id, _norm(skill.skill_name)) for skill in index.skills.values()
    }
    missing_cards = frozenset(card_id for card_id, _name in engine_pairs - index_pairs)
    extra_cards = frozenset(card_id for card_id, _name in index_pairs - engine_pairs)
    return Report(
        label="skills (POKEMON CardData.skills vs dim_skill, by card+name)",
        engine_count=len(engine_pairs),
        index_count=len(index_pairs),
        missing=missing_cards,
        extra=extra_cards,
        name_mismatches=(),
    )


def reconcile_attacks(index: CardIndex) -> Report:
    engine_attacks = api.all_attack()
    engine_ids = {a.attackId for a in engine_attacks}
    index_ids = set(index.attacks.keys())

    mismatches = tuple(
        (a.attackId, a.name, index.get_attack(a.attackId).move_name)  # type: ignore[union-attr]
        for a in engine_attacks
        if a.attackId in index_ids
        and _norm(a.name) != _norm(index.get_attack(a.attackId).move_name)  # type: ignore[union-attr]
    )
    return Report(
        label="attacks (all_attack vs dim_attack)",
        engine_count=len(engine_ids),
        index_count=len(index_ids),
        missing=frozenset(engine_ids - index_ids),
        extra=frozenset(index_ids - engine_ids),
        name_mismatches=mismatches,
    )


def _print_report(report: Report) -> None:
    print(f"--- {report.label} ---")
    print(f"engine ids: {report.engine_count:>5}   index ids: {report.index_count:>5}")
    print(f"missing (engine - index): {len(report.missing):>4}  "
          f"{'<- CRITICAL' if report.missing else 'OK'}")
    if report.missing:
        print(f"  ids: {sorted(report.missing)[:20]}"
              f"{' ...' if len(report.missing) > 20 else ''}")
    print(f"extra   (index - engine): {len(report.extra):>4}  "
          f"{'<- CRITICAL' if report.extra_is_critical and report.extra else '(log only)'}")
    if report.extra:
        print(f"  ids: {sorted(report.extra)[:20]}"
              f"{' ...' if len(report.extra) > 20 else ''}")
    print(f"name mismatches on shared ids: {len(report.name_mismatches):>4}  "
          f"{'<- CRITICAL' if report.name_mismatches else 'OK'}")
    for id_, engine_name, index_name in report.name_mismatches[:10]:
        print(f"  id {id_}: engine={engine_name!r} index={index_name!r}")
    print()


def main() -> int:
    index = CardIndex()
    reports = (
        reconcile_cards(index),
        reconcile_attacks(index),
        reconcile_skills(index),
        reconcile_tera(index),
    )
    for report in reports:
        _print_report(report)
    if any(r.critical for r in reports):
        print("RESULT: FAIL — index does not cover the engine's id space.")
        return 1
    print("RESULT: OK — CardIndex fully covers engine cards and attacks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
