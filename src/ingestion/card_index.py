"""O(1) runtime lookup over the processed card star schema.

CardIndex loads the Parquet tables produced by build_card_model.py once,
converts them into plain-Python frozen dataclasses keyed by integer ids,
and never touches a DataFrame again — every lookup is a dict access.

Attack ids are engine-aligned: index.attack(id) resolves the same id the
engine emits in Option.attackId (verified by src/ingestion/reconcile.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import polars as pl

from .build_card_model import (
    BRIDGE_ENERGY_PARQUET,
    DIM_ATTACK_PARQUET,
    DIM_CARD_PARQUET,
    DIM_SKILL_PARQUET,
    PROCESSED_DIR,
)


@dataclass(frozen=True, slots=True)
class Attack:
    """One real attack, keyed by the engine's attack id."""

    attack_id: int
    card_id: int
    move_name: str
    damage_base: int | None
    damage_modifier_code: int | None
    cost_total: int
    effect: str | None
    # (energy_type_code, qty) pairs, sorted by energy_type_code.
    cost: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class Skill:
    """One Pokémon ability row (subset of the engine's CardData.skills)."""

    skill_id: int
    card_id: int
    skill_name: str
    effect: str | None


@dataclass(frozen=True, slots=True)
class Card:
    """One card, with its attack/skill ids in CSV order."""

    card_id: int
    card_name: str
    expansion: str
    collection_no: str | None
    stage_code: int | None
    category: str | None
    previous_stage: str | None
    hp: int | None
    type_code: int | None
    weakness_code: int | None
    resistance_code: int | None
    retreat_cost: int | None
    is_ex: bool
    is_mega_ex: bool
    is_ace_spec: bool
    # Tera Pokémon take no damage from attacks while on the Bench
    # (engine behavior verified by tests/test_tera_bench_immunity.py).
    is_tera: bool
    attack_ids: tuple[int, ...]
    skill_ids: tuple[int, ...]


class CardIndex:
    """Dict-backed index over dim_card / dim_attack / dim_skill / bridge."""

    __slots__ = ("_cards", "_attacks", "_skills")

    def __init__(self, processed_dir: Path = PROCESSED_DIR) -> None:
        dim_card = pl.read_parquet(processed_dir / DIM_CARD_PARQUET.name)
        dim_attack = pl.read_parquet(processed_dir / DIM_ATTACK_PARQUET.name)
        dim_skill = pl.read_parquet(processed_dir / DIM_SKILL_PARQUET.name)
        bridge = pl.read_parquet(processed_dir / BRIDGE_ENERGY_PARQUET.name)

        costs: dict[int, list[tuple[int, int]]] = {}
        for attack_id, _card_id, energy_type_code, qty in bridge.iter_rows():
            costs.setdefault(attack_id, []).append((energy_type_code, qty))

        attacks: dict[int, Attack] = {}
        attack_ids_by_card: dict[int, list[int]] = {}
        for row in dim_attack.iter_rows(named=True):
            attack = Attack(
                attack_id=row["attack_id"],
                card_id=row["card_id"],
                move_name=row["move_name"],
                damage_base=row["damage_base"],
                damage_modifier_code=row["damage_modifier_code"],
                cost_total=row["cost_total"],
                effect=row["effect"],
                cost=tuple(costs.get(row["attack_id"], ())),
            )
            attacks[attack.attack_id] = attack
            attack_ids_by_card.setdefault(attack.card_id, []).append(attack.attack_id)

        skills: dict[int, Skill] = {}
        skill_ids_by_card: dict[int, list[int]] = {}
        for row in dim_skill.iter_rows(named=True):
            skill = Skill(**row)
            skills[skill.skill_id] = skill
            skill_ids_by_card.setdefault(skill.card_id, []).append(skill.skill_id)

        cards: dict[int, Card] = {}
        for row in dim_card.iter_rows(named=True):
            cards[row["card_id"]] = Card(
                attack_ids=tuple(attack_ids_by_card.get(row["card_id"], ())),
                skill_ids=tuple(skill_ids_by_card.get(row["card_id"], ())),
                **row,
            )

        self._cards: Mapping[int, Card] = MappingProxyType(cards)
        self._attacks: Mapping[int, Attack] = MappingProxyType(attacks)
        self._skills: Mapping[int, Skill] = MappingProxyType(skills)

    # ------------------------------------------------------------------ #
    # O(1) lookups
    # ------------------------------------------------------------------ #

    def card(self, card_id: int) -> Card:
        """Strict lookup; raises KeyError on unknown id."""
        return self._cards[card_id]

    def attack(self, attack_id: int) -> Attack:
        """Strict lookup; raises KeyError on unknown id."""
        return self._attacks[attack_id]

    def get_card(self, card_id: int) -> Card | None:
        """Fail-safe lookup: None on unknown id (engine may emit new ids)."""
        return self._cards.get(card_id)

    def get_attack(self, attack_id: int) -> Attack | None:
        """Fail-safe lookup: None on unknown id (engine may emit new ids)."""
        return self._attacks.get(attack_id)

    def get_skill(self, skill_id: int) -> Skill | None:
        """Fail-safe lookup: None on unknown id."""
        return self._skills.get(skill_id)

    def attacks_of(self, card_id: int) -> tuple[Attack, ...]:
        card = self._cards.get(card_id)
        if card is None:
            return ()
        return tuple(self._attacks[a] for a in card.attack_ids)

    def skills_of(self, card_id: int) -> tuple[Skill, ...]:
        card = self._cards.get(card_id)
        if card is None:
            return ()
        return tuple(self._skills[s] for s in card.skill_ids)

    @property
    def cards(self) -> Mapping[int, Card]:
        return self._cards

    @property
    def attacks(self) -> Mapping[int, Attack]:
        return self._attacks

    @property
    def skills(self) -> Mapping[int, Skill]:
        return self._skills

    def __len__(self) -> int:
        return len(self._cards)


# --------------------------------------------------------------------------- #
# Build + profiling entry point
# --------------------------------------------------------------------------- #


def _profile() -> None:
    import random
    import time
    import tracemalloc

    from . import build_card_model

    print("=== build ===")
    t0 = time.perf_counter()
    model = build_card_model.build_star_schema()
    build_card_model.persist(model)
    print(f"pipeline ran in {time.perf_counter() - t0:.3f}s")
    print(f"cards:   {model.dim_card.height}")
    print(f"attacks: {model.dim_attack.height}")
    print(f"skills:  {model.dim_skill.height}")
    print(f"costs:   {model.bridge_attack_energy.height}")

    print("\n=== CardIndex memory (tracemalloc) ===")
    tracemalloc.start()
    before = tracemalloc.take_snapshot()
    index = CardIndex()
    after = tracemalloc.take_snapshot()
    tracemalloc.stop()
    allocated = sum(stat.size_diff for stat in after.compare_to(before, "filename"))
    print(f"resident allocations for CardIndex: {allocated / 1024:.1f} KiB")

    print("\n=== lookup latency (10k random lookups) ===")
    rng = random.Random(42)
    card_ids = list(index.cards.keys())
    attack_ids = list(index.attacks.keys())
    card_queries = [rng.choice(card_ids) for _ in range(10_000)]
    attack_queries = [rng.choice(attack_ids) for _ in range(10_000)]

    t0 = time.perf_counter()
    for cid in card_queries:
        index.get_card(cid)
    card_us = (time.perf_counter() - t0) / len(card_queries) * 1e6

    t0 = time.perf_counter()
    for aid in attack_queries:
        index.get_attack(aid)
    attack_us = (time.perf_counter() - t0) / len(attack_queries) * 1e6

    print(f"card lookup:   {card_us:.3f} us/query")
    print(f"attack lookup: {attack_us:.3f} us/query")

    sample = index.card(card_ids[len(card_ids) // 2])
    print(f"\nsample: {sample.card_name} (hp={sample.hp}) -> "
          f"{[index.attack(a).move_name for a in sample.attack_ids]}")


if __name__ == "__main__":
    _profile()
