"""O(1) runtime lookup over the processed card star schema.

CardIndex loads the Parquet tables produced by build_card_model.py once,
converts them into plain-Python frozen dataclasses keyed by integer ids,
and never touches a DataFrame again — every lookup is a dict access.
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
    PROCESSED_DIR,
)


@dataclass(frozen=True, slots=True)
class Attack:
    """One move/ability row (see MoveKind for kind_code semantics)."""

    attack_id: int
    card_id: int
    move_name: str
    kind_code: int
    damage_base: int | None
    damage_modifier_code: int | None
    cost_total: int | None
    effect: str | None
    # (energy_type_code, qty) pairs, sorted by energy_type_code.
    cost: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class Card:
    """One card, with its attack ids in CSV order."""

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
    attack_ids: tuple[int, ...]


class CardIndex:
    """Dict-backed index over dim_card / dim_attack / bridge_attack_energy."""

    __slots__ = ("_cards", "_attacks")

    def __init__(self, processed_dir: Path = PROCESSED_DIR) -> None:
        dim_card = pl.read_parquet(processed_dir / DIM_CARD_PARQUET.name)
        dim_attack = pl.read_parquet(processed_dir / DIM_ATTACK_PARQUET.name)
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
                kind_code=row["kind_code"],
                damage_base=row["damage_base"],
                damage_modifier_code=row["damage_modifier_code"],
                cost_total=row["cost_total"],
                effect=row["effect"],
                cost=tuple(costs.get(row["attack_id"], ())),
            )
            attacks[attack.attack_id] = attack
            attack_ids_by_card.setdefault(attack.card_id, []).append(attack.attack_id)

        cards: dict[int, Card] = {}
        for row in dim_card.iter_rows(named=True):
            cards[row["card_id"]] = Card(
                attack_ids=tuple(attack_ids_by_card.get(row["card_id"], ())),
                **row,
            )

        self._cards: Mapping[int, Card] = MappingProxyType(cards)
        self._attacks: Mapping[int, Attack] = MappingProxyType(attacks)

    # ------------------------------------------------------------------ #
    # O(1) lookups
    # ------------------------------------------------------------------ #

    def card(self, card_id: int) -> Card:
        return self._cards[card_id]

    def attack(self, attack_id: int) -> Attack:
        return self._attacks[attack_id]

    def attacks_of(self, card_id: int) -> tuple[Attack, ...]:
        return tuple(self._attacks[a] for a in self._cards[card_id].attack_ids)

    @property
    def cards(self) -> Mapping[int, Card]:
        return self._cards

    @property
    def attacks(self) -> Mapping[int, Attack]:
        return self._attacks

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
    print(f"moves:   {model.dim_attack.height}")
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
        index.card(cid)
    card_us = (time.perf_counter() - t0) / len(card_queries) * 1e6

    t0 = time.perf_counter()
    for aid in attack_queries:
        index.attack(aid)
    attack_us = (time.perf_counter() - t0) / len(attack_queries) * 1e6

    print(f"card lookup:   {card_us:.3f} µs/query")
    print(f"attack lookup: {attack_us:.3f} µs/query")

    sample = index.card(card_ids[len(card_ids) // 2])
    print(f"\nsample: {sample.card_name} (hp={sample.hp}) -> "
          f"{[index.attack(a).move_name for a in sample.attack_ids]}")


if __name__ == "__main__":
    _profile()
