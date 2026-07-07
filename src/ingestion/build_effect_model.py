"""Distill attack effect texts into the structured dim_effect table.

Hybrid extractor: deterministic sentence-level regex rules cover the
regular templates; the residue is exported to
data/processed/effects_to_review.csv and hand-curated into the versioned
src/ingestion/dim_effect_overrides.csv, which the build re-imports as
deterministic facts. No LLM, no network at runtime — dim_effect is a
plain Parquet consumed by heuristics (and later by the policy net).

Grain: one row per (attack_id, effect_seq); an attack with coin flip +
status + recoil yields several rows.

Run from the repo root:  python -m src.ingestion.build_effect_model
"""

from __future__ import annotations

import csv
import enum
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Final, Iterable

import polars as pl

from .build_card_model import DamageModifier, PROCESSED_DIR, REPO_ROOT
from .card_index import Attack, CardIndex

DIM_EFFECT_PARQUET: Final[Path] = PROCESSED_DIR / "dim_effect.parquet"
REVIEW_CSV: Final[Path] = PROCESSED_DIR / "effects_to_review.csv"
OVERRIDES_CSV: Final[Path] = Path(__file__).with_name("dim_effect_overrides.csv")


class EffectType(enum.IntEnum):
    """Coarse, heuristic-oriented classification of one effect clause."""

    NONE = 0                # vanilla attack, no text
    DAMAGE_BONUS = 1        # does X more damage (condition/per-unit)
    DAMAGE_SCALE = 2        # X damage for each <unit> (printed ×)
    STATUS = 3              # magnitude = SpecialConditionType code
    HEAL = 4
    DRAW = 5
    SEARCH = 6
    ENERGY_ACCEL = 7        # attach energy from deck/discard/hand
    ENERGY_DISCARD_SELF = 8
    ENERGY_DISCARD_OPP = 9
    SELF_DAMAGE = 10
    BENCH_DAMAGE = 11       # opponent's bench
    OWN_BENCH_DAMAGE = 12
    SNIPE = 13              # X damage to N of opponent's Pokémon (any)
    GUST = 14               # opponent's active is switched out
    SWITCH_SELF = 15
    SELF_LOCK = 16          # this Pokémon can't attack next turn
    OPP_LOCK = 17           # defending can't attack/retreat, hand locks
    PROTECT = 18            # prevent/reduce damage next turn
    PIERCE = 19             # ignores effects / weakness-resistance
    COUNTERS = 20           # put N damage counters
    DISRUPT_RESOURCE = 21   # opp discards/reveals hand, deck mill
    RECOVER = 22            # from discard pile to hand/bench
    PRIZE_MANIP = 23
    FAIL_UNLESS_HEADS = 24  # if tails, this attack does nothing
    FAIL_CONDITION = 25     # if <cond fails>, this attack does nothing
    HAND_COST = 26          # discard cards from your own hand as a cost
    UNKNOWN = 27


class EffectTarget(enum.IntEnum):
    SELF = 0
    OPP_ACTIVE = 1
    OPP_BENCH = 2
    OPP_ANY = 3
    OWN_BENCH = 4
    PLAYER = 5  # affects a player's hand/deck/prizes rather than a Pokémon


@dataclass(frozen=True, slots=True)
class EffectRow:
    attack_id: int
    effect_seq: int
    effect_type: int
    magnitude: int | None
    target: int | None
    condition: str | None
    coin_flip: bool
    confidence: float


# --------------------------------------------------------------------------- #
# Sentence-level rules
# --------------------------------------------------------------------------- #

_STATUS_CODE: Final[dict[str, int]] = {
    "poisoned": 0, "burned": 1, "asleep": 2, "paralyzed": 3, "confused": 4,
}

_NUM: Final[str] = r"(\d+)"

# Each rule: (EffectType, compiled pattern, magnitude group index or None,
#             target, condition tag). First match wins per rule; several
# rules may fire on the same sentence.
_Rule = tuple[EffectType, re.Pattern[str], int | None, EffectTarget | None, str | None]

_RULES: Final[list[_Rule]] = [
    (EffectType.FAIL_UNLESS_HEADS,
     re.compile(r"if tails, this attack does nothing"), None, None, "tails_fail"),
    (EffectType.FAIL_CONDITION,
     re.compile(r"if you can't, this attack does nothing|if you don't, this attack does nothing"),
     None, None, "cond_fail"),
    (EffectType.STATUS,
     re.compile(r"this pokemon is now (poisoned|burned|asleep|paralyzed|confused)"),
     1, EffectTarget.SELF, None),
    (EffectType.STATUS,
     re.compile(r"is now (poisoned|burned|asleep|paralyzed|confused)"),
     1, EffectTarget.OPP_ACTIVE, None),
    (EffectType.DAMAGE_BONUS,
     re.compile(rf"does {_NUM} more damage for each"), 1, EffectTarget.OPP_ACTIVE, "per_unit"),
    (EffectType.DAMAGE_BONUS,
     re.compile(rf"does {_NUM} more damage"), 1, EffectTarget.OPP_ACTIVE, "if"),
    (EffectType.DAMAGE_BONUS,
     re.compile(rf"you may do {_NUM} more damage"), 1, EffectTarget.OPP_ACTIVE, "optional"),
    (EffectType.DAMAGE_BONUS,
     re.compile(rf"does {_NUM} less damage for each"), 1, EffectTarget.OPP_ACTIVE, "per_unit_minus"),
    (EffectType.DAMAGE_BONUS,
     re.compile(rf"takes {_NUM} more damage from attacks"), 1, EffectTarget.OPP_ACTIVE, "vulnerability"),
    (EffectType.DAMAGE_SCALE,
     re.compile(rf"{_NUM} damage (?:for|times) each"), 1, EffectTarget.OPP_ACTIVE, "per_unit"),
    (EffectType.DAMAGE_SCALE,
     re.compile(rf"{_NUM} damage times the number"), 1, EffectTarget.OPP_ACTIVE, "per_unit"),
    (EffectType.DAMAGE_SCALE,
     re.compile(rf"{_NUM} damage [^.]{{0,50}}for each"), 1, EffectTarget.OPP_ACTIVE, "per_unit"),
    (EffectType.BENCH_DAMAGE,
     re.compile(rf"does {_NUM} damage to (?:each|\d+) of your opponent's benched"),
     1, EffectTarget.OPP_BENCH, "each"),
    (EffectType.BENCH_DAMAGE,
     re.compile(rf"does {_NUM} damage to 1 of your opponent's benched"),
     1, EffectTarget.OPP_BENCH, "one"),
    (EffectType.OWN_BENCH_DAMAGE,
     re.compile(rf"does {_NUM} damage to (?:each|1) of your(?! opponent) benched"),
     1, EffectTarget.OWN_BENCH, None),
    (EffectType.BENCH_DAMAGE,
     re.compile(rf"does {_NUM} damage to each benched pokemon \(both yours"),
     1, EffectTarget.OPP_BENCH, "each_both"),
    (EffectType.OWN_BENCH_DAMAGE,
     re.compile(rf"does {_NUM} damage to each benched pokemon \(both yours"),
     1, EffectTarget.OWN_BENCH, "each_both"),
    (EffectType.BENCH_DAMAGE,
     re.compile(rf"does {_NUM} damage to each benched pokemon that has"),
     1, EffectTarget.OPP_BENCH, "conditional_each"),
    (EffectType.SNIPE,
     re.compile(rf"does {_NUM} damage to (\d+) of your opponent's pokemon"),
     1, EffectTarget.OPP_ANY, None),
    (EffectType.SELF_DAMAGE,
     re.compile(rf"do(?:es)? {_NUM} damage to itself"), 1, EffectTarget.SELF, None),
    (EffectType.SELF_DAMAGE,
     re.compile(r"discard this pokemon and all attached"), None, EffectTarget.SELF, "self_discard"),
    (EffectType.COUNTERS,
     re.compile(rf"(?:put|place) (?:up to )?{_NUM} damage counters? on"), 1, EffectTarget.OPP_ANY, None),
    (EffectType.HEAL,
     re.compile(rf"heal {_NUM} damage"), 1, EffectTarget.SELF, None),
    (EffectType.HEAL,
     re.compile(r"heal all damage"), None, EffectTarget.SELF, "all"),
    (EffectType.HEAL,
     re.compile(r"heal from this pokemon"), None, EffectTarget.SELF, "equal_damage"),
    (EffectType.HEAL,
     re.compile(r"recovers from all special conditions"), None, EffectTarget.SELF, "cure_status"),
    (EffectType.DRAW,
     re.compile(rf"draw {_NUM} cards?"), 1, EffectTarget.PLAYER, None),
    (EffectType.DRAW,
     re.compile(r"draw a card"), None, EffectTarget.PLAYER, "one"),
    (EffectType.DRAW,
     re.compile(rf"draw cards until you have {_NUM}"), None, EffectTarget.PLAYER, "to_hand_size"),
    (EffectType.SEARCH,
     re.compile(r"search your deck"), None, EffectTarget.PLAYER, None),
    (EffectType.ENERGY_ACCEL,
     re.compile(r"attach [^.]{0,80}energy (?:cards? )?from your (deck|discard pile|hand)"),
     None, EffectTarget.SELF, None),
    (EffectType.ENERGY_DISCARD_SELF,
     re.compile(r"discard (?:an?|\d+|up to \d+|all|any amount of) [^.]{0,40}energy "
                r"(?:cards? )?from (?:this pokemon|your [^.]{0,25}pokemon)"),
     None, EffectTarget.SELF, None),
    (EffectType.ENERGY_DISCARD_SELF,
     re.compile(r"discard all energy from this pokemon"), None, EffectTarget.SELF, "all"),
    (EffectType.ENERGY_DISCARD_OPP,
     re.compile(r"discard (?:an?|\d+|all) [^.]{0,40}energy (?:cards? )?from (?:your opponent's|the defending)"),
     None, EffectTarget.OPP_ACTIVE, None),
    (EffectType.GUST,
     re.compile(r"switch in 1 of your opponent's benched"), None, EffectTarget.OPP_BENCH, "you_choose"),
    (EffectType.GUST,
     re.compile(r"switch out your opponent's active"), None, EffectTarget.OPP_ACTIVE, "opp_chooses"),
    (EffectType.SWITCH_SELF,
     re.compile(r"switch this pokemon with 1 of your benched"), None, EffectTarget.SELF, None),
    (EffectType.SELF_LOCK,
     re.compile(r"during your next turn, this pokemon can't"), None, EffectTarget.SELF, None),
    (EffectType.SELF_LOCK,
     re.compile(r"this pokemon can't attack during your next turn"), None, EffectTarget.SELF, None),
    (EffectType.SELF_LOCK,
     re.compile(r"this pokemon can't use [^.]{0,40}again"), None, EffectTarget.SELF, None),
    (EffectType.OPP_LOCK,
     re.compile(r"(?:the )?defending pokemon can't (?:use attacks|attack|retreat)"),
     None, EffectTarget.OPP_ACTIVE, None),
    (EffectType.OPP_LOCK,
     re.compile(r"that pokemon can't (?:retreat|use)"), None, EffectTarget.OPP_ACTIVE, None),
    (EffectType.OPP_LOCK,
     re.compile(r"can't use (?:attacks|that attack)"), None, EffectTarget.OPP_ACTIVE, None),
    (EffectType.OPP_LOCK,
     re.compile(r"that attack doesn't happen|tries to use an attack"),
     None, EffectTarget.OPP_ACTIVE, "coin_gate"),
    (EffectType.OPP_LOCK,
     re.compile(r"attacks used by the defending pokemon cost"),
     None, EffectTarget.OPP_ACTIVE, "cost_up"),
    (EffectType.OPP_LOCK,
     re.compile(r"your opponent can't play"), None, EffectTarget.PLAYER, None),
    (EffectType.PROTECT,
     re.compile(r"prevent all damage"), None, EffectTarget.SELF, "all"),
    (EffectType.PROTECT,
     re.compile(rf"takes? {_NUM} less damage"), 1, EffectTarget.SELF, None),
    (EffectType.PROTECT,
     re.compile(rf"do(?:es)? {_NUM} less damage"), 1, EffectTarget.SELF, None),
    (EffectType.OPP_LOCK,
     re.compile(r"choose 1 of your opponent's [^.]{0,40}attacks"),
     None, EffectTarget.OPP_ACTIVE, "select_attack"),
    (EffectType.PIERCE,
     re.compile(r"isn't affected by"), None, EffectTarget.OPP_ACTIVE, None),
    (EffectType.DISRUPT_RESOURCE,
     re.compile(r"your opponent (?:discards|reveals)"), None, EffectTarget.PLAYER, None),
    (EffectType.DISRUPT_RESOURCE,
     re.compile(r"discard the top (?:\d+ )?cards? of (?:your opponent's|each player's) deck"),
     None, EffectTarget.PLAYER, None),
    (EffectType.DISRUPT_RESOURCE,
     re.compile(r"discard the top (?:\d+ )?cards? of your deck"),
     None, EffectTarget.PLAYER, "self_deck"),
    (EffectType.DISRUPT_RESOURCE,
     re.compile(r"(?:discard|choose) (?:a|\d+) random cards? from your opponent's hand"),
     None, EffectTarget.PLAYER, None),
    (EffectType.DISRUPT_RESOURCE,
     re.compile(r"discard (?:a|that) stadium"), None, EffectTarget.PLAYER, "stadium"),
    (EffectType.STATUS,
     re.compile(r"is now affected by that special condition"), None, EffectTarget.OPP_ACTIVE, "chosen"),
    (EffectType.SWITCH_SELF,
     re.compile(r"(?:shuffle|put) this pokemon [^.]{0,45}into your (?:deck|hand)"),
     None, EffectTarget.SELF, "escape"),
    (EffectType.RECOVER,
     re.compile(r"from your discard pile (?:into|onto|to) your (?:hand|bench)"), None, EffectTarget.PLAYER, None),
    (EffectType.RECOVER,
     re.compile(r"put [^.]{0,60} from your discard pile"), None, EffectTarget.PLAYER, None),
    (EffectType.RECOVER,
     re.compile(r"(?:put|move) [^.]{0,70}into your hand"), None, EffectTarget.PLAYER, None),
    (EffectType.OPP_LOCK,
     re.compile(r"they can't (?:play|attach|use)"), None, EffectTarget.PLAYER, None),
    (EffectType.DISRUPT_RESOURCE,
     re.compile(r"discard all pokemon tools?"), None, EffectTarget.OPP_ACTIVE, None),
    (EffectType.ENERGY_ACCEL,
     re.compile(r"move (?:an?|\d+|up to \d+|all|any amount of) [^.]{0,40}energ"),
     None, EffectTarget.SELF, "move"),
    (EffectType.ENERGY_DISCARD_SELF,
     re.compile(r"shuffle [^.]{0,40}energ[^.]{0,40}into your deck"), None, EffectTarget.SELF, "to_deck"),
    (EffectType.HAND_COST,
     re.compile(r"discard (?:an?|\d+|up to \d+) [^.]{0,50}from your hand"), None, EffectTarget.PLAYER, None),
    (EffectType.FAIL_CONDITION,
     re.compile(r"can be used only if|you can't use this attack"), None, None, "usage_gate"),
    (EffectType.FAIL_CONDITION,
     re.compile(r"(?<!if tails, )this attack does nothing"), None, None, "gate"),
]

# Bookkeeping sentences that carry no gameplay signal of their own.
_IGNORE: Final[list[re.Pattern[str]]] = [
    re.compile(r"^\(?don't apply weakness"),
    re.compile(r"^then, shuffle (?:your deck|those cards)"),
    re.compile(r"^(?:then, )?shuffle your deck(?: afterward)?\.?$"),
    re.compile(r"^\(?existing (?:special conditions|effects)"),
    re.compile(r"^\(?damage is not an effect"),
    re.compile(r"^\(?your opponent chooses"),
    re.compile(r"^\(?discard all cards attached"),
    re.compile(r"^\(?this includes"),
    re.compile(r"^if heads, choose a special condition"),
]

# A sentence that ONLY announces coin flips modifies the following clauses.
_PURE_FLIP: Final[re.Pattern[str]] = re.compile(
    r"^(?:your opponent )?flips? a? ?(?:coins?|\d+ coins?)"
    r"(?: for each [^.]{0,60}?)?(?: until you get tails)?[.!?]?$"
)

_REGEX_CONFIDENCE: Final[float] = 0.95


def _normalize(text: str) -> str:
    text = text.replace("’", "'").replace("é", "e").replace("É", "E")
    text = text.replace("×", "x")
    return " ".join(text.lower().split())


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def extract_effects(attack: Attack) -> list[EffectRow]:
    """Deterministic extraction for one attack; UNKNOWN row per opaque sentence."""
    text = attack.effect or ""
    if not text.strip():
        return [EffectRow(attack.attack_id, 0, int(EffectType.NONE),
                          None, None, None, False, 1.0)]
    rows: list[EffectRow] = []
    seq = 0
    flip_pending = False  # a bare "Flip a coin." governs the NEXT clauses
    for sentence in _sentences(_normalize(text)):
        if any(pattern.search(sentence) for pattern in _IGNORE):
            continue
        if _PURE_FLIP.match(sentence):
            flip_pending = True
            continue
        coin = flip_pending or "flip" in sentence or "heads" in sentence or "tails" in sentence
        matched = False
        seen_types: set[int] = set()
        for effect_type, pattern, mag_group, target, condition in _RULES:
            if int(effect_type) in seen_types:
                continue
            m = pattern.search(sentence)
            if not m:
                continue
            matched = True
            seen_types.add(int(effect_type))
            magnitude: int | None = None
            if mag_group is not None:
                token = m.group(mag_group)
                magnitude = _STATUS_CODE[token] if token in _STATUS_CODE else int(token)
            if coin and condition == "if" and "if heads" in sentence:
                condition = "if_heads"
            rows.append(EffectRow(
                attack.attack_id, seq, int(effect_type), magnitude,
                int(target) if target is not None else None,
                condition, coin, _REGEX_CONFIDENCE,
            ))
            seq += 1
        if not matched:
            rows.append(EffectRow(attack.attack_id, seq, int(EffectType.UNKNOWN),
                                  None, None, None, coin, 0.0))
            seq += 1
    if not rows:  # every sentence was bookkeeping
        rows.append(EffectRow(attack.attack_id, 0, int(EffectType.NONE),
                              None, None, None, False, 1.0))
    return rows


# --------------------------------------------------------------------------- #
# Overrides
# --------------------------------------------------------------------------- #


def load_overrides(path: Path = OVERRIDES_CSV) -> dict[int, list[EffectRow]]:
    """Hand-curated rows, keyed by attack_id; they replace UNKNOWN rows."""
    overrides: dict[int, list[EffectRow]] = {}
    if not path.exists():
        return overrides
    with open(path, newline="", encoding="utf-8") as fh:
        for record in csv.DictReader(fh):
            attack_id = int(record["attack_id"])
            rows = overrides.setdefault(attack_id, [])
            rows.append(EffectRow(
                attack_id=attack_id,
                effect_seq=len(rows),  # re-sequenced on merge
                effect_type=int(EffectType[record["effect_type"]]),
                magnitude=int(record["magnitude"]) if record["magnitude"] else None,
                target=int(EffectTarget[record["target"]]) if record["target"] else None,
                condition=record["condition"] or None,
                coin_flip=record["coin_flip"].strip().lower() in ("1", "true", "yes"),
                confidence=1.0,
            ))
    return overrides


def merge_overrides(rows: list[EffectRow], overrides: dict[int, list[EffectRow]]) -> list[EffectRow]:
    """Replace each overridden attack's UNKNOWN rows with the curated rows."""
    merged: dict[int, list[EffectRow]] = {}
    for row in rows:
        merged.setdefault(row.attack_id, []).append(row)
    for attack_id, curated in overrides.items():
        kept = [r for r in merged.get(attack_id, [])
                if r.effect_type != int(EffectType.UNKNOWN)]
        merged[attack_id] = kept + curated
    out: list[EffectRow] = []
    for attack_id in sorted(merged):
        for seq, row in enumerate(merged[attack_id]):
            out.append(EffectRow(attack_id, seq, row.effect_type, row.magnitude,
                                 row.target, row.condition, row.coin_flip, row.confidence))
    return out


# --------------------------------------------------------------------------- #
# Build / persist / verify
# --------------------------------------------------------------------------- #

_EFFECT_SCHEMA: Final[dict[str, pl.DataType]] = {
    "attack_id": pl.UInt16(),
    "effect_seq": pl.UInt8(),
    "effect_type": pl.Int8(),
    "magnitude": pl.Int16(),
    "target": pl.Int8(),
    "condition": pl.Utf8(),
    "coin_flip": pl.Boolean(),
    "confidence": pl.Float32(),
}


def build_dim_effect(index: CardIndex) -> pl.DataFrame:
    rows: list[EffectRow] = []
    for attack in index.attacks.values():
        rows.extend(extract_effects(attack))
    rows = merge_overrides(rows, load_overrides())
    return pl.DataFrame(
        [(r.attack_id, r.effect_seq, r.effect_type, r.magnitude, r.target,
          r.condition, r.coin_flip, r.confidence) for r in rows],
        schema=_EFFECT_SCHEMA,
        orient="row",
    )


def write_review_csv(index: CardIndex, dim_effect: pl.DataFrame,
                     path: Path = REVIEW_CSV) -> int:
    """Export attacks that still carry UNKNOWN rows, damage attacks first."""
    unknown_ids = (
        dim_effect.filter(pl.col("effect_type") == int(EffectType.UNKNOWN))
        .get_column("attack_id").unique().to_list()
    )
    records = []
    for attack_id in unknown_ids:
        attack = index.attack(attack_id)
        records.append({
            "attack_id": attack_id,
            "card_name": index.card(attack.card_id).card_name,
            "move_name": attack.move_name,
            "damage_base": attack.damage_base,
            "effect_text": " ".join((attack.effect or "").split()),
        })
    records.sort(key=lambda r: (r["damage_base"] is None, r["attack_id"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["attack_id", "card_name", "move_name",
                                                "damage_base", "effect_text"])
        writer.writeheader()
        writer.writerows(records)
    return len(records)


def report(index: CardIndex, dim_effect: pl.DataFrame, audit_sample: int = 30) -> None:
    total = len(index.attacks)
    unknown_ids = set(
        dim_effect.filter(pl.col("effect_type") == int(EffectType.UNKNOWN))
        .get_column("attack_id").to_list()
    )
    damage_ids = {a.attack_id for a in index.attacks.values()
                  if (a.damage_base or 0) > 0}
    damage_unknown = unknown_ids & damage_ids

    print(f"attacks:                    {total}")
    print(f"effect rows:                {dim_effect.height}")
    print(f"attacks fully classified:   {total - len(unknown_ids)} "
          f"({(total - len(unknown_ids)) / total:.1%})")
    print(f"damage attacks w/ unknown:  {len(damage_unknown)} of {len(damage_ids)} "
          f"(goal: 0)")
    if damage_unknown:
        print(f"  ids: {sorted(damage_unknown)[:20]}")

    print("\n--- damage-modifier cross-check ---")
    scale_ids = set(dim_effect.filter(pl.col("effect_type") == int(EffectType.DAMAGE_SCALE))
                    .get_column("attack_id").to_list())
    mult_ids = {a.attack_id for a in index.attacks.values()
                if a.damage_modifier_code == int(DamageModifier.MULTIPLY)}
    print(f"printed 'x' attacks: {len(mult_ids)} | with DAMAGE_SCALE/BONUS row: "
          f"{len(mult_ids & (scale_ids | set(dim_effect.filter(pl.col('effect_type') == int(EffectType.DAMAGE_BONUS)).get_column('attack_id').to_list())))}")

    print(f"\n--- audit sample ({audit_sample} random attacks with text) ---")
    import random
    rng = random.Random(7)
    with_text = [a for a in index.attacks.values() if a.effect and a.effect.strip()]
    by_attack: dict[int, list[str]] = {}
    for row in dim_effect.iter_rows(named=True):
        label = EffectType(row["effect_type"]).name
        if row["magnitude"] is not None:
            label += f"({row['magnitude']})"
        if row["coin_flip"]:
            label += "*coin"
        by_attack.setdefault(row["attack_id"], []).append(label)
    for attack in rng.sample(with_text, min(audit_sample, len(with_text))):
        name = index.card(attack.card_id).card_name
        text = " ".join((attack.effect or "").split())[:70]
        print(f"[{attack.attack_id:>4}] {name[:24]:24s} {attack.move_name[:18]:18s} "
              f"-> {', '.join(by_attack.get(attack.attack_id, []))}")
        print(f"       {text}")


class EffectIndex:
    """O(1) runtime lookup: attack_id -> tuple of EffectRow (dict-backed)."""

    __slots__ = ("_by_attack",)

    def __init__(self, parquet_path: Path = DIM_EFFECT_PARQUET) -> None:
        table = pl.read_parquet(parquet_path)
        by_attack: dict[int, list[EffectRow]] = {}
        for row in table.iter_rows(named=True):
            by_attack.setdefault(row["attack_id"], []).append(EffectRow(**row))
        self._by_attack: dict[int, tuple[EffectRow, ...]] = {
            attack_id: tuple(rows) for attack_id, rows in by_attack.items()
        }

    def effects_of(self, attack_id: int) -> tuple[EffectRow, ...]:
        """Fail-safe: unknown attack ids yield an empty tuple."""
        return self._by_attack.get(attack_id, ())

    def has(self, attack_id: int, effect_type: EffectType) -> bool:
        return any(r.effect_type == int(effect_type) for r in self.effects_of(attack_id))

    def __len__(self) -> int:
        return len(self._by_attack)


def main() -> None:
    index = CardIndex()
    dim_effect = build_dim_effect(index)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dim_effect.write_parquet(DIM_EFFECT_PARQUET, compression="zstd")
    n_review = write_review_csv(index, dim_effect)
    print(f"wrote {DIM_EFFECT_PARQUET.relative_to(REPO_ROOT)} "
          f"({DIM_EFFECT_PARQUET.stat().st_size:,} bytes)")
    print(f"wrote {REVIEW_CSV.relative_to(REPO_ROOT)} ({n_review} attacks to review)\n")
    report(index, dim_effect)


if __name__ == "__main__":
    main()
