"""Build a star-schema card model from the raw EN_Card_Data.csv.

The raw CSV is at CARD x MOVE grain (Card ID repeats for cards with
multiple attacks). This module normalizes it into three tables:

    dim_card             1 row per Card ID
    dim_attack           1 row per move/ability row (attack_id PK)
    bridge_attack_energy 1 row per (attack_id, energy_type) cost entry

All surrogate/natural keys are integers; enumerated string domains
(stage, energy type, damage modifier, move kind) are encoded as Int8
codes with the mappings exported below so downstream consumers never
need to compare strings.

Designed for the Kaggle runtime (12.2 GiB RAM / 2 vCPU): single pass
over a ~2k-row CSV with Polars, output as zstd Parquet.
"""

from __future__ import annotations

import enum
from pathlib import Path
from typing import Final, NamedTuple

import polars as pl

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
RAW_CSV: Final[Path] = REPO_ROOT / "data" / "raw" / "EN_Card_Data.csv"
PROCESSED_DIR: Final[Path] = REPO_ROOT / "data" / "processed"

DIM_CARD_PARQUET: Final[Path] = PROCESSED_DIR / "dim_card.parquet"
DIM_ATTACK_PARQUET: Final[Path] = PROCESSED_DIR / "dim_attack.parquet"
DIM_SKILL_PARQUET: Final[Path] = PROCESSED_DIR / "dim_skill.parquet"
BRIDGE_ENERGY_PARQUET: Final[Path] = PROCESSED_DIR / "bridge_attack_energy.parquet"

# --------------------------------------------------------------------------- #
# Raw CSV schema (17 columns, CARD x MOVE grain)
# --------------------------------------------------------------------------- #

CSV_SCHEMA: Final[dict[str, pl.DataType]] = {
    "Card ID": pl.UInt16(),
    "Card Name": pl.Utf8(),
    "Expansion": pl.Utf8(),
    "Collection No.": pl.Utf8(),
    "Stage (Pokémon)/Type (Energy and Trainer)": pl.Utf8(),
    "Rule": pl.Utf8(),
    "Category": pl.Utf8(),
    "Previous stage": pl.Utf8(),
    "HP": pl.UInt16(),
    "Type": pl.Utf8(),
    "Weakness": pl.Utf8(),
    "Resistance (Type)": pl.Utf8(),
    "Retreat": pl.UInt8(),
    "Move Name": pl.Utf8(),
    "Cost": pl.Utf8(),
    "Damage": pl.Utf8(),
    "Effect Explanation": pl.Utf8(),
}

NULL_VALUES: Final[list[str]] = ["n/a", "N/A", ""]

# --------------------------------------------------------------------------- #
# Enumerated domains (Int8 codes)
# --------------------------------------------------------------------------- #


class EnergyType(enum.IntEnum):
    """Energy color codes — IDENTICAL to the engine's cg.api.EnergyType.

    Sharing the engine namespace means bridge_attack_energy codes compare
    directly against Pokemon.energies in observations. `●` in Cost means
    Colorless; 竜 (JP) means Dragon; {A} is "every type" (rainbow).
    """

    COLORLESS = 0    # {C} / ●
    GRASS = 1        # {G}
    FIRE = 2         # {R}
    WATER = 3        # {W}
    LIGHTNING = 4    # {L}
    PSYCHIC = 5      # {P}
    FIGHTING = 6     # {F}
    DARKNESS = 7     # {D}
    METAL = 8        # {M}
    DRAGON = 9       # {N} / 竜
    RAINBOW = 10     # {A} (provides every type)
    TEAM_ROCKET = 11 # {Team Rocket} (PSYCHIC and DARKNESS)


ENERGY_TOKEN_TO_CODE: Final[dict[str, int]] = {
    "{C}": EnergyType.COLORLESS,
    "●": EnergyType.COLORLESS,
    "{G}": EnergyType.GRASS,
    "{R}": EnergyType.FIRE,
    "{W}": EnergyType.WATER,
    "{L}": EnergyType.LIGHTNING,
    "{P}": EnergyType.PSYCHIC,
    "{F}": EnergyType.FIGHTING,
    "{D}": EnergyType.DARKNESS,
    "{M}": EnergyType.METAL,
    "{N}": EnergyType.DRAGON,
    "竜": EnergyType.DRAGON,
    "{A}": EnergyType.RAINBOW,
    "{Team Rocket}": EnergyType.TEAM_ROCKET,
}

# Matches every energy token we know how to encode.
_ENERGY_TOKEN_RE: Final[str] = r"\{[GRWLPFDMCNA]\}|\{Team Rocket\}|●|竜"


class Stage(enum.IntEnum):
    """Card stage/type codes covering Pokémon, Trainer and Energy cards."""

    BASIC_ENERGY = 1
    SPECIAL_ENERGY = 2
    ITEM = 3
    POKEMON_TOOL = 4
    SUPPORTER = 5
    STADIUM = 6
    BASIC_POKEMON = 7
    STAGE_1_POKEMON = 8
    STAGE_2_POKEMON = 9


STAGE_LABEL_TO_CODE: Final[dict[str, int]] = {
    "Basic Energy": Stage.BASIC_ENERGY,
    "Special Energy": Stage.SPECIAL_ENERGY,
    "Item": Stage.ITEM,
    "Pokémon Tool": Stage.POKEMON_TOOL,
    "Supporter": Stage.SUPPORTER,
    "Stadium": Stage.STADIUM,
    "Basic Pokémon": Stage.BASIC_POKEMON,
    "Stage 1 Pokémon": Stage.STAGE_1_POKEMON,
    "Stage 2 Pokémon": Stage.STAGE_2_POKEMON,
}

POKEMON_STAGE_CODES: Final[frozenset[int]] = frozenset(
    {Stage.BASIC_POKEMON, Stage.STAGE_1_POKEMON, Stage.STAGE_2_POKEMON}
)


class DamageModifier(enum.IntEnum):
    """Suffix/prefix on printed damage: 30x, 50+, -120."""

    NONE = 0
    MULTIPLY = 1  # trailing ×
    PLUS = 2      # trailing +
    MINUS = 3     # leading -


class MoveKind(enum.IntEnum):
    """Classification of a raw move-grain CSV row."""

    ATTACK = 0    # real attack -> dim_attack
    ABILITY = 1   # "[Ability] ..." rows -> dim_skill
    TERA = 2      # "[Tera]" marker rows -> dropped (captured as dim_card.is_tera)
    EXCLUDED = 3  # rows the engine has no id for -> dropped (see FORCE_* below)


class CardModel(NamedTuple):
    """The in-memory star schema."""

    dim_card: pl.DataFrame
    dim_attack: pl.DataFrame
    dim_skill: pl.DataFrame
    bridge_attack_energy: pl.DataFrame


# --------------------------------------------------------------------------- #
# Pipeline stages
# --------------------------------------------------------------------------- #


def load_raw(csv_path: Path = RAW_CSV) -> pl.DataFrame:
    """Read the raw CSV with an explicit schema, normalizing n/a and empty to null."""
    return pl.read_csv(
        csv_path,
        schema=CSV_SCHEMA,
        null_values=NULL_VALUES,
        encoding="utf8",
    ).rename(
        {
            "Card ID": "card_id",
            "Card Name": "card_name",
            "Expansion": "expansion",
            "Collection No.": "collection_no",
            "Stage (Pokémon)/Type (Energy and Trainer)": "stage",
            "Rule": "rule",
            "Category": "category",
            "Previous stage": "previous_stage",
            "HP": "hp",
            "Type": "type",
            "Weakness": "weakness",
            "Resistance (Type)": "resistance",
            "Retreat": "retreat_cost",
            "Move Name": "move_name",
            "Cost": "cost",
            "Damage": "damage",
            "Effect Explanation": "effect",
        }
    )


def _first_energy_code(col: str) -> pl.Expr:
    """Encode the first recognized energy token of a column as Int8 (null if none)."""
    return (
        pl.col(col)
        .str.extract(_ENERGY_TOKEN_RE, 0)
        .replace_strict(
            {token: int(code) for token, code in ENERGY_TOKEN_TO_CODE.items()},
            default=None,
            return_dtype=pl.Int8,
        )
    )


def build_dim_card(raw: pl.DataFrame) -> pl.DataFrame:
    """Collapse CARD x MOVE rows into one row per Card ID."""
    return (
        raw.group_by("card_id", maintain_order=True)
        .first()
        .with_columns(
            pl.col("stage")
            .replace_strict(
                {label: int(code) for label, code in STAGE_LABEL_TO_CODE.items()},
                default=None,
                return_dtype=pl.Int8,
            )
            .alias("stage_code"),
            _first_energy_code("type").alias("type_code"),
            _first_energy_code("weakness").alias("weakness_code"),
            _first_energy_code("resistance").alias("resistance_code"),
            pl.col("rule").eq("Pokémon ex").fill_null(False).alias("is_ex"),
            pl.col("rule").eq("Mega Pokémon ex").fill_null(False).alias("is_mega_ex"),
            pl.col("rule").eq("ACE SPEC").fill_null(False).alias("is_ace_spec"),
            # Category "Tera(<type>)" marks Tera Pokémon; verified 1:1 against
            # the engine's CardData.tera flag by reconcile.py.
            pl.col("category")
            .str.starts_with("Tera")
            .fill_null(False)
            .alias("is_tera"),
            pl.col("hp").cast(pl.Int16),
            pl.col("retreat_cost").cast(pl.Int8),
        )
        .select(
            "card_id",
            "card_name",
            "expansion",
            "collection_no",
            "stage_code",
            "category",
            "previous_stage",
            "hp",
            "type_code",
            "weakness_code",
            "resistance_code",
            "retreat_cost",
            "is_ex",
            "is_mega_ex",
            "is_ace_spec",
            "is_tera",
        )
        .sort("card_id")
    )


# Rows the engine numbers differently than a plain reading of the CSV.
# Cards 223/957 print their Tera marker WITHOUT brackets ("Tera"), so the
# bracket heuristic below misses them. Card 979 (Koraidon ex) is the
# inverse quirk: the engine numbers its "Tera" row as a real attack
# (engine id 1408) but has no id at all for "Impact Blow" — that attack
# is unplayable in the engine build, so we drop the row entirely.
# reconcile.py fails loudly if an engine update ever changes this list.
FORCE_TERA_ROWS: Final[frozenset[tuple[int, str]]] = frozenset(
    {(223, "Tera"), (957, "Tera")}
)
FORCE_EXCLUDED_ROWS: Final[frozenset[tuple[int, str]]] = frozenset(
    {(979, "Impact Blow")}
)


def _row_in(rows: frozenset[tuple[int, str]]) -> pl.Expr:
    return pl.any_horizontal(
        (pl.col("card_id") == card_id) & (pl.col("move_name") == move_name)
        for card_id, move_name in sorted(rows)
    )


def _classify_moves(raw: pl.DataFrame) -> pl.DataFrame:
    """All move-grain rows (CSV order) with a MoveKind code attached."""
    return (
        raw.filter(pl.col("move_name").is_not_null())
        .sort("card_id", maintain_order=True)
        .with_columns(
            pl.when(_row_in(FORCE_EXCLUDED_ROWS))
            .then(pl.lit(int(MoveKind.EXCLUDED), dtype=pl.Int8))
            .when(pl.col("move_name").str.starts_with("[Ability]"))
            .then(pl.lit(int(MoveKind.ABILITY), dtype=pl.Int8))
            .when(pl.col("move_name").str.starts_with("[") | _row_in(FORCE_TERA_ROWS))
            .then(pl.lit(int(MoveKind.TERA), dtype=pl.Int8))
            .otherwise(pl.lit(int(MoveKind.ATTACK), dtype=pl.Int8))
            .alias("kind_code"),
        )
    )


def build_dim_attack(moves: pl.DataFrame) -> pl.DataFrame:
    """One row per real attack.

    attack_id is sequential over ATTACK rows only, in CSV (card_id) order,
    which matches the engine's cg.api.all_attack() numbering: the engine
    treats [Ability]/[Tera] rows as skills, not attacks. reconcile.py
    verifies this alignment id-by-id against the engine.
    """
    return (
        moves.filter(pl.col("kind_code") == int(MoveKind.ATTACK))
        .with_row_index("attack_id", offset=1)
        .with_columns(
            pl.col("attack_id").cast(pl.UInt16),
            pl.col("damage").str.extract(r"(\d+)", 1).cast(pl.Int16).alias("damage_base"),
            pl.when(pl.col("damage").str.contains("×"))
            .then(pl.lit(int(DamageModifier.MULTIPLY), dtype=pl.Int8))
            .when(pl.col("damage").str.contains(r"\+"))
            .then(pl.lit(int(DamageModifier.PLUS), dtype=pl.Int8))
            .when(pl.col("damage").str.starts_with("-"))
            .then(pl.lit(int(DamageModifier.MINUS), dtype=pl.Int8))
            .when(pl.col("damage").is_not_null())
            .then(pl.lit(int(DamageModifier.NONE), dtype=pl.Int8))
            .otherwise(None)
            .alias("damage_modifier_code"),
            # "No cost" and status-only attacks both yield 0 matched tokens.
            pl.col("cost")
            .str.count_matches(_ENERGY_TOKEN_RE)
            .fill_null(0)
            .cast(pl.Int8)
            .alias("cost_total"),
        )
        .select(
            "attack_id",
            "card_id",
            "move_name",
            "damage_base",
            "damage_modifier_code",
            "cost_total",
            "cost",
            "effect",
        )
    )


def build_dim_skill(moves: pl.DataFrame) -> pl.DataFrame:
    """One row per Pokémon ability (a subset of the engine's CardData.skills).

    [Tera] rows are NOT skills: they mark the card as a Tera Pokémon
    (dim_card.is_tera) whose printed type differs and which takes no attack
    damage while on the Bench (semantics verified in tests/test_tera_bench_immunity.py).
    """
    return (
        moves.filter(pl.col("kind_code") == int(MoveKind.ABILITY))
        .with_row_index("skill_id", offset=1)
        .with_columns(
            pl.col("skill_id").cast(pl.UInt16),
            pl.col("move_name")
            .str.strip_prefix("[Ability] ")
            .alias("skill_name"),
        )
        .select("skill_id", "card_id", "skill_name", "effect")
    )


def build_bridge_attack_energy(dim_attack: pl.DataFrame) -> pl.DataFrame:
    """Explode attack costs into (attack_id, energy_type_code, qty) rows."""
    return (
        dim_attack.select("attack_id", "card_id", "cost")
        .filter(pl.col("cost").is_not_null())
        .with_columns(pl.col("cost").str.extract_all(_ENERGY_TOKEN_RE).alias("token"))
        .explode("token")
        .filter(pl.col("token").is_not_null())
        .with_columns(
            pl.col("token")
            .replace_strict(
                {token: int(code) for token, code in ENERGY_TOKEN_TO_CODE.items()},
                return_dtype=pl.Int8,
            )
            .alias("energy_type_code")
        )
        .group_by("attack_id", "card_id", "energy_type_code", maintain_order=True)
        .agg(pl.len().cast(pl.Int8).alias("qty"))
        .sort("attack_id", "energy_type_code")
    )


def build_star_schema(csv_path: Path = RAW_CSV) -> CardModel:
    """Run the full in-memory pipeline: raw CSV -> star schema."""
    raw = load_raw(csv_path)
    moves = _classify_moves(raw)
    dim_card = build_dim_card(raw)
    dim_attack = build_dim_attack(moves)
    dim_skill = build_dim_skill(moves)
    bridge = build_bridge_attack_energy(dim_attack)
    # The raw cost string was only needed to derive the bridge table.
    dim_attack = dim_attack.drop("cost")
    return CardModel(dim_card, dim_attack, dim_skill, bridge)


def persist(model: CardModel, out_dir: Path = PROCESSED_DIR) -> None:
    """Write each table as zstd-compressed Parquet."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model.dim_card.write_parquet(out_dir / DIM_CARD_PARQUET.name, compression="zstd")
    model.dim_attack.write_parquet(out_dir / DIM_ATTACK_PARQUET.name, compression="zstd")
    model.dim_skill.write_parquet(out_dir / DIM_SKILL_PARQUET.name, compression="zstd")
    model.bridge_attack_energy.write_parquet(
        out_dir / BRIDGE_ENERGY_PARQUET.name, compression="zstd"
    )


def main() -> None:
    model = build_star_schema()
    persist(model)
    print(f"dim_card:             {model.dim_card.height:>5} cards")
    print(f"dim_attack:           {model.dim_attack.height:>5} attacks (engine-aligned ids)")
    print(f"dim_skill:            {model.dim_skill.height:>5} abilities")
    print(f"bridge_attack_energy: {model.bridge_attack_energy.height:>5} cost rows")
    for path in (
        DIM_CARD_PARQUET,
        DIM_ATTACK_PARQUET,
        DIM_SKILL_PARQUET,
        BRIDGE_ENERGY_PARQUET,
    ):
        print(f"wrote {path.relative_to(REPO_ROOT)} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
