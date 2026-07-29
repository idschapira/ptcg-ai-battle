"""Audit src/deckbuilding/legality.py against the engine's own verdict.

legality.py encodes deck-construction rules by hand. The engine is the
only authority on which of them are real, and we have already been
wrong twice in the permissive direction — rejecting decks the engine
happily starts (lineless ability techs, Team Rocket energy coverage),
both found by accident on real ladder lists. This looks for the rest on
purpose.

Method: for every decklist we hold, ask both oracles.

    legality.validate_deck   our rules
    cg.game.battle_start     the engine, mirror-started against itself

The engine's verdict is definitive in one direction only: a deck it
STARTS is legal, full stop. A deck it refuses may be refused for a
reason unrelated to construction, so those are reported as "engine
refused" rather than counted as agreement.

Two failure modes, and they are not symmetric:

    FALSE REJECT   we say illegal, the engine starts it. This one costs
                   real money — it silently removes candidates from
                   every deck search we run.
    FALSE ACCEPT   we say legal, the engine refuses. Cheap: the deck
                   fails loudly the first time it is used.

Run from the repo root:
    python -m src.analysis.legality_audit
    python -m src.analysis.legality_audit --dir data/decks
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

from ..deckbuilding.legality import read_deck_ids, validate_deck
from ..ingestion.build_card_model import REPO_ROOT
from ..ingestion.card_index import CardIndex

DECKS_DIR: Final[Path] = REPO_ROOT / "data" / "decks"


def engine_accepts(deck: list[int]) -> tuple[bool, str]:
    """Does the engine start a mirror game with this list?"""
    from cg import game as cg_game

    try:
        obs_dict, start = cg_game.battle_start(list(deck), list(deck))
    except Exception as exc:  # noqa: BLE001 — a raise is a refusal
        return False, f"{type(exc).__name__}: {exc}"
    try:
        if obs_dict is None:
            return False, str(getattr(start, "errorType", "unknown"))
        return True, ""
    finally:
        cg_game.battle_finish()


def audit(paths: list[Path], index: CardIndex) -> list[dict]:
    rows = []
    for path in paths:
        ids = read_deck_ids(path)
        report = validate_deck(ids, index)
        accepted, why = engine_accepts(ids)
        rows.append({
            "deck": path.name,
            "cards": len(ids),
            "ours_ok": report.ok,
            "engine_ok": accepted,
            "engine_why": why,
            "errors": list(report.errors),
        })
    return rows


# --------------------------------------------------------------------------- #
# Rule probes
# --------------------------------------------------------------------------- #
#
# Auditing only the decks we HOLD is survivorship bias: they all passed
# our rules already, so agreement proves nothing about the rules we may
# be over-enforcing. These probes go the other way — deliberately break
# ONE rule at a time on a known-good list and ask the engine whether it
# cares. A probe the engine starts is a rule we enforce and it does not.


def _probe_extra_copy(deck: list[int], index: CardIndex) -> list[int] | None:
    """A 5th copy of a non-energy card (breaks the 4-copy rule)."""
    from collections import Counter
    counts = Counter(deck)
    for card_id, n in counts.items():
        card = index.get_card(card_id)
        if card is None or card.stage_code in (1, 2):
            continue          # basic/special energy: copy limits differ
        if n >= 4:
            out = list(deck)
            out.remove(next(c for c in out if c != card_id))
            out.append(card_id)
            return out
    return None


def _probe_broken_evolution(deck: list[int],
                            index: CardIndex) -> list[int] | None:
    """Drop every copy of a Stage-1's pre-evolution, keep the Stage 1."""
    names = {}
    for card_id in deck:
        card = index.get_card(card_id)
        if card is not None:
            names.setdefault(card.card_name, card)
    for card in names.values():
        if card.stage_code != 8:            # Stage 1
            continue
        pre = card.previous_stage
        if not pre:
            continue
        victims = [c for c in deck
                   if (index.get_card(c) or card).card_name == pre]
        if not victims or len(victims) >= len(deck):
            continue
        out = [c for c in deck if c not in set(victims)]
        filler = next((c for c in deck
                       if (index.get_card(c) or card).stage_code == 1), None)
        if filler is None:
            continue
        out.extend([filler] * (len(deck) - len(out)))
        return out
    return None


def _probe_no_basic_pokemon(deck: list[int],
                            index: CardIndex) -> list[int] | None:
    """Replace every Basic Pokémon with basic energy."""
    filler = next((c for c in deck
                   if (index.get_card(c) is not None
                       and index.get_card(c).stage_code == 1)), None)
    if filler is None:
        return None
    out = []
    replaced = 0
    for card_id in deck:
        card = index.get_card(card_id)
        if card is not None and card.stage_code == 7:
            out.append(filler)
            replaced += 1
        else:
            out.append(card_id)
    return out if replaced else None


PROBES: Final[tuple[tuple[str, str, object], ...]] = (
    ("5th copy of a card", "at most 4 copies per name", _probe_extra_copy),
    ("Stage 1 with no pre-evolution", "evolution lines are coherent",
     _probe_broken_evolution),
    ("no Basic Pokemon at all", "at least one Basic Pokemon",
     _probe_no_basic_pokemon),
)


def run_probes(base: list[int], index: CardIndex) -> None:
    print("\n== rule probes (break one rule, ask the engine) ==")
    for label, rule, build in PROBES:
        deck = build(base, index)  # type: ignore[operator]
        if deck is None or len(deck) != len(base):
            print(f"  {label:34s} could not construct a probe — SKIPPED")
            continue
        report = validate_deck(deck, index)
        accepted, why = engine_accepts(deck)
        if report.ok:
            note = "our rules did NOT reject it — probe missed"
        elif accepted:
            note = (f"FALSE REJECT: engine starts it, we block it "
                    f"({rule})")
        else:
            note = f"engine also refuses ({why or 'no reason given'})"
        print(f"  {label:34s} ours={'ok' if report.ok else 'NO':2s} "
              f"engine={'ok' if accepted else 'NO':2s}  {note}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true",
                        help="also break each rule deliberately and ask "
                             "the engine whether it enforces it")
    parser.add_argument("--dir", type=Path, default=DECKS_DIR)
    parser.add_argument("--extra", type=Path, nargs="*", default=None,
                        help="additional decklists (e.g. deck.csv)")
    args = parser.parse_args()

    paths = sorted(args.dir.glob("*.csv")) if args.dir.exists() else []
    paths += [p for p in (args.extra or []) if p.exists()]
    if not paths:
        raise SystemExit(f"no decklists under {args.dir}")

    index = CardIndex()
    rows = audit(paths, index)

    print(f"\n== legality audit: {len(rows)} decklists ==")
    print(f"  {'deck':38s} {'n':>3s} {'ours':>6s} {'engine':>7s}  verdict")
    false_rejects, false_accepts = [], []
    for row in rows:
        if row["ours_ok"] and row["engine_ok"]:
            tag = "agree (legal)"
        elif not row["ours_ok"] and not row["engine_ok"]:
            tag = "agree (rejected)"
        elif not row["ours_ok"] and row["engine_ok"]:
            tag = "FALSE REJECT — we block a deck the engine starts"
            false_rejects.append(row)
        else:
            tag = "FALSE ACCEPT — engine refused"
            false_accepts.append(row)
        print(f"  {row['deck']:38s} {row['cards']:3d} "
              f"{'ok' if row['ours_ok'] else 'NO':>6s} "
              f"{'ok' if row['engine_ok'] else 'NO':>7s}  {tag}")

    if false_rejects:
        print(f"\n  {len(false_rejects)} FALSE REJECT(S) — rules to revisit:")
        for row in false_rejects:
            print(f"    {row['deck']}")
            for error in row["errors"]:
                print(f"      - {error}")
    else:
        print("\n  no false rejects: every deck our rules block, the engine "
              "also refuses")
    if false_accepts:
        print(f"\n  {len(false_accepts)} deck(s) we pass and the engine "
              f"refused:")
        for row in false_accepts:
            print(f"    {row['deck']}: {row['engine_why']}")
    print("\n  NOTE: every list above already passed our rules once, so "
          "agreement here is\n  survivorship bias, not evidence. Use "
          "--probe for the real test.")

    if args.probe:
        base = read_deck_ids(REPO_ROOT / "deck.csv")
        run_probes(base, index)


if __name__ == "__main__":
    main()
