"""Per-turn accuracy of the runtime opponent-deck estimator, on real games.

The estimator (src/rl_models/opponent_estimator.py) is the thing that
lets the submission search at all, so its error rate is the honest cap
on how much the search can be trusted. This measures it the only way
that counts: by replaying REAL ladder games observation by observation,
from each seat's own filtered view, and asking what the estimator would
have believed at every turn.

Ground truth is the label the SAME rules give the player's fully
observed deck at the end of the replay. That is a ceiling, not the true
decklist — a card never drawn is never seen, so a deck can be labelled
"unknown" here while its owner knows perfectly well what it is. Read the
numbers as "does the estimator converge to what the whole game reveals",
not "does it read the opponent's mind".

Three quantities per turn bucket, and they answer different questions:

    label accuracy   of all seats, how often the current label already
                     matches the end-of-game label (includes the seats
                     where we abstain — this is the raw signal)
    precision        of the calls where the estimator is CONFIDENT, how
                     often it is right. This is the number that gates
                     the search: a wrong confident call feeds a wrong
                     deck to search_begin.
    coverage         what share of decisions are confident at all — how
                     much of the game the search is even eligible for.

Run from the repo root:
    python -m src.analysis.estimator_accuracy
    python -m src.analysis.estimator_accuracy --limit 200 --sweep
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Final, Iterable

from ..deckbuilding.archetype_rules import UNKNOWN, label_archetype
from ..ingestion.build_card_model import REPO_ROOT
from ..ingestion.card_index import CardIndex
from ..ingestion.replays_download import REPLAYS_DIR
from ..rl_models.opponent_estimator import (DEFAULT_MIN_CONTAINMENT,
                                            DEFAULT_MIN_OBSERVED,
                                            OpponentDeckEstimator)
from .meta_radar import _DAY_DIR, observed_serials

logger = logging.getLogger(__name__)

TURN_BUCKETS: Final[tuple[tuple[str, int, int], ...]] = (
    ("t1-2", 1, 2), ("t3-4", 3, 4), ("t5-6", 5, 6), ("t7-9", 7, 9),
    ("t10-14", 10, 14), ("t15+", 15, 10_000),
)


def _bucket(turn: int) -> str:
    for name, low, high in TURN_BUCKETS:
        if low <= turn <= high:
            return name
    return "t1-2" if turn < 1 else "t15+"


def _truth_labels(replay: dict, index: CardIndex) -> dict[int, str]:
    """player index -> label of everything the whole replay revealed."""
    seen = observed_serials(replay)
    names: dict[int, list[str]] = {0: [], 1: []}
    for (player, _serial), card_id in seen.items():
        card = index.get_card(card_id)
        if card is not None and player in names:
            names[player].append(card.card_name)
    return {p: label_archetype(n) for p, n in names.items()}


def _iter_replays(corpus: Path, limit: int | None) -> Iterable[dict]:
    n = 0
    if not corpus.exists():
        return
    for day_dir in sorted(corpus.iterdir()):
        if not day_dir.is_dir() or not _DAY_DIR.match(day_dir.name):
            continue
        for path in sorted(day_dir.glob("*.json")):
            if limit is not None and n >= limit:
                return
            try:
                with open(path, encoding="utf-8") as fh:
                    yield json.load(fh)
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("unreadable replay %s: %s", path.name, exc)
                continue
            n += 1


class _Tally:
    """Per-bucket counters for one (min_observed, min_containment) setting."""

    def __init__(self) -> None:
        self.seen: Counter = Counter()
        self.label_ok: Counter = Counter()
        self.confident: Counter = Counter()
        self.confident_ok: Counter = Counter()
        self.wrong_confident: Counter = Counter()
        self.containment_when_wrong: list[float] = []

    def add(self, bucket: str, predicted: str, truth: str,
            confident: bool, containment: float) -> None:
        self.seen[bucket] += 1
        correct = predicted == truth
        self.label_ok[bucket] += int(correct)
        if confident:
            self.confident[bucket] += 1
            self.confident_ok[bucket] += int(correct)
            if not correct:
                self.wrong_confident[f"{truth} -> {predicted}"] += 1
                self.containment_when_wrong.append(containment)

    def totals(self) -> tuple[int, int, int, int]:
        return (sum(self.seen.values()), sum(self.label_ok.values()),
                sum(self.confident.values()),
                sum(self.confident_ok.values()))


def measure(corpus: Path, index: CardIndex, limit: int | None,
            min_observed: int, min_containment: float,
            include_unknown_truth: bool) -> _Tally:
    """Replay the corpus through the estimator at one threshold setting."""
    tally = _Tally()
    for replay in _iter_replays(corpus, limit):
        truth = _truth_labels(replay, index)
        estimators = {
            seat: OpponentDeckEstimator(index=index,
                                        min_observed=min_observed,
                                        min_containment=min_containment)
            for seat in (0, 1)
        }
        for step in replay.get("steps") or []:
            if not isinstance(step, list):
                continue
            for seat, entry in enumerate(step):
                if not isinstance(entry, dict) or seat not in estimators:
                    continue
                obs = entry.get("observation")
                if not isinstance(obs, dict):
                    continue
                state = obs.get("current")
                if not isinstance(state, dict):
                    continue
                # seat's own view -> what it can infer about the OTHER seat
                opponent = 1 - seat
                target = truth.get(opponent, UNKNOWN)
                if target == UNKNOWN and not include_unknown_truth:
                    continue
                estimate = estimators[seat].observe(obs)
                turn = state.get("turn")
                tally.add(_bucket(turn if isinstance(turn, int) else 1),
                          estimate.archetype, target, estimate.usable,
                          estimate.containment)
    return tally


def report(tally: _Tally, title: str) -> None:
    print(f"\n== {title} ==")
    print(f"  {'bucket':8s} {'n':>7s} {'label acc':>10s} "
          f"{'coverage':>9s} {'precision':>10s}")
    for name, _lo, _hi in TURN_BUCKETS:
        n = tally.seen[name]
        if not n:
            continue
        conf = tally.confident[name]
        acc = tally.label_ok[name] / n
        cov = conf / n
        prec = (tally.confident_ok[name] / conf) if conf else float("nan")
        prec_s = "     n/a" if conf == 0 else f"{prec:9.1%}"
        print(f"  {name:8s} {n:7d} {acc:9.1%} {cov:8.1%} {prec_s}")
    total, label_ok, conf, conf_ok = tally.totals()
    if total:
        prec = (conf_ok / conf) if conf else float("nan")
        print(f"  {'TOTAL':8s} {total:7d} {label_ok / total:9.1%} "
              f"{conf / total:8.1%} "
              f"{'     n/a' if not conf else f'{prec:9.1%}'}")
    if tally.wrong_confident:
        print("  confident MISLABELS (truth -> predicted):")
        for pair, n in tally.wrong_confident.most_common(8):
            print(f"    {n:6d}  {pair}")
    else:
        print("  confident mislabels: none")


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=REPLAYS_DIR)
    parser.add_argument("--limit", type=int, default=None,
                        help="max replays (default: the whole corpus)")
    parser.add_argument("--min-observed", type=int,
                        default=DEFAULT_MIN_OBSERVED)
    parser.add_argument("--min-containment", type=float,
                        default=DEFAULT_MIN_CONTAINMENT)
    parser.add_argument("--include-unknown-truth", action="store_true",
                        help="also score seats whose end-of-game label is "
                             "unknown (the estimator should abstain there)")
    parser.add_argument("--sweep", action="store_true",
                        help="scan the threshold grid instead of one point")
    args = parser.parse_args()

    index = CardIndex()
    if not args.sweep:
        tally = measure(args.corpus, index, args.limit, args.min_observed,
                        args.min_containment, args.include_unknown_truth)
        report(tally, f"min_observed={args.min_observed} "
                      f"min_containment={args.min_containment}")
        return

    grid = [(n, c) for n in (4, 6, 8, 12, 16) for c in (0.6, 0.75, 0.85, 0.95)]
    rows = []
    for n_obs, cont in grid:
        tally = measure(args.corpus, index, args.limit, n_obs, cont,
                        args.include_unknown_truth)
        total, _label_ok, conf, conf_ok = tally.totals()
        rows.append((n_obs, cont, total, conf, conf_ok))
    print(f"\n== threshold sweep ({rows[0][2]} decisions scored) ==")
    print(f"  {'min_obs':>7s} {'min_cont':>9s} {'coverage':>9s} "
          f"{'precision':>10s}")
    for n_obs, cont, total, conf, conf_ok in rows:
        cov = conf / total if total else 0.0
        prec = (conf_ok / conf) if conf else float("nan")
        prec_s = "     n/a" if not conf else f"{prec:9.1%}"
        print(f"  {n_obs:7d} {cont:9.2f} {cov:8.1%} {prec_s}")


if __name__ == "__main__":
    main()
