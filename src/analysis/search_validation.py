"""High-N validation of the runtime search agent, sharded over processes.

Searching costs seconds per decision, so the sample sizes this project
demands (a ~150-game A/B on this engine is noise: the seats cannot be
paired, and the measured spread at that N is ~10pp) are unaffordable
single-threaded. This shards the games across processes and pools them.

Processes, not threads, and not by preference: cg.api keeps ONE
module-level ``agent_ptr`` and ``search_end()`` frees every search
allocated against it, with no lock anywhere. Two threads searching at
once would corrupt each other's state. Separate processes each load
their own copy of the engine binary and are genuinely independent.

Pooling is legitimate here because shards differ only in their seeds:
the engine reseeds every game from ``std::random_device`` regardless
(see the ab_test module docstring), so games are i.i.d. across shards
and the pooled count is a plain binomial. Seats alternate inside each
shard, as always.

Every number printed is a pooled Wilson 95% interval, never a bare
winrate, and episode wall time is reported against the 600s bank with
the Kaggle slowdown projection beside it.

Run from the repo root:
    python -m src.analysis.search_validation --preset effect --games 600
    python -m src.analysis.search_validation --preset field --games 300
    python -m src.analysis.search_validation --preset floor --games 400
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Final

from ..environment_wrapper.ab_test import (BANK_S, KAGGLE_SLOWDOWN,
                                           ArmMetrics, ArmSpec, arm_factory,
                                           binomial_p_value, verdict,
                                           wilson_interval)
from ..ingestion.build_card_model import REPO_ROOT

DECKS: Final[Path] = REPO_ROOT / "data" / "decks"
OUT_DIR: Final[Path] = REPO_ROOT / "data" / "processed" / "search_validation"

# (label, deck A @ arm A, deck B @ arm B). Presets encode the three
# questions the checkpoint has to answer, so nobody has to remember the
# right command line.
PRESETS: Final[dict[str, tuple[tuple[str, str, str], ...]]] = {
    # (a) what does the SEARCH itself buy? Same deck both sides, so the
    # only difference is the search; the estimator is trivially correct
    # in the mirror, which isolates search from estimation.
    "effect": (
        ("search vs prior (mirror)",
         "deck.csv@search-crustle", "deck.csv@crustle-v3"),
    ),
    # (a2) the DIAGNOSTIC for (a)'s negative result: same comparison,
    # but the rollouts model the opponent with the pilot that matches
    # the estimated archetype instead of a generic heuristic. If the
    # sign flips here, the opponent MODEL was the defect, not the search.
    "effect-match": (
        ("search(match) vs prior (mirror)",
         "deck.csv@search-crustle-match", "deck.csv@crustle-v3"),
    ),
    # (a3) the OTHER candidate cause: the optimizer's curse. At 4x4 each
    # candidate is scored on four Bernoulli rollouts, so the argmax over
    # four such estimates largely selects whichever got lucky. Requiring
    # a margin over the PRIOR'S choice before overriding it tests that
    # directly, and "-both" tests the two fixes together.
    "effect-margin": (
        ("search(margin) vs prior (mirror)",
         "deck.csv@search-crustle-margin", "deck.csv@crustle-v3"),
    ),
    "effect-both": (
        ("search(match+margin) vs prior (mirror)",
         "deck.csv@search-crustle-both", "deck.csv@crustle-v3"),
    ),
    # THE DECISIVE TEST. The +14.7pp the search showed against
    # Grimmsnarl was measured against a HeuristicAgent — which is
    # EXACTLY the model its own rollouts use. That hands the search a
    # perfect opponent model for free, and the real ladder never will.
    # These cells replay the same matchup against pilots the rollouts do
    # NOT model, so the question "does the gain survive a wrong opponent
    # model" gets a number instead of an argument.
    #
    # Both arms of each pair face the SAME opponent, so prior-vs-search
    # stays the comparison; only the opponent's pilot changes.
    "wrongmodel-bc": (
        ("prior vs BC-Grimmsnarl",
         "deck.csv@crustle-v3",
         "data/decks/meta_grimmsnarl.csv@network,models/bc_grimmsnarl.npz,"
         "models/feature_stats.npz"),
        ("search vs BC-Grimmsnarl",
         "deck.csv@search-crustle",
         "data/decks/meta_grimmsnarl.csv@network,models/bc_grimmsnarl.npz,"
         "models/feature_stats.npz"),
    ),
    "wrongmodel-module": (
        ("prior vs Grimmsnarl module",
         "deck.csv@crustle-v3",
         "data/decks/meta_grimmsnarl.csv@grimmsnarl-module"),
        ("search vs Grimmsnarl module",
         "deck.csv@search-crustle",
         "data/decks/meta_grimmsnarl.csv@grimmsnarl-module"),
    ),
    # THE HONEST NUMBER — "faithful but not exact".
    #
    # Everything so far compared the search against opponents that were
    # either its own rollout model (upper bound, +14.7pp) or nothing like
    # it (lower bound, -2.5pp). The ladder is the analogue of neither: a
    # real opponent is a competent human playing a known archetype, and
    # the best model we can build of them is a clone of a DIFFERENT
    # competent human playing the SAME archetype.
    #
    # Majkel1337 and Yushin Ito both play Alakazam box exclusively (567
    # and 440 games in the corpus), so cloning each gives exactly that
    # pair. All four cells face the SAME opponent (BC-Majkel) and differ
    # only in what models it inside the rollouts — so this is one
    # fidelity ladder, not four experiments:
    #
    #   prior         no search at all                  (baseline)
    #   BC-Majkel     the opponent's OWN clone          (exact, ceiling)
    #   BC-Yushin     another human, same archetype     (THE HONEST ONE)
    #   heuristic     nothing in common                 (divergent, floor)
    "fidelity-ladder": (
        ("prior vs BC-Majkel (baseline)",
         "deck.csv@crustle-v3",
         "data/decks/meta_alakazam.csv@network,models/bc_majkel.npz,"
         "models/feature_stats.npz"),
        ("search[rollout=BC-Majkel, EXACT] vs BC-Majkel",
         "deck.csv@search-net,models/bc_majkel.npz",
         "data/decks/meta_alakazam.csv@network,models/bc_majkel.npz,"
         "models/feature_stats.npz"),
        ("search[rollout=BC-Yushin, FAITHFUL-NOT-EXACT] vs BC-Majkel",
         "deck.csv@search-net,models/bc_yushin.npz",
         "data/decks/meta_alakazam.csv@network,models/bc_majkel.npz,"
         "models/feature_stats.npz"),
        ("search[rollout=heuristic, DIVERGENT] vs BC-Majkel",
         "deck.csv@search-crustle",
         "data/decks/meta_alakazam.csv@network,models/bc_majkel.npz,"
         "models/feature_stats.npz"),
    ),
    # THE HONEST NUMBER, ON A CELL WITH HEADROOM.
    #
    # The first fidelity ladder was run on Alakazam and was inconclusive
    # for a reason that had nothing to do with the search: our prior
    # already beats that matchup 91.4%, so there were 8.6pp of room and
    # every cell landed inside the noise. Grimmsnarl is the cell that
    # matters (41-47% of the top of the ladder) AND the cell with room
    # (prior ~46-52%), and it has two clonable humans: Luca (158 games)
    # and Dries @ Tufa Labs (43).
    #
    # All cells face the SAME opponent, BC-Luca, and differ only in what
    # models it inside the rollouts.
    "fidelity-grimmsnarl": (
        ("prior vs BC-Luca (baseline)",
         "deck.csv@crustle-v3",
         "data/decks/meta_grimmsnarl.csv@network,models/bc_luca.npz,"
         "models/feature_stats.npz"),
        ("search[rollout=BC-Luca, EXACT] vs BC-Luca",
         "deck.csv@search-net,models/bc_luca.npz",
         "data/decks/meta_grimmsnarl.csv@network,models/bc_luca.npz,"
         "models/feature_stats.npz"),
        ("search[rollout=BC-Dries, FAITHFUL-NOT-EXACT] vs BC-Luca",
         "deck.csv@search-net,models/bc_dries.npz",
         "data/decks/meta_grimmsnarl.csv@network,models/bc_luca.npz,"
         "models/feature_stats.npz"),
        ("search[rollout=heuristic, DIVERGENT] vs BC-Luca",
         "deck.csv@search-crustle",
         "data/decks/meta_grimmsnarl.csv@network,models/bc_luca.npz,"
         "models/feature_stats.npz"),
    ),
    # the honest cell again, this time with adaptive allocation on, so
    # the effect is read at the cost that actually fits the bank
    "fidelity-adaptive": (
        ("search[BC-Yushin, adaptive] vs BC-Majkel",
         "deck.csv@search-net-adaptive,models/bc_yushin.npz",
         "data/decks/meta_alakazam.csv@network,models/bc_majkel.npz,"
         "models/feature_stats.npz"),
    ),
    # (3)+(4) adaptive allocation: does it fit the bank, and does the
    # gain survive (or grow) when the budget is concentrated?
    "adaptive-grimmsnarl": (
        ("search(adaptive) vs Grimmsnarl heuristic",
         "deck.csv@search-crustle-adaptive",
         "data/decks/meta_grimmsnarl.csv@heuristic"),
        ("search(adaptive) vs BC-Grimmsnarl",
         "deck.csv@search-crustle-adaptive",
         "data/decks/meta_grimmsnarl.csv@network,models/bc_grimmsnarl.npz,"
         "models/feature_stats.npz"),
        ("search(adaptive) vs Grimmsnarl module",
         "deck.csv@search-crustle-adaptive",
         "data/decks/meta_grimmsnarl.csv@grimmsnarl-module"),
    ),
    "adaptive-mirror": (
        ("search(adaptive) vs prior (mirror)",
         "deck.csv@search-crustle-adaptive", "deck.csv@crustle-v3"),
    ),
    # (b) no regression against the real field, each opponent flown by
    # the pilot the calibration work settled on.
    "field": (
        ("vs Alakazam (calibrated)",
         "deck.csv@search-crustle", "data/decks/meta_alakazam.csv@heuristic"),
        ("vs Grimmsnarl",
         "deck.csv@search-crustle", "data/decks/meta_grimmsnarl.csv@heuristic"),
        ("vs Spidops",
         "deck.csv@search-crustle", "data/decks/meta_spidops.csv@heuristic"),
        ("vs Starmie",
         "deck.csv@search-crustle", "data/decks/meta_starmie.csv@heuristic"),
        ("vs Crustle/Kangaskhan",
         "deck.csv@search-crustle",
         "data/decks/meta_crustle_kangaskhan.csv@heuristic"),
    ),
    # (c) the FLOOR: with the estimator blind the agent is its prior, so
    # this must land on a coin flip. Anything else means the search arm
    # is not degrading to what we think it degrades to.
    "floor": (
        ("blind search vs prior (must be ~50%)",
         "deck.csv@search-crustle-blind", "deck.csv@crustle-v3"),
    ),
    # the same field, flown by the CURRENT SHIP, so (b) has a baseline
    # measured on the same machine on the same day.
    "field-baseline": (
        ("prior vs Alakazam (calibrated)",
         "deck.csv@crustle-v3", "data/decks/meta_alakazam.csv@heuristic"),
        ("prior vs Grimmsnarl",
         "deck.csv@crustle-v3", "data/decks/meta_grimmsnarl.csv@heuristic"),
        ("prior vs Spidops",
         "deck.csv@crustle-v3", "data/decks/meta_spidops.csv@heuristic"),
        ("prior vs Starmie",
         "deck.csv@crustle-v3", "data/decks/meta_starmie.csv@heuristic"),
        ("prior vs Crustle/Kangaskhan",
         "deck.csv@crustle-v3",
         "data/decks/meta_crustle_kangaskhan.csv@heuristic"),
    ),
}


@dataclass
class ShardResult:
    """One shard's contribution — everything pools by addition."""

    a_wins: int = 0
    b_wins: int = 0
    draws: int = 0
    errors: list[str] = field(default_factory=list)
    turns: list[int] = field(default_factory=list)
    a_episode_s: list[float] = field(default_factory=list)
    b_episode_s: list[float] = field(default_factory=list)
    a_search: str = ""
    a_budget: str = ""
    a_estimator: str = ""
    searched: int = 0
    changed: int = 0
    decisions: int = 0
    search_exceptions: int = 0


def _run_shard(payload: tuple[str, str, int, int]) -> dict:
    """Worker: play ``games`` of one matchup at ``seed``. Fresh engine."""
    a_text, b_text, games, seed = payload
    # imported inside the worker so each process builds its own indexes
    from ..deckbuilding.gauntlet import run_pair
    from ..deckbuilding.legality import read_deck_ids
    from ..ingestion.build_effect_model import EffectIndex
    from ..ingestion.card_index import CardIndex

    index, effects = CardIndex(), EffectIndex()
    deck_a_text, arm_a = a_text.split("@", 1)
    deck_b_text, arm_b = b_text.split("@", 1)
    deck_a = read_deck_ids(REPO_ROOT / deck_a_text)
    deck_b = read_deck_ids(REPO_ROOT / deck_b_text)
    spec_a, spec_b = ArmSpec.parse(arm_a), ArmSpec.parse(arm_b)
    metrics_a, metrics_b = ArmMetrics(), ArmMetrics()
    pair = run_pair(arm_factory(spec_a, index, effects, metrics_a, deck_a),
                    arm_factory(spec_b, index, effects, metrics_b, deck_b),
                    deck_a, deck_b, games, seed)
    result = ShardResult(
        a_wins=pair.a_wins, b_wins=pair.b_wins, draws=pair.draws,
        errors=list(pair.errors),
        a_episode_s=list(metrics_a.episode_wall_s),
        b_episode_s=list(metrics_b.episode_wall_s),
    )
    if metrics_a.search is not None:
        result.a_search = metrics_a.search.summary()
        result.searched = metrics_a.search.searched
        result.changed = metrics_a.search.changed
        result.decisions = metrics_a.search.decisions
        result.search_exceptions = metrics_a.search.exceptions
    if metrics_a.budget is not None:
        result.a_budget = metrics_a.budget.summary()
    if metrics_a.estimator is not None:
        result.a_estimator = metrics_a.estimator.summary()
    return asdict(result)


def _pool(shards: list[dict]) -> ShardResult:
    total = ShardResult()
    for s in shards:
        total.a_wins += s["a_wins"]
        total.b_wins += s["b_wins"]
        total.draws += s["draws"]
        total.errors.extend(s["errors"])
        total.a_episode_s.extend(s["a_episode_s"])
        total.b_episode_s.extend(s["b_episode_s"])
        total.searched += s["searched"]
        total.changed += s["changed"]
        total.decisions += s["decisions"]
        total.search_exceptions += s["search_exceptions"]
        if s["a_search"] and not total.a_search:
            total.a_search = s["a_search"]
            total.a_budget = s["a_budget"]
            total.a_estimator = s["a_estimator"]
    return total


def _episode_stats(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    ordered = sorted(values)
    n = len(ordered)
    return (ordered[n // 2], ordered[min(n - 1, int(n * 0.95))], ordered[-1])


def run_matchup(label: str, a_text: str, b_text: str, games: int,
                workers: int, seed: int, bar: float) -> dict:
    per_shard = max(1, games // workers)
    payloads = [(a_text, b_text, per_shard, seed + 100_000 * i)
                for i in range(workers)]
    t0 = time.perf_counter()
    with mp.Pool(processes=workers) as pool:
        shards = pool.map(_run_shard, payloads)
    elapsed = time.perf_counter() - t0
    total = _pool(shards)
    decided = total.a_wins + total.b_wins
    lo, hi = wilson_interval(total.a_wins, decided)
    median, p95, worst = _episode_stats(total.a_episode_s)

    print(f"\n{label}")
    print(f"  A={a_text}")
    print(f"  B={b_text}")
    print(f"  n={decided + total.draws} decided={decided} "
          f"draws={total.draws}  wins A={total.a_wins} B={total.b_wins}")
    print(f"  winrate A = {total.a_wins / max(decided, 1):.1%}  "
          f"IC95 [{lo:.1%}, {hi:.1%}]  "
          f"p={binomial_p_value(total.a_wins, decided, bar):.4f}  "
          f"VERDICT vs {bar:.0%}: {verdict(total.a_wins, decided, bar)}")
    print(f"  exceptions {len(total.errors)} (must be 0)")
    for error in total.errors[:3]:
        print(f"    {error}")
    if total.a_episode_s:
        print(f"  A episode wall  median {median:.1f}s  p95 {p95:.1f}s  "
              f"max {worst:.1f}s  |  x{KAGGLE_SLOWDOWN:.0f}: max "
              f"{worst * KAGGLE_SLOWDOWN:.0f}s = "
              f"{worst * KAGGLE_SLOWDOWN / BANK_S:.0%} of the bank")
    if total.a_search:
        print(f"  A search    {total.a_search}")
        print(f"  A budget    {total.a_budget}")
        print(f"  A estimator {total.a_estimator}")
    print(f"  wall {elapsed / 60:.1f} min over {workers} shards")

    return {"label": label, "a": a_text, "b": b_text,
            "a_wins": total.a_wins, "b_wins": total.b_wins,
            "draws": total.draws, "ci": [lo, hi],
            "exceptions": len(total.errors),
            "episode_median_s": median, "episode_p95_s": p95,
            "episode_max_s": worst,
            "search": total.a_search, "budget": total.a_budget,
            "estimator": total.a_estimator}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=tuple(PRESETS), required=True)
    parser.add_argument("--games", type=int, default=400,
                        help="games per matchup, split across shards")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bar", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for label, a_text, b_text in PRESETS[args.preset]:
        rows.append(run_matchup(label, a_text, b_text, args.games,
                                args.workers, args.seed, args.bar))
    out = args.out or (OUT_DIR / f"{args.preset}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
