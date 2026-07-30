"""Reusable A/B harness (agent / deck / field) with real statistics.

Encodes the decision discipline the ad-hoc A/Bs (Gate C, v2-vs-v1,
gauntlet non-regression) applied by hand: alternating seats, Wilson 95%
confidence intervals over DECIDED games, and a PASS/HOLD/FAIL verdict
against a configurable bar. Reuses gauntlet.run_pair (which reuses
selfplay.play_one_game) — the engine loop is never reimplemented.

PAIRED SEEDS (common random numbers): NOT POSSIBLE with this engine, by
source inspection — battle_start() exposes no seed; the C++ ApiBattleStart
hardcodes `config.seed = std::random_device()` + a random seed_seq
(Api.h:29-78); and shuffles call `std::shuffle(..., std::random_device())`
DIRECTLY (CardMove.h:263, EffectInstant.h:585), bypassing even the
game's own mt19937 — so patching the config seed still would not pair
the shuffles. Fallback, documented here: agent RNG is seeded per game
(reproducible in distribution), game randomness is independent, and the
harness compensates with sample size + Wilson CIs. Head-to-head modes
are still variance-efficient: both pilots share every game.

Verdict semantics (bar B, default 0.5):
    PASS  = CI lower bound  > B   (statistically above the bar)
    FAIL  = CI upper bound  < B   (statistically below)
    HOLD  = CI straddles B        (inconclusive: collect more games)

CLI (repo root):
  python -m src.environment_wrapper.ab_test --mode agent \
      --deck data/decks/seed_crustle.csv --a crustle-v2 --b crustle \
      --games 200 --bar 0.5
  python -m src.environment_wrapper.ab_test --mode deck \
      --pilot heuristic --a-deck data/decks/seed_crustle.csv \
      --b-deck data/decks/placeholder_abomasnow.csv --games 200
  python -m src.environment_wrapper.ab_test --mode field \
      --candidate data/decks/seed_crustle.csv@crustle-v2 --games 80 \
      --baseline-ref refs/field_v1.json --save-ref refs/field_v2.json
  python -m src.environment_wrapper.ab_test --mode matchup \
      --a deck.csv@crustle-v3 \
      --b "data/decks/meta_alakazam.csv@network,models/bc_alakazam.npz" \
      --games 200

Arm spec grammar: kind[,weights[,stats]] with kind in
{random, heuristic, crustle, crustle-v2, crustle-v3, network}.
matchup mode pairs deck AND pilot per side (<deck.csv>@<armspec>) — the
full cross needed to calibrate the internal field against ladder
matchups (each side plays its own deck with its own brain).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Final

from ..agent_heuristics.random_agent import RandomAgent
from ..deckbuilding.gauntlet import PairResult, discover_decks, run_pair
from ..deckbuilding.legality import read_deck_ids, validate_deck
from ..ingestion.build_effect_model import EffectIndex
from ..ingestion.card_index import CardIndex
from .selfplay import Agent

Z_95: Final[float] = 1.959963984540054
REGRESSION_MARGIN: Final[float] = 0.05

ARM_KINDS: Final[tuple[str, ...]] = (
    "random", "heuristic", "crustle", "crustle-v2", "crustle-v3", "network",
    # runtime search (submission candidate). "-blind" pins the estimator
    # off so the arm degrades to its prior — that is the FLOOR arm, and
    # comparing it against plain crustle-v3 is how the floor gets proven
    # empirically rather than asserted.
    # "-match" models the opponent in rollouts with the pilot that fits
    # the ESTIMATED archetype instead of a generic heuristic.
    # "-margin" makes the search prove its case before overriding the
    # prior; "-both" applies the matched opponent model AND the margin.
    # "-adaptive" spends the bank only on CONTESTED decisions.
    "search-crustle", "search-crustle-blind", "search-crustle-match",
    "search-crustle-margin", "search-crustle-both", "search-crustle-adaptive",
    # "search-net,<npz>" models the ALAKAZAM opponent in rollouts with
    # that behaviour-cloned net. Pinning the clone is what lets the
    # fidelity ladder be built on one opponent: exact clone, a DIFFERENT
    # human's clone of the same archetype, or none at all.
    "search-net", "search-net-adaptive",
    # parametric league pilot: "grimmsnarl-module" flies meta_grimmsnarl
    # with GrimmsnarlModule + its shipped theta. Needed as an opponent
    # that is NOT the rollout model.
    "grimmsnarl-module")

# Arms that carry their own search/estimator/budget instrumentation.
SEARCH_ARMS: Final[frozenset[str]] = frozenset(
    {"search-crustle", "search-crustle-blind", "search-crustle-match",
     "search-crustle-margin", "search-crustle-both",
     "search-crustle-adaptive", "search-net", "search-net-adaptive"})

# One extra win in four determinizations — the smallest gain a 4x4
# search can express that is not a single lucky rollout.
OVERRIDE_MARGIN: Final[float] = 0.25


# --------------------------------------------------------------------------- #
# Statistics layer (pure, unit-tested)
# --------------------------------------------------------------------------- #


def wilson_interval(wins: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion; (0,1) when n=0."""
    if n <= 0:
        return 0.0, 1.0
    p = wins / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def verdict(wins: int, n: int, bar: float) -> str:
    """PASS/HOLD/FAIL from the Wilson CI against the bar (see module doc)."""
    lo, hi = wilson_interval(wins, n)
    if lo > bar:
        return "PASS"
    if hi < bar:
        return "FAIL"
    return "HOLD"


def newcombe_difference(w1: int, n1: int, w2: int,
                        n2: int) -> tuple[float, float, float]:
    """(diff, lo, hi) for p2 - p1: Newcombe's hybrid-score interval.

    The right interval for a difference of two INDEPENDENT proportions.
    Two separate Wilson intervals cannot be compared by eye -- their
    overlapping is not the same question as the difference containing
    zero -- and the normal approximation on the difference misbehaves
    near 0 and 1, which is exactly where these winrates live. Newcombe
    method 10 combines the two Wilson intervals instead:

        lo = (p2-p1) - sqrt((p2-l2)^2 + (u1-p1)^2)
        hi = (p2-p1) + sqrt((u2-p2)^2 + (p1-l1)^2)
    """
    if n1 <= 0 or n2 <= 0:
        return 0.0, -1.0, 1.0
    p1, p2 = w1 / n1, w2 / n2
    l1, u1 = wilson_interval(w1, n1)
    l2, u2 = wilson_interval(w2, n2)
    diff = p2 - p1
    lo = diff - math.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2)
    hi = diff + math.sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2)
    return diff, max(-1.0, lo), min(1.0, hi)


def binomial_p_value(wins: int, n: int, p0: float = 0.5) -> float:
    """Two-sided normal-approximation p-value for H0: p == p0."""
    if n <= 0 or not 0.0 < p0 < 1.0:
        return 1.0
    z = (wins / n - p0) / math.sqrt(p0 * (1.0 - p0) / n)
    return min(1.0, 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2)))))


# --------------------------------------------------------------------------- #
# Arms (pilot specs) and instrumentation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ArmSpec:
    """One pilot: kind plus optional network weights/stats overrides."""

    kind: str
    weights: Path | None = None
    stats: Path | None = None

    @classmethod
    def parse(cls, text: str) -> "ArmSpec":
        parts = [p.strip() for p in text.split(",")]
        kind = parts[0]
        if kind not in ARM_KINDS:
            raise SystemExit(f"unknown arm kind '{kind}' (choose from {ARM_KINDS})")
        weights = Path(parts[1]) if len(parts) > 1 and parts[1] else None
        stats = Path(parts[2]) if len(parts) > 2 and parts[2] else None
        return cls(kind, weights, stats)

    def label(self) -> str:
        extra = f",{self.weights.name}" if self.weights else ""
        return f"{self.kind}{extra}"


@dataclass
class ArmMetrics:
    """Per-arm instrumentation aggregated across games.

    ``episode_wall_s`` is the list of per-GAME wall times for this arm.
    For search arms that is the number the 600s bank is spent against —
    mean latency per selection says nothing useful once an agent is
    allowed to think for seconds at a time.
    """

    calls: int = 0
    time_us: float = 0.0
    episode_wall_s: list[float] = field(default_factory=list)
    search: object | None = None       # RuntimeSearchStats, if a search arm
    budget: object | None = None       # BudgetStats, if a search arm
    estimator: object | None = None    # EstimatorStats, if a search arm

    @property
    def mean_latency_us(self) -> float:
        return self.time_us / self.calls if self.calls else 0.0

    def episode_percentiles(self) -> tuple[float, float, float]:
        """(median, p95, max) episode wall seconds; zeros when empty."""
        if not self.episode_wall_s:
            return 0.0, 0.0, 0.0
        ordered = sorted(self.episode_wall_s)
        n = len(ordered)
        return (ordered[n // 2],
                ordered[min(n - 1, int(n * 0.95))],
                ordered[-1])


class _Instrumented:
    """Wraps one agent instance, feeding the arm's shared metrics.

    One instance per GAME (run_pair builds agents per game), so this is
    also where per-episode wall time is accumulated: the first call
    opens a new episode bucket and every call adds to it.
    """

    __slots__ = ("_agent", "_metrics", "_episode")

    def __init__(self, agent: Agent, metrics: ArmMetrics) -> None:
        self._agent = agent
        self._metrics = metrics
        self._episode: int | None = None

    def __call__(self, obs_dict: dict) -> list[int]:
        if self._episode is None:
            self._metrics.episode_wall_s.append(0.0)
            self._episode = len(self._metrics.episode_wall_s) - 1
        t0 = time.perf_counter()
        answer = self._agent(obs_dict)
        elapsed = time.perf_counter() - t0
        self._metrics.time_us += elapsed * 1e6
        self._metrics.calls += 1
        self._metrics.episode_wall_s[self._episode] += elapsed
        return answer


def arm_factory(spec: ArmSpec, index: CardIndex, effects: EffectIndex,
                metrics: ArmMetrics,
                deck: list[int] | None = None) -> Callable[[int], Agent]:
    """Per-game factory for an arm (index/effects shared; network shared).

    ``deck`` is the arm's OWN 60. Search arms need it to determinize
    their own hidden zones; the other kinds ignore it.
    """
    if spec.kind in SEARCH_ARMS:
        from ..rl_models.budget import BudgetGuard, BudgetStats
        from ..rl_models.opponent_estimator import (EstimatorStats,
                                                    OpponentDeckEstimator)
        from ..rl_models.runtime_search_agent import (RuntimeSearchAgent,
                                                      RuntimeSearchStats)
        # stats are SHARED across the pair's games so the report can say
        # what the search actually did over the whole run; the estimator
        # and the guard are per-game, because their state is per-episode.
        search_stats = RuntimeSearchStats()
        budget_stats = BudgetStats()
        estimator_stats = EstimatorStats()
        metrics.search = search_stats
        metrics.budget = budget_stats
        metrics.estimator = estimator_stats
        from ..rl_models.runtime_search_agent import (OPPONENT_PILOT_GENERIC,
                                                      OPPONENT_PILOT_MATCH)
        blind = spec.kind == "search-crustle-blind"
        opponent_pilot = (
            OPPONENT_PILOT_MATCH
            if spec.kind in ("search-crustle-match", "search-crustle-both")
            else OPPONENT_PILOT_GENERIC)
        margin = (OVERRIDE_MARGIN
                  if spec.kind in ("search-crustle-margin",
                                   "search-crustle-both")
                  else 0.0)
        from ..rl_models.runtime_search_agent import (
            DEFAULT_CONTESTED_MARGIN, OPPONENT_PILOT_NETWORK)
        contested = (DEFAULT_CONTESTED_MARGIN
                     if spec.kind in ("search-crustle-adaptive",
                                      "search-net-adaptive") else None)
        # search-net pins WHICH clone models the Alakazam opponent; the
        # weights field of the arm spec carries the npz path.
        nets = None
        if spec.kind in ("search-net", "search-net-adaptive"):
            if spec.weights is None:
                raise SystemExit(f"{spec.kind} needs ,<npz> (rollout model)")
            if not spec.weights.exists():
                raise SystemExit(f"rollout model missing: {spec.weights}")
            opponent_pilot = OPPONENT_PILOT_NETWORK
            # Point EVERY modellable archetype at this clone. Only the
            # archetype the estimator actually reports is ever used, and
            # each experiment faces one opponent, so this is unambiguous
            # and keeps the arm usable for any cell.
            from ..rl_models.runtime_search_agent import ARCHETYPE_NETWORKS
            nets = {k: str(spec.weights) for k in ARCHETYPE_NETWORKS}

        def base(s: int) -> Agent:
            return RuntimeSearchAgent(
                index=index, effects=effects, seed=s,
                own_deck_ids=deck or [],
                enable_search=not blind,
                opponent_pilot=opponent_pilot,
                override_margin=margin,
                contested_margin=contested,
                archetype_networks=nets,
                estimator=OpponentDeckEstimator(index=index,
                                                stats=estimator_stats),
                guard=BudgetGuard(stats=budget_stats),
                stats=search_stats)
    elif spec.kind == "grimmsnarl-module":
        from ..league.modules import MODULES
        from ..league.parametric_agent import ParametricHeuristicAgent
        module = MODULES["grimmsnarl"]
        theta = None
        theta_path = (spec.weights if spec.weights is not None
                      else Path("data/theta/grimmsnarl_heur_v1.json"))
        if theta_path.exists():
            # from_dict is keyed by NAME and clips into the legal bands,
            # so a stale genome lands on the right knobs or not at all
            with open(theta_path, encoding="utf-8") as fh:
                theta = module.schema.from_dict(json.load(fh))
        else:
            raise SystemExit(f"theta not found: {theta_path}")
        base = lambda s: ParametricHeuristicAgent(  # noqa: E731
            module=module, theta=theta, seed=s, index=index, effects=effects)
    elif spec.kind == "heuristic":
        from ..agent_heuristics.heuristic_agent import HeuristicAgent
        base = lambda s: HeuristicAgent(seed=s, index=index, effects=effects)
    elif spec.kind == "crustle":
        from ..agent_heuristics.crustle_agent import CrustleAgent
        base = lambda s: CrustleAgent(seed=s, index=index, effects=effects)
    elif spec.kind in ("crustle-v2", "crustle-v3"):
        from ..agent_heuristics.crustle_agent import CrustleAgent
        variant = spec.kind.removeprefix("crustle-")
        base = lambda s: CrustleAgent(seed=s, index=index, effects=effects,
                                      variant=variant)
    elif spec.kind == "network":
        from ..rl_models.network_agent import NetworkAgent
        network = NetworkAgent(index=index, effects=effects,
                               weights_path=spec.weights,
                               stats_path=spec.stats)
        if network._fallback is not None:
            raise SystemExit(f"network weights missing for arm {spec.label()}")
        base = lambda s: network
    else:
        base = lambda s: RandomAgent(seed=s)
    return lambda s: _Instrumented(base(s), metrics)


# --------------------------------------------------------------------------- #
# Comparison runner + report
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Comparison:
    """One two-arm comparison, fully summarized."""

    label: str
    pair: PairResult
    a_metrics: ArmMetrics
    b_metrics: ArmMetrics
    bar: float

    @property
    def decided(self) -> int:
        return self.pair.a_wins + self.pair.b_wins

    @property
    def ci(self) -> tuple[float, float]:
        return wilson_interval(self.pair.a_wins, self.decided)

    @property
    def verdict(self) -> str:
        return verdict(self.pair.a_wins, self.decided, self.bar)

    @property
    def p_value(self) -> float:
        return binomial_p_value(self.pair.a_wins, self.decided, self.bar)


def compare(label: str, spec_a: ArmSpec, spec_b: ArmSpec,
            deck_a: list[int], deck_b: list[int], n_games: int, seed: int,
            bar: float, index: CardIndex, effects: EffectIndex) -> Comparison:
    metrics_a, metrics_b = ArmMetrics(), ArmMetrics()
    pair = run_pair(arm_factory(spec_a, index, effects, metrics_a, deck_a),
                    arm_factory(spec_b, index, effects, metrics_b, deck_b),
                    deck_a, deck_b, n_games, seed)
    return Comparison(label, pair, metrics_a, metrics_b, bar)


def print_comparison(c: Comparison) -> None:
    lo, hi = c.ci
    pair = c.pair
    games = pair.games + len(pair.errors)
    selections = c.a_metrics.calls + c.b_metrics.calls
    print(f"{c.label}")
    print(f"  n={games} decided={c.decided} draws={pair.draws} "
          f"wins A={pair.a_wins} B={pair.b_wins}")
    print(f"  winrate A = {pair.winrate_a:.1%}  "
          f"IC95 [{lo:.1%}, {hi:.1%}]  p={c.p_value:.4f} (H0: p={c.bar:.0%})")
    print(f"  avg turns {pair.avg_turns:.1f}  "
          f"avg selections/game {selections / max(pair.games, 1):.1f}  "
          f"latency A {c.a_metrics.mean_latency_us:.0f}us / "
          f"B {c.b_metrics.mean_latency_us:.0f}us")
    print(f"  exceptions {len(pair.errors)} (must be 0)")
    for error in pair.errors[:5]:
        print(f"    {error}")
    for tag, metrics in (("A", c.a_metrics), ("B", c.b_metrics)):
        _print_budget_block(tag, metrics)
    print(f"  VERDICT vs bar {c.bar:.0%}: {c.verdict}")


# The real runtime constraint is a per-episode BANK, not a per-move
# deadline: actTimeout=0 and remainingOverageTime starts at 600s per
# agent per episode (environment specification, verified in 1012/1012
# corpus replays). Kaggle runs on 2 vCPU; we assume it is up to 3x
# slower than the dev box, so every measured episode time is reported
# with that projection next to it.
BANK_S: Final[float] = 600.0
KAGGLE_SLOWDOWN: Final[float] = 3.0


def _print_budget_block(tag: str, metrics: ArmMetrics) -> None:
    """Episode wall time vs the 600s bank — only for instrumented arms."""
    if not metrics.episode_wall_s:
        return
    median, p95, worst = metrics.episode_percentiles()
    print(f"  [{tag}] episode wall  median {median:.1f}s  p95 {p95:.1f}s  "
          f"max {worst:.1f}s   "
          f"| x{KAGGLE_SLOWDOWN:.0f} projection: median "
          f"{median * KAGGLE_SLOWDOWN:.0f}s  max "
          f"{worst * KAGGLE_SLOWDOWN:.0f}s  "
          f"({worst * KAGGLE_SLOWDOWN / BANK_S:.0%} of the {BANK_S:.0f}s bank)")
    if metrics.search is not None:
        print(f"  [{tag}] search    {metrics.search.summary()}")
    if metrics.budget is not None:
        print(f"  [{tag}] budget    {metrics.budget.summary()}")
    if metrics.estimator is not None:
        print(f"  [{tag}] estimator {metrics.estimator.summary()}")


def _load_deck(path: Path, index: CardIndex) -> list[int]:
    ids = read_deck_ids(path)
    report = validate_deck(ids, index)
    if not report.ok:
        for error in report.errors:
            print(f"  - {error}")
        raise SystemExit(f"deck {path} is ILLEGAL — aborting")
    return ids


# --------------------------------------------------------------------------- #
# CLI modes
# --------------------------------------------------------------------------- #


def _mode_agent(args, index: CardIndex, effects: EffectIndex) -> None:
    deck = _load_deck(args.deck, index)
    spec_a, spec_b = ArmSpec.parse(args.a), ArmSpec.parse(args.b)
    c = compare(f"[agent A/B] deck={args.deck.name}: "
                f"A={spec_a.label()} vs B={spec_b.label()}",
                spec_a, spec_b, deck, deck, args.games, args.seed, args.bar,
                index, effects)
    print_comparison(c)


def _mode_deck(args, index: CardIndex, effects: EffectIndex) -> None:
    spec = ArmSpec.parse(args.pilot)
    deck_a = _load_deck(args.a_deck, index)
    deck_b = _load_deck(args.b_deck, index)
    c = compare(f"[deck A/B] pilot={spec.label()}: "
                f"A={args.a_deck.name} vs B={args.b_deck.name}",
                spec, spec, deck_a, deck_b, args.games, args.seed, args.bar,
                index, effects)
    print_comparison(c)


def _parse_deck_arm(text: str, flag: str,
                    index: CardIndex) -> tuple[Path, list[int], ArmSpec]:
    """<deck.csv>@<armspec> -> (path, validated deck ids, arm spec)."""
    if "@" not in text:
        raise SystemExit(f"{flag} must be <deck.csv>@<armspec>")
    deck_text, arm_text = text.split("@", 1)
    path = Path(deck_text)
    return path, _load_deck(path, index), ArmSpec.parse(arm_text)


def _mode_matchup(args, index: CardIndex, effects: EffectIndex) -> None:
    path_a, deck_a, spec_a = _parse_deck_arm(args.a, "--a", index)
    path_b, deck_b, spec_b = _parse_deck_arm(args.b, "--b", index)
    c = compare(f"[matchup] A={path_a.name}@{spec_a.label()} vs "
                f"B={path_b.name}@{spec_b.label()}",
                spec_a, spec_b, deck_a, deck_b, args.games, args.seed,
                args.bar, index, effects)
    print_comparison(c)


def _mode_field(args, index: CardIndex, effects: EffectIndex) -> None:
    if "@" not in args.candidate:
        raise SystemExit("--candidate must be <deck.csv>@<armspec>")
    deck_path, arm_text = args.candidate.split("@", 1)
    spec = ArmSpec.parse(arm_text)
    candidate_deck = _load_deck(Path(deck_path), index)
    opponent = ArmSpec.parse("heuristic")  # same field pilot as the history

    baseline: dict[str, float] = {}
    if args.baseline_ref is not None and args.baseline_ref.exists():
        with open(args.baseline_ref, encoding="utf-8") as fh:
            baseline = {k: float(v) for k, v in json.load(fh).items()}

    results: dict[str, float] = {}
    rates: list[float] = []
    worst: tuple[str, float] | None = None
    regressions: list[str] = []
    for name, path in discover_decks().items():
        if path.resolve() == Path(deck_path).resolve():
            continue
        c = compare(f"[field] candidate vs {name}", spec, opponent,
                    candidate_deck, _load_deck(path, index),
                    args.games, args.seed, args.bar, index, effects)
        print_comparison(c)
        rate = c.pair.winrate_a
        results[name] = rate
        rates.append(rate)
        if worst is None or rate < worst[1]:
            worst = (name, rate)
        ref = baseline.get(name)
        if ref is not None and rate < ref - REGRESSION_MARGIN:
            regressions.append(f"{name}: {rate:.1%} < ref {ref:.1%}")

    mean = sum(rates) / len(rates) if rates else 0.0
    if worst is not None:
        print(f"\n[field] mean winrate: {mean:.1%}  "
              f"worst matchup: {worst[0]} {worst[1]:.1%}")
    else:
        print("\n[field] no opposing decks found")
    if baseline:
        print(f"[field] regressions vs {args.baseline_ref.name} "
              f"(margin {REGRESSION_MARGIN:.0%}): "
              f"{regressions if regressions else 'none'}")
    if args.save_ref is not None:
        args.save_ref.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_ref, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=1)
        print(f"[field] reference saved: {args.save_ref}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("agent", "deck", "field", "matchup"),
                        required=True)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bar", type=float, default=0.5)
    parser.add_argument("--deck", type=Path, help="agent mode: shared deck")
    parser.add_argument("--a", type=str,
                        help="agent mode: arm A spec; "
                             "matchup mode: <deck.csv>@<armspec>")
    parser.add_argument("--b", type=str,
                        help="agent mode: arm B spec; "
                             "matchup mode: <deck.csv>@<armspec>")
    parser.add_argument("--pilot", type=str, help="deck mode: shared pilot")
    parser.add_argument("--a-deck", type=Path, help="deck mode: deck A")
    parser.add_argument("--b-deck", type=Path, help="deck mode: deck B")
    parser.add_argument("--candidate", type=str,
                        help="field mode: <deck.csv>@<armspec>")
    parser.add_argument("--baseline-ref", type=Path, default=None)
    parser.add_argument("--save-ref", type=Path, default=None)
    args = parser.parse_args()

    index = CardIndex()
    effects = EffectIndex()
    if args.mode == "agent":
        if not (args.deck and args.a and args.b):
            parser.error("--mode agent requires --deck, --a, --b")
        _mode_agent(args, index, effects)
    elif args.mode == "deck":
        if not (args.pilot and args.a_deck and args.b_deck):
            parser.error("--mode deck requires --pilot, --a-deck, --b-deck")
        _mode_deck(args, index, effects)
    elif args.mode == "matchup":
        if not (args.a and args.b):
            parser.error("--mode matchup requires --a, --b")
        _mode_matchup(args, index, effects)
    else:
        if not args.candidate:
            parser.error("--mode field requires --candidate")
        _mode_field(args, index, effects)


if __name__ == "__main__":
    main()
