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

Arm spec grammar: kind[,weights[,stats]] with kind in
{random, heuristic, crustle, crustle-v2, network}.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
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
    "random", "heuristic", "crustle", "crustle-v2", "network")


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
    """Per-arm instrumentation aggregated across games."""

    calls: int = 0
    time_us: float = 0.0

    @property
    def mean_latency_us(self) -> float:
        return self.time_us / self.calls if self.calls else 0.0


class _Instrumented:
    """Wraps one agent instance, feeding the arm's shared metrics."""

    __slots__ = ("_agent", "_metrics")

    def __init__(self, agent: Agent, metrics: ArmMetrics) -> None:
        self._agent = agent
        self._metrics = metrics

    def __call__(self, obs_dict: dict) -> list[int]:
        t0 = time.perf_counter()
        answer = self._agent(obs_dict)
        self._metrics.time_us += (time.perf_counter() - t0) * 1e6
        self._metrics.calls += 1
        return answer


def arm_factory(spec: ArmSpec, index: CardIndex, effects: EffectIndex,
                metrics: ArmMetrics) -> Callable[[int], Agent]:
    """Per-game factory for an arm (index/effects shared; network shared)."""
    if spec.kind == "heuristic":
        from ..agent_heuristics.heuristic_agent import HeuristicAgent
        base = lambda s: HeuristicAgent(seed=s, index=index, effects=effects)
    elif spec.kind == "crustle":
        from ..agent_heuristics.crustle_agent import CrustleAgent
        base = lambda s: CrustleAgent(seed=s, index=index, effects=effects)
    elif spec.kind == "crustle-v2":
        from ..agent_heuristics.crustle_agent import CrustleAgent
        base = lambda s: CrustleAgent(seed=s, index=index, effects=effects,
                                      variant="v2")
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
    pair = run_pair(arm_factory(spec_a, index, effects, metrics_a),
                    arm_factory(spec_b, index, effects, metrics_b),
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
    print(f"  VERDICT vs bar {c.bar:.0%}: {c.verdict}")


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
    parser.add_argument("--mode", choices=("agent", "deck", "field"),
                        required=True)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bar", type=float, default=0.5)
    parser.add_argument("--deck", type=Path, help="agent mode: shared deck")
    parser.add_argument("--a", type=str, help="agent mode: arm A spec")
    parser.add_argument("--b", type=str, help="agent mode: arm B spec")
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
    else:
        if not args.candidate:
            parser.error("--mode field requires --candidate")
        _mode_field(args, index, effects)


if __name__ == "__main__":
    main()
