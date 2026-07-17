"""Build and validate submission tarballs (Gate A) — parameterized.

Targets (--target):
    crustle  (default)  submission.tar.gz          Final A: repo main.py +
                        deck.csv (Crustle e10, CrustleAgent v3, rollback
                        NetworkAgent par 5D). Identical to the historical
                        Gate A bundle.
    spidops             submission_spidops.tar.gz  Final B: main_spidops.py
                        packaged AS main.py + data/decks/meta_spidops.csv
                        packaged AS deck.csv (NetworkAgent with the MATED
                        pair bc_spidops_v2.npz + feature_stats.npz).

The target NEVER touches the tracked Final-A files: per-target sources
are renamed via tar arcname at build time, so shipping one final cannot
clobber the other.

Bundle layout (tar root == agent root on Kaggle):
    main.py                     entrypoint exposing agent(obs_dict)
    deck.csv                    60 card ids
    cg/                         official engine bindings + binaries
    src/                        runtime modules (numpy inference — never torch)
    models/*.npz                policy weights + frozen feature stats
    data/processed/*.parquet    CardIndex tables

Validations: main.py resolves at the tar root, size stays under the
competition limit, and a smoke test runs the packaged main.py in a clean
subprocess TWICE: as a module import and via exec() without __file__
(the REAL Kaggle loader). The module smoke also reports per-selection
latency over a full engine game. Run smokes on an idle machine — the
live-engine game can flake under heavy CPU load.

Run from the repo root:
    python -m src.build_submission
    python -m src.build_submission --target spidops
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Final

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

SIZE_LIMIT_MIB: Final[float] = 197.7

# Runtime every target needs (deck/main/pilot weights are per-target).
SHARED_BUNDLE: Final[tuple[str, ...]] = (
    "cg",
    "src/__init__.py",
    "src/ingestion/__init__.py",
    "src/ingestion/build_card_model.py",
    "src/ingestion/build_effect_model.py",
    "src/ingestion/dim_effect_overrides.csv",
    "src/ingestion/card_index.py",
    "src/environment_wrapper/__init__.py",
    "src/environment_wrapper/wrapper.py",
    "src/agent_heuristics/__init__.py",
    "src/agent_heuristics/random_agent.py",
    "src/agent_heuristics/heuristic_agent.py",
    "src/rl_models/__init__.py",
    "src/rl_models/encoding.py",
    "src/rl_models/normalization.py",
    "src/rl_models/network_numpy.py",
    "src/rl_models/network_agent.py",
    "models/feature_stats.npz",
    "data/processed/dim_card.parquet",
    "data/processed/dim_attack.parquet",
    "data/processed/dim_skill.parquet",
    "data/processed/bridge_attack_energy.parquet",
    "data/processed/dim_effect.parquet",
)


@dataclass(frozen=True)
class TargetConfig:
    """One shippable final: sources, extras and smoke expectations."""

    name: str
    output_name: str
    main_source: str            # packaged as main.py
    deck_source: str            # packaged as deck.csv
    extra_entries: tuple[str, ...]
    deck_sentinel: int          # card id that proves WHICH deck shipped
    # python asserts pinning WHICH pilot shipped (module / exec namespaces)
    pilot_assert_module: str
    pilot_assert_exec: str


TARGETS: Final[dict[str, TargetConfig]] = {
    "crustle": TargetConfig(
        name="crustle",
        output_name="submission.tar.gz",
        main_source="main.py",
        deck_source="deck.csv",
        extra_entries=("src/agent_heuristics/crustle_agent.py",
                       "models/policy_value.npz"),
        deck_sentinel=345,  # Crustle
        pilot_assert_module=(
            'assert type(main._agent).__name__ == "CrustleAgent", type(main._agent)\n'
            'assert main._agent._v3 is True, "packaged pilot must be the v3 variant"\n'
            'from src.rl_models.network_agent import NetworkAgent\n'
            'rollback = NetworkAgent(deck_path="deck.csv")\n'
            'assert rollback._fallback is None, "rollback network weights not in bundle"\n'
        ),
        pilot_assert_exec=(
            'assert env["_agent"]._v3 is True, "packaged pilot must be the v3 variant"\n'
        ),
    ),
    "spidops": TargetConfig(
        name="spidops",
        output_name="submission_spidops.tar.gz",
        main_source="main_spidops.py",
        deck_source="data/decks/meta_spidops.csv",
        extra_entries=("models/bc_spidops_v2.npz",),
        deck_sentinel=401,  # Team Rocket's Spidops
        pilot_assert_module=(
            'assert type(main._agent).__name__ == "NetworkAgent", type(main._agent)\n'
            'assert main._agent._fallback is None, "bc_spidops_v2 weights not in bundle"\n'
        ),
        pilot_assert_exec=(
            'assert env["_agent"]._fallback is None, "bc_spidops_v2 weights not in bundle"\n'
        ),
    ),
}


_SMOKE_SCRIPT_TEMPLATE: Final[str] = textwrap.dedent(
    """
    import time
    import main

    fake_select = {
        "select": {
            "type": 9, "context": 41, "minCount": 1, "maxCount": 1,
            "remainDamageCounter": 0, "remainEnergyCost": 0,
            "option": [{"type": 1}, {"type": 2}],
            "deck": None, "contextCard": None, "effect": None,
        },
        "logs": [],
        "current": None,
    }
    answer = main.agent(fake_select)
    assert isinstance(answer, list), f"not a list: {answer!r}"
    assert len(answer) == 1, f"minCount/maxCount violated: {answer!r}"
    assert answer[0] in (0, 1), f"index out of range: {answer!r}"

    __PILOT_ASSERT__

    # Initial selection: obs.select is None -> the 60 deck.csv ids.
    deck = main.agent({"select": None, "logs": [], "current": None})
    assert len(deck) == 60, f"deck must have 60 cards, got {len(deck)}"
    assert all(isinstance(c, int) for c in deck), "deck ids must be int"
    assert __SENTINEL__ in deck, "packaged deck.csv is not the target deck"

    # End to end: a full engine game piloted by the packaged agent on both
    # sides, with per-selection latency (the submission budget is ms/move).
    from cg import game
    obs_dict, start = game.battle_start(list(deck), list(deck))
    assert obs_dict is not None, f"battle_start failed: {start}"
    times = []
    try:
        result = -1
        for _ in range(3000):
            result = obs_dict["current"]["result"]
            if result != -1:
                break
            t0 = time.perf_counter()
            answer = main.agent(obs_dict)
            times.append((time.perf_counter() - t0) * 1e6)
            obs_dict = game.battle_select(answer)
        assert result in (0, 1, 2), f"game did not finish (result={result})"
        turn = obs_dict["current"]["turn"]
    finally:
        game.battle_finish()
    times.sort()
    n = len(times)
    print(f"smoke OK: selection={answer} deck head={deck[:5]} "
          f"full game result={result} turn={turn} | latency "
          f"mean={sum(times)/n:.0f}us p99={times[min(n-1, int(n*0.99))]:.0f}us "
          f"({n} selections)")
    """
)

# kaggle_environments does NOT import main.py: it runs exec(source, env)
# in a namespace WITHOUT __file__. This step replicates that loader
# exactly — it would have caught the "Validation Episode failed"
# NameError on __file__ that a run-as-file smoke cannot see.
_EXEC_LOADER_TEMPLATE: Final[str] = textwrap.dedent(
    """
    env = {}
    with open("main.py", encoding="utf-8") as fh:
        source = fh.read()
    exec(compile(source, "<kaggle-exec>", "exec"), env)  # no __file__
    assert "__file__" not in env, "loader fidelity broken: __file__ leaked"
    agent = env["agent"]
    __PILOT_ASSERT__

    fake_select = {
        "select": {
            "type": 9, "context": 41, "minCount": 1, "maxCount": 1,
            "remainDamageCounter": 0, "remainEnergyCost": 0,
            "option": [{"type": 1}, {"type": 2}],
            "deck": None, "contextCard": None, "effect": None,
        },
        "logs": [],
        "current": None,
    }
    answer = agent(fake_select)
    assert isinstance(answer, list) and len(answer) == 1, repr(answer)
    assert answer[0] in (0, 1), f"index out of range: {answer!r}"

    deck = agent({"select": None, "logs": [], "current": None})
    assert len(deck) == 60, f"deck must have 60 cards, got {len(deck)}"
    assert all(isinstance(c, int) for c in deck), "deck ids must be int"
    assert __SENTINEL__ in deck, "packaged deck.csv is not the target deck"

    from cg import game
    obs_dict, start = game.battle_start(list(deck), list(deck))
    assert obs_dict is not None, f"battle_start failed: {start}"
    try:
        result = -1
        for _ in range(3000):
            result = obs_dict["current"]["result"]
            if result != -1:
                break
            obs_dict = game.battle_select(agent(obs_dict))
        assert result in (0, 1, 2), f"game did not finish (result={result})"
    finally:
        game.battle_finish()
    print(f"exec-loader smoke OK (no __file__): selection={answer} "
          f"deck head={deck[:5]} full game result={result}")
    """
)


def _render(template: str, target: TargetConfig, exec_mode: bool) -> str:
    pilot = (target.pilot_assert_exec if exec_mode
             else target.pilot_assert_module)
    return (template
            .replace("__PILOT_ASSERT__", pilot.rstrip())
            .replace("__SENTINEL__", str(target.deck_sentinel)))


def _iter_shared_files() -> list[Path]:
    files: list[Path] = []
    for entry in SHARED_BUNDLE:
        path = REPO_ROOT / entry
        if not path.exists():
            raise FileNotFoundError(f"bundle entry missing: {entry}")
        if path.is_dir():
            files.extend(p for p in sorted(path.rglob("*"))
                         if p.is_file() and "__pycache__" not in p.parts)
        else:
            files.append(path)
    return files


def build(target: TargetConfig) -> Path:
    output = REPO_ROOT / target.output_name
    files = _iter_shared_files()
    for entry in target.extra_entries:
        path = REPO_ROOT / entry
        if not path.exists():
            raise FileNotFoundError(f"bundle entry missing: {entry}")
        files.append(path)
    main_src = REPO_ROOT / target.main_source
    deck_src = REPO_ROOT / target.deck_source
    for path in (main_src, deck_src):
        if not path.exists():
            raise FileNotFoundError(f"target source missing: {path}")
    with tarfile.open(output, "w:gz") as tar:
        for path in files:
            tar.add(path, arcname=path.relative_to(REPO_ROOT).as_posix())
        tar.add(main_src, arcname="main.py")
        tar.add(deck_src, arcname="deck.csv")
    return output


def validate(archive: Path) -> None:
    with tarfile.open(archive, "r:gz") as tar:
        names = tar.getnames()
    if "main.py" not in names:
        raise AssertionError("main.py is not at the tar root")
    if "deck.csv" not in names:
        raise AssertionError("deck.csv is not at the tar root")

    size_mib = archive.stat().st_size / (1024 * 1024)
    if size_mib >= SIZE_LIMIT_MIB:
        raise AssertionError(f"{size_mib:.1f} MiB exceeds the {SIZE_LIMIT_MIB} MiB limit")
    print(f"validate OK: main.py+deck.csv at root, {len(names)} members, "
          f"{size_mib:.1f} MiB (< {SIZE_LIMIT_MIB} MiB)")


def smoke_test(archive: Path, target: TargetConfig) -> None:
    """Run the packaged main.py in a clean subprocess, twice: as a module
    import (dev loader) and via exec() without __file__ (the REAL Kaggle
    loader — kaggle_environments never sets __file__)."""
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(tmp, filter="data")
        for template, exec_mode in ((_SMOKE_SCRIPT_TEMPLATE, False),
                                    (_EXEC_LOADER_TEMPLATE, True)):
            script = _render(template, target, exec_mode)
            result = subprocess.run(
                [sys.executable, "-c", script],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                raise AssertionError(
                    f"smoke test failed:\n{result.stdout}\n{result.stderr}")
            print(result.stdout.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=tuple(TARGETS), default="crustle")
    args = parser.parse_args()
    target = TARGETS[args.target]
    archive = build(target)
    print(f"built {archive.relative_to(REPO_ROOT)} (target={target.name})")
    validate(archive)
    smoke_test(archive, target)


if __name__ == "__main__":
    main()
