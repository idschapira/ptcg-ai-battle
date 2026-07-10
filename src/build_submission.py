"""Build and validate submission.tar.gz (Gate A).

Bundle layout (tar root == agent root on Kaggle):
    main.py                     entrypoint exposing agent(obs_dict)
    deck.csv                    60 card ids
    cg/                         official engine bindings + binaries
    src/                        runtime modules (ingestion, wrapper, agents,
                                rl_models numpy inference — never torch)
    models/*.npz                policy-net weights + frozen feature stats
    data/processed/*.parquet    CardIndex tables

Validations: main.py resolves at the tar root, size stays under the
competition limit, and a smoke test imports the packaged main.py in a
clean subprocess and runs a fake selection + the initial deck selection.

Run from the repo root:  python -m src.build_submission
"""

from __future__ import annotations

import subprocess
import sys
import tarfile
import tempfile
import textwrap
from pathlib import Path
from typing import Final

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
OUTPUT: Final[Path] = REPO_ROOT / "submission.tar.gz"

SIZE_LIMIT_MIB: Final[float] = 197.7

# Everything the agent needs at runtime — and nothing else.
BUNDLE: Final[tuple[str, ...]] = (
    "main.py",
    "deck.csv",
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
    "src/agent_heuristics/crustle_agent.py",
    "src/rl_models/__init__.py",
    "src/rl_models/encoding.py",
    "src/rl_models/normalization.py",
    "src/rl_models/network_numpy.py",
    "src/rl_models/network_agent.py",
    "models/feature_stats.npz",
    "models/policy_value.npz",
    "data/processed/dim_card.parquet",
    "data/processed/dim_attack.parquet",
    "data/processed/dim_skill.parquet",
    "data/processed/bridge_attack_energy.parquet",
    "data/processed/dim_effect.parquet",
)

_SMOKE_SCRIPT: Final[str] = textwrap.dedent(
    """
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

    # The bundle must ship the SPECIALIZED pilot (Crustle strategy overlay)
    # in its v2 variant, and the rollback pilot (NetworkAgent + models/*.npz)
    # must stay loadable.
    assert type(main._agent).__name__ == "CrustleAgent", type(main._agent)
    assert main._agent._v2 is True, "packaged pilot must be the v2 variant"
    from src.rl_models.network_agent import NetworkAgent
    rollback = NetworkAgent(deck_path="deck.csv")
    assert rollback._fallback is None, "rollback network weights not in bundle"

    # Initial selection: obs.select is None -> the 60 deck.csv ids.
    deck = main.agent({"select": None, "logs": [], "current": None})
    assert len(deck) == 60, f"deck must have 60 cards, got {len(deck)}"
    assert all(isinstance(c, int) for c in deck), "deck ids must be int"
    assert 345 in deck, "packaged deck.csv must be the Crustle list"

    # End to end: a full engine game piloted by the packaged agent on both
    # sides. Must finish decisively with every answer legal (the engine
    # rejects illegal answers by ending the game with an error result).
    from cg import game
    obs_dict, start = game.battle_start(list(deck), list(deck))
    assert obs_dict is not None, f"battle_start failed: {start}"
    try:
        result = -1
        for _ in range(3000):
            result = obs_dict["current"]["result"]
            if result != -1:
                break
            obs_dict = game.battle_select(main.agent(obs_dict))
        assert result in (0, 1, 2), f"game did not finish (result={result})"
        turn = obs_dict["current"]["turn"]
    finally:
        game.battle_finish()

    print(f"smoke OK: selection={answer} deck head={deck[:5]} "
          f"full game result={result} turn={turn}")
    """
)

# kaggle_environments does NOT import main.py: it runs exec(source, env)
# in a namespace WITHOUT __file__. This step replicates that loader
# exactly — it would have caught the "Validation Episode failed"
# NameError on __file__ that a run-as-file smoke cannot see.
_EXEC_LOADER_SCRIPT: Final[str] = textwrap.dedent(
    """
    env = {}
    with open("main.py", encoding="utf-8") as fh:
        source = fh.read()
    exec(compile(source, "<kaggle-exec>", "exec"), env)  # no __file__
    assert "__file__" not in env, "loader fidelity broken: __file__ leaked"
    agent = env["agent"]
    assert env["_agent"]._v2 is True, "packaged pilot must be the v2 variant"

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
    assert 345 in deck, "packaged deck.csv must be the Crustle list"

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


def _iter_bundle_files() -> list[Path]:
    files: list[Path] = []
    for entry in BUNDLE:
        path = REPO_ROOT / entry
        if not path.exists():
            raise FileNotFoundError(f"bundle entry missing: {entry}")
        if path.is_dir():
            files.extend(p for p in sorted(path.rglob("*"))
                         if p.is_file() and "__pycache__" not in p.parts)
        else:
            files.append(path)
    return files


def build(output: Path = OUTPUT) -> Path:
    files = _iter_bundle_files()
    with tarfile.open(output, "w:gz") as tar:
        for path in files:
            tar.add(path, arcname=path.relative_to(REPO_ROOT).as_posix())
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


def smoke_test(archive: Path) -> None:
    """Run the packaged main.py in a clean subprocess, twice: as a module
    import (dev loader) and via exec() without __file__ (the REAL Kaggle
    loader — kaggle_environments never sets __file__)."""
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(tmp, filter="data")
        for script in (_SMOKE_SCRIPT, _EXEC_LOADER_SCRIPT):
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
    archive = build()
    print(f"built {archive.relative_to(REPO_ROOT)}")
    validate(archive)
    smoke_test(archive)


if __name__ == "__main__":
    main()
