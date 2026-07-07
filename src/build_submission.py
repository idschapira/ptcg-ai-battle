"""Build and validate submission.tar.gz (Gate A).

Bundle layout (tar root == agent root on Kaggle):
    main.py                     entrypoint exposing agent(obs_dict)
    deck.csv                    60 card ids
    cg/                         official engine bindings + binaries
    src/                        runtime modules (ingestion, wrapper, agents)
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

    deck = main.agent({"select": None, "logs": [], "current": None})
    assert len(deck) == 60, f"deck must have 60 cards, got {len(deck)}"
    assert all(isinstance(c, int) for c in deck), "deck ids must be int"

    print(f"smoke OK: selection={answer} deck head={deck[:5]}")
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
    """Import the packaged main.py in a clean subprocess and run fake selections."""
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(tmp, filter="data")
        result = subprocess.run(
            [sys.executable, "-c", _SMOKE_SCRIPT],
            cwd=tmp,
            capture_output=True,
            text=True,
            timeout=120,
        )
    if result.returncode != 0:
        raise AssertionError(f"smoke test failed:\n{result.stdout}\n{result.stderr}")
    print(result.stdout.strip())


def main() -> None:
    archive = build()
    print(f"built {archive.relative_to(REPO_ROOT)}")
    validate(archive)
    smoke_test(archive)


if __name__ == "__main__":
    main()
