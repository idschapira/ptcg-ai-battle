"""Download top-episode replays published by the competition.

VERIFIED (Kaggle web, 2026-07-07):
  - Replays are published as official DAILY DATASETS, one per day:
        kaggle/pokemon-tcg-ai-battle-episodes-<YYYY-MM-DD>
    indexed by kaggle/pokemon-tcg-ai-battle-episodes-index (manifest.csv
    with columns: date, daily_dataset_slug, daily_dataset_url,
    episode_count, total_bytes, top_avg_score, median_avg_score).
  - Each daily dataset holds ~5k <episode_id>.json replays (~2 MB each,
    ~21 GB/day raw) plus its own manifest.csv ("list of included episodes
    and their scores"), selected daily and ranked by average agent rating.
  - Replay JSON is the kaggle-environments wrapper (name "cabt",
    schema_version 1): configuration, info{Agents, EpisodeId, TeamNames},
    rewards, statuses, steps.

VERIFIED against real downloads (2026-07-05 episodes): steps[i] entries
follow the standard kaggle-environments schema {action, observation,
reward, status, info}, and steps[t].action answers steps[t-1].observation
(the action is stored one step AFTER the observation it responded to).
generate_sample_replays() below emits the same pairing.

Everything is parameterizable via ReplaySource so a schema change is a
config edit, not a rewrite. Auth: Kaggle CLI conventions (KAGGLE_USERNAME
/KAGGLE_KEY env vars or ~/.kaggle/kaggle.json).

Usage (repo root):
    python -m src.ingestion.replays_download --date 2026-07-05 --max-episodes 50
    python -m src.ingestion.replays_download --sample 20   # offline mode
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from .build_card_model import REPO_ROOT

logger = logging.getLogger(__name__)

REPLAYS_DIR: Final[Path] = REPO_ROOT / "data" / "raw" / "replays"


@dataclass(frozen=True)
class ReplaySource:
    """Where replays live; edit here (or pass flags) if Kaggle reorganizes."""

    index_slug: str = "kaggle/pokemon-tcg-ai-battle-episodes-index"
    daily_slug_template: str = "kaggle/pokemon-tcg-ai-battle-episodes-{date}"
    manifest_name: str = "manifest.csv"


def _kaggle(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run the Kaggle CLI via the current interpreter (no PATH assumptions)."""
    command = [sys.executable, "-m", "kaggle", *args]
    result = subprocess.run(command, capture_output=True, text=True, cwd=cwd,
                            timeout=600)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        if "credentials" in stderr.lower() or "unauthorized" in stderr.lower():
            raise RuntimeError(
                "Kaggle CLI is not authenticated. Set KAGGLE_USERNAME/KAGGLE_KEY "
                "or place ~/.kaggle/kaggle.json (Account -> Create New Token)."
            )
        raise RuntimeError(f"kaggle {' '.join(args)} failed: {stderr[:500]}")
    return result


def _download_file(slug: str, file_name: str, dest_dir: Path,
                   force: bool = False) -> Path:
    """Download one dataset file (idempotent: skips if already present).

    force=True re-fetches even when cached — used to refresh the INDEX
    manifest, which grows a new row per published day (a stale cache
    made the daily job miss every new date)."""
    target = dest_dir / file_name
    if target.exists():
        if not force:
            return target
        target.unlink()
    dest_dir.mkdir(parents=True, exist_ok=True)
    _kaggle("datasets", "download", slug, "-f", file_name, "-p", str(dest_dir))
    archive = dest_dir / f"{file_name}.zip"
    if archive.exists():  # the CLI zips large files
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest_dir)
        archive.unlink()
    if not target.exists():
        raise FileNotFoundError(f"{file_name} missing after download from {slug}")
    return target


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def download_replays(source: ReplaySource, date: str | None,
                     max_episodes: int, dest: Path = REPLAYS_DIR) -> int:
    """Download the top `max_episodes` replays for `date` (default: latest)."""
    index_dir = dest / "_index"
    index_manifest = _download_file(source.index_slug, source.manifest_name, index_dir)
    days = _read_manifest(index_manifest)
    if not days:
        raise RuntimeError("index manifest.csv is empty")
    row = (next((d for d in days if d.get("date") == date), None)
           if date else days[-1])
    if row is None and date is not None:
        # self-heal: the cached index may predate the requested day
        logger.info("date %s not in cached index — refreshing manifest", date)
        index_manifest = _download_file(source.index_slug,
                                        source.manifest_name, index_dir,
                                        force=True)
        days = _read_manifest(index_manifest)
        row = next((d for d in days if d.get("date") == date), None)
    if row is None:
        raise RuntimeError(f"date {date} not present in index manifest")
    slug_name = row["daily_dataset_slug"]
    slug = slug_name if "/" in slug_name else f"kaggle/{slug_name}"
    logger.info("daily dataset: %s (%s episodes listed)", slug, row.get("episode_count"))

    day_dir = dest / row["date"]
    daily_manifest = _download_file(slug, source.manifest_name, day_dir)
    episodes = _read_manifest(daily_manifest)
    # rank by score column when present (name per dataset docs may vary)
    score_key = next((k for k in ("avg_score", "score", "top_avg_score")
                      if episodes and k in episodes[0]), None)
    if score_key:
        episodes.sort(key=lambda e: -float(e.get(score_key) or 0))
    id_key = next((k for k in ("episode_id", "EpisodeId", "id")
                   if episodes and k in episodes[0]), None)
    if id_key is None:
        raise RuntimeError(f"cannot find episode id column in {daily_manifest}; "
                           f"columns={list(episodes[0].keys()) if episodes else []}")
    downloaded = 0
    for episode in episodes[:max_episodes]:
        file_name = f"{episode[id_key]}.json"
        _download_file(slug, file_name, day_dir)
        downloaded += 1
    logger.info("downloaded/present: %d replay files in %s", downloaded, day_dir)
    return downloaded


# --------------------------------------------------------------------------- #
# Offline sample mode: local self-play wrapped in the verified cabt schema
# --------------------------------------------------------------------------- #


def generate_sample_replays(n_games: int, dest: Path = REPLAYS_DIR / "sample",
                            seed: int = 0) -> int:
    """Generate kaggle-environments-shaped replays from local self-play.

    Used when no Kaggle credentials are available: exercises the full
    download->parse->encode pipeline with the same wrapper layout.
    """
    from cg import game as cg_game

    from ..agent_heuristics.heuristic_agent import HeuristicAgent
    from ..agent_heuristics.random_agent import RandomAgent, read_deck_csv

    dest.mkdir(parents=True, exist_ok=True)
    deck = read_deck_csv()
    written = 0
    for game_index in range(n_games):
        agents = (HeuristicAgent(seed=seed + 2 * game_index),
                  RandomAgent(seed=seed + 2 * game_index + 1))
        decisions: list[tuple[int, dict, list[int]]] = []
        obs_dict, start = cg_game.battle_start(list(deck), list(deck))
        if obs_dict is None:
            raise RuntimeError(f"battle_start failed: {start.errorType}")
        try:
            result = -1
            for _ in range(20_000):
                state = obs_dict["current"]
                result = state["result"]
                if result != -1:
                    break
                acting = state["yourIndex"]
                answer = agents[acting](obs_dict)
                observation = {k: obs_dict.get(k)
                               for k in ("select", "logs", "current")}
                observation["remainingOverageTime"] = 60
                decisions.append((acting, observation, answer))
                obs_dict = cg_game.battle_select(answer)
        finally:
            cg_game.battle_finish()
        # kaggle-environments pairing: the answer to steps[t-1]'s observation
        # is stored in steps[t], so each action lands one step after its
        # observation (the final step exists only to carry the last action).
        steps: list[list[dict]] = []
        for step_index in range(len(decisions) + 1):
            pair = []
            for agent_index in range(2):
                entry: dict = {"action": None,
                               "observation": {"remainingOverageTime": 60},
                               "reward": 0, "status": "INACTIVE", "info": {}}
                if (step_index < len(decisions)
                        and decisions[step_index][0] == agent_index):
                    entry["observation"] = decisions[step_index][1]
                    entry["status"] = "ACTIVE"
                if (step_index > 0
                        and decisions[step_index - 1][0] == agent_index):
                    entry["action"] = decisions[step_index - 1][2]
                pair.append(entry)
            steps.append(pair)
        rewards = ([0, 0] if result == 2 else
                   [1 if player == result else -1 for player in range(2)])
        replay = {
            "name": "cabt",
            "schema_version": 1,
            "configuration": {"seed": seed + game_index},
            "info": {"EpisodeId": 90_000_000 + game_index,
                     "TeamNames": ["local-heuristic", "local-random"]},
            "rewards": rewards,
            "statuses": ["DONE", "DONE"],
            "steps": steps,
        }
        path = dest / f"{replay['info']['EpisodeId']}.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(replay, fh, ensure_ascii=False)
        written += 1
    logger.info("wrote %d sample replays to %s", written, dest)
    return written


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=None, help="index date (default: latest)")
    parser.add_argument("--max-episodes", type=int, default=50)
    parser.add_argument("--sample", type=int, default=0, metavar="N",
                        help="offline mode: generate N local replays instead")
    parser.add_argument("--index-slug", default=ReplaySource.index_slug)
    parser.add_argument("--daily-slug-template", default=ReplaySource.daily_slug_template)
    args = parser.parse_args()

    if args.sample > 0:
        count = generate_sample_replays(args.sample)
    else:
        source = ReplaySource(index_slug=args.index_slug,
                              daily_slug_template=args.daily_slug_template)
        count = download_replays(source, args.date, args.max_episodes)
    print(f"replays available: {count}")


if __name__ == "__main__":
    main()
