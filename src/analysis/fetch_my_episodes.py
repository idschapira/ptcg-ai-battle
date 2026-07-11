"""Fetch OUR submission's ladder episodes (especially losses) from Kaggle.

The daily replay datasets only publish the competition's TOP episodes;
a mid-table submission never appears there. This module uses the same
endpoints the community simulation scrapers rely on:

  POST https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes
       {"submissionId": <id>}            -> episodes + per-agent rewards
  GET  https://www.kaggleusercontent.com/episodes/<episodeId>.json
       -> full kaggle-environments replay (steps/info/rewards)

Both are unauthenticated/public (verified 11/Jul/2026). reward semantics
per agent: 1 = win, -1 = loss, 0 = draw. Downloads are idempotent (skip
existing files) and land in viewer/episodes/ (gitignored).

Run from the repo root:
    python -m src.analysis.fetch_my_episodes                # v2, losses
    python -m src.analysis.fetch_my_episodes --submission 54535198 \
        --results loss draw --max 12
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import requests

from ..ingestion.build_card_model import REPO_ROOT

LIST_URL: Final[str] = ("https://www.kaggle.com/api/i/"
                        "competitions.EpisodeService/ListEpisodes")
REPLAY_URL: Final[str] = "https://www.kaggleusercontent.com/episodes/{id}.json"
EPISODES_DIR: Final[Path] = REPO_ROOT / "viewer" / "episodes"

V2_SUBMISSION_ID: Final[int] = 54553592  # "Crustle V2" (tracker ref)

_REWARD_RESULT: Final[dict[int, str]] = {1: "win", -1: "loss", 0: "draw"}


@dataclass(frozen=True)
class EpisodeSummary:
    """One ladder episode from our submission's perspective."""

    episode_id: int
    create_time: str
    result: str          # win / loss / draw / unknown
    our_index: int       # our seat (0/1)
    opponent: str        # opponent team name if resolvable
    score_delta: float | None

    @property
    def label(self) -> str:
        delta = (f"{self.score_delta:+.1f}"
                 if self.score_delta is not None else "?")
        return (f"{self.episode_id}  {self.create_time[:16]}  "
                f"seat={self.our_index}  {self.result:7s}  elo {delta}  "
                f"vs {self.opponent}")


def list_episodes(submission_id: int,
                  timeout: int = 120) -> list[EpisodeSummary]:
    """All COMPLETED episodes of one submission, newest first (None-safe)."""
    resp = requests.post(LIST_URL, json={"submissionId": submission_id},
                         timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    teams = {t.get("id"): t.get("teamName", "?")
             for t in payload.get("teams", []) or []}

    episodes: list[EpisodeSummary] = []
    for episode in payload.get("episodes", []) or []:
        if episode.get("state") != "COMPLETED":
            continue
        ours = None
        theirs = None
        for agent in episode.get("agents", []) or []:
            if agent.get("submissionId") == submission_id:
                ours = agent
            else:
                theirs = agent
        if ours is None:
            continue
        initial, updated = ours.get("initialScore"), ours.get("updatedScore")
        delta = (updated - initial
                 if initial is not None and updated is not None else None)
        episodes.append(EpisodeSummary(
            episode_id=int(episode.get("id", 0)),
            create_time=str(episode.get("createTime", "")),
            result=_REWARD_RESULT.get(ours.get("reward"), "unknown"),
            our_index=int(ours.get("index", 0) or 0),
            opponent=teams.get((theirs or {}).get("teamId"), "?"),
            score_delta=delta,
        ))
    episodes.sort(key=lambda e: e.create_time, reverse=True)
    return episodes


def download_replay(episode_id: int, dest_dir: Path = EPISODES_DIR,
                    timeout: int = 300) -> Path:
    """Idempotent replay download (kaggle-environments JSON)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / f"{episode_id}.json"
    if target.exists() and target.stat().st_size > 0:
        return target
    resp = requests.get(REPLAY_URL.format(id=episode_id), timeout=timeout)
    resp.raise_for_status()
    json.loads(resp.text)  # validate before writing
    target.write_text(resp.text, encoding="utf-8")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=int, default=V2_SUBMISSION_ID)
    parser.add_argument("--results", nargs="+", default=["loss"],
                        choices=("win", "loss", "draw", "unknown"),
                        help="which outcomes to download (default: losses)")
    parser.add_argument("--max", type=int, default=12,
                        help="cap on downloads (newest first)")
    parser.add_argument("--dest", type=Path, default=EPISODES_DIR)
    args = parser.parse_args()

    episodes = list_episodes(args.submission)
    counts = {"win": 0, "loss": 0, "draw": 0, "unknown": 0}
    for episode in episodes:
        counts[episode.result] = counts.get(episode.result, 0) + 1
    total = len(episodes)
    winrate = counts["win"] / max(counts["win"] + counts["loss"], 1)
    print(f"submission {args.submission}: {total} episódios COMPLETED — "
          f"{counts['win']}V/{counts['loss']}D/{counts['draw']}E "
          f"(winrate decididos {winrate:.1%})")

    wanted = [e for e in episodes if e.result in set(args.results)]
    print(f"baixando até {args.max} de {len(wanted)} episódios "
          f"com resultado em {args.results}:")
    downloaded = 0
    for episode in wanted[:args.max]:
        try:
            path = download_replay(episode.episode_id, args.dest)
        except (requests.RequestException, json.JSONDecodeError) as exc:
            print(f"  FALHOU {episode.episode_id}: {exc}")
            continue
        downloaded += 1
        print(f"  {episode.label}  -> {path.relative_to(REPO_ROOT)}")
    print(f"baixados/presentes: {downloaded} em {args.dest}")


if __name__ == "__main__":
    main()
