"""Merge per-date replay datasets into one training corpus (Sprint 5B).

Globs data/processed/replays/replay_dataset_*.npz (one per day, written
by replays_parse.py --date) and concatenates them into replay_corpus.npz
with the same ragged layout the trainers consume:
    states, options_flat, option_counts, labels, values, episode_ids
plus replay_corpus.meta.json aggregating each day's validation counters
(games, pairs, coverage, unknown ids, MAX_OPTIONS overflows).

Everything under data/processed/replays/ is gitignored — the corpus is
reproducible from the daily Kaggle datasets.

Usage (repo root):
    python -m src.ingestion.replays_merge
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Final

import numpy as np

from .replays_parse import REPLAYS_OUT_DIR

logger = logging.getLogger(__name__)

CORPUS_PATH: Final[Path] = REPLAYS_OUT_DIR / "replay_corpus.npz"

_ARRAY_KEYS: Final[tuple[str, ...]] = (
    "states", "options_flat", "option_counts", "labels", "values",
    "episode_ids",
)


def merge(sources_dir: Path = REPLAYS_OUT_DIR,
          out_path: Path = CORPUS_PATH) -> dict:
    sources = sorted(p for p in sources_dir.glob("replay_dataset_*.npz"))
    if not sources:
        raise FileNotFoundError(
            f"no replay_dataset_*.npz under {sources_dir} — run "
            f"python -m src.ingestion.replays_parse --date <D> first")

    parts: dict[str, list[np.ndarray]] = {key: [] for key in _ARRAY_KEYS}
    per_source: list[dict] = []
    for source in sources:
        with np.load(source) as data:
            n = len(data["labels"])
            for key in _ARRAY_KEYS:
                if key == "values" and key not in data:  # pre-v2 file
                    parts[key].append(np.zeros(n, dtype=np.int8))
                else:
                    parts[key].append(data[key])
        meta_file = source.with_name(source.name.replace(".npz", ".meta.json"))
        source_meta: dict = {"source": source.name, "samples": n}
        if meta_file.exists():
            with open(meta_file, encoding="utf-8") as fh:
                day = json.load(fh)
            source_meta.update({key: day.get(key) for key in
                                ("games", "decision_pairs", "coverage",
                                 "skipped_unknown_id", "unknown_ids",
                                 "overflow_warnings", "overflow_lens")})
        per_source.append(source_meta)

    merged = {key: np.concatenate(parts[key], axis=0) for key in _ARRAY_KEYS}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **merged)

    total_pairs = sum(s.get("decision_pairs") or 0 for s in per_source)
    total_skipped = sum(s.get("skipped_unknown_id") or 0 for s in per_source)
    seen = total_pairs + total_skipped
    unknown_ids: dict[str, int] = {}
    overflow_lens: list[int] = []
    for source_meta in per_source:
        for key, count in (source_meta.get("unknown_ids") or {}).items():
            unknown_ids[key] = unknown_ids.get(key, 0) + count
        overflow_lens.extend(source_meta.get("overflow_lens") or [])
    meta = {
        "schema": "ptcg-replay-corpus-v1",
        "sources": per_source,
        "n_sources": len(sources),
        "games": sum(s.get("games") or 0 for s in per_source),
        "samples": int(len(merged["labels"])),
        "decision_pairs": total_pairs,
        "skipped_unknown_id": total_skipped,
        "coverage": (total_pairs / seen) if seen else 1.0,
        "unknown_ids": unknown_ids,
        "overflow_warnings": sum(s.get("overflow_warnings") or 0
                                 for s in per_source),
        "overflow_lens": overflow_lens,
    }
    meta_path = out_path.with_name(out_path.name.replace(".npz", ".meta.json"))
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    return meta


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources-dir", type=Path, default=REPLAYS_OUT_DIR)
    parser.add_argument("--out", type=Path, default=CORPUS_PATH)
    args = parser.parse_args()

    meta = merge(args.sources_dir, args.out)
    print(f"sources merged:   {meta['n_sources']}")
    for source in meta["sources"]:
        print(f"  {source['source']}: {source.get('games')} games, "
              f"{source.get('decision_pairs')} pairs")
    print(f"total games:      {meta['games']}")
    print(f"total samples:    {meta['samples']} "
          f"({meta['decision_pairs']} decision pairs)")
    print(f"id coverage:      {meta['coverage']:.4%} "
          f"({meta['skipped_unknown_id']} skipped)")
    if meta["unknown_ids"]:
        print(f"unknown ids:      {meta['unknown_ids']}")
    print(f"MAX_OPTIONS overflows: {meta['overflow_warnings']}"
          f"{' lens=' + str(meta['overflow_lens']) if meta['overflow_lens'] else ''}")
    print(f"corpus: {args.out} ({args.out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
