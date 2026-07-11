"""Print the ELO time series collected by scripts/track_elo.sh.

Reads data/elo/elo_log.csv (one row per COMPLETE submission per daily
collection + a TOP row for the leaderboard leader) and prints one
series per submission, plus the gap to the leaderboard top. Stdlib
only; if matplotlib happens to be installed, --plot saves a PNG.

Run from the repo root:
    python -m src.analysis.elo_report [--plot data/elo/elo.png]
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Final

from ..ingestion.build_card_model import REPO_ROOT

CSV_PATH: Final[Path] = REPO_ROOT / "data" / "elo" / "elo_log.csv"


def load_series(path: Path = CSV_PATH) -> dict[str, list[tuple[str, float]]]:
    """description -> [(collect_date, score), ...] (None-safe on gaps)."""
    series: dict[str, list[tuple[str, float]]] = defaultdict(list)
    if not path.exists():
        return series
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            raw = (row.get("public_score") or "").strip()
            if not raw:
                continue
            try:
                score = float(raw)
            except ValueError:
                continue
            label = (row.get("description") or row.get("ref") or "?").strip()
            series[label].append((row.get("collect_date") or "?", score))
    return dict(series)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--plot", type=Path, default=None,
                        help="save a PNG (only if matplotlib is available)")
    args = parser.parse_args()

    series = load_series(args.csv)
    if not series:
        print(f"nenhum dado em {args.csv} — rode scripts/track_elo.sh antes")
        return

    top = series.pop("TOP", [])
    top_latest = top[-1][1] if top else None
    for label in sorted(series):
        points = series[label]
        line = "  ".join(f"{d}:{s:.1f}" for d, s in points)
        latest = points[-1][1]
        gap = f"  (gap p/ topo: {latest - top_latest:+.1f})" if top_latest else ""
        print(f"{label:24s} {line}{gap}")
    if top:
        print(f"{'[leaderboard TOP]':24s} "
              + "  ".join(f"{d}:{s:.1f}" for d, s in top))

    if args.plot is not None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib indisponível — pulei o --plot")
            return
        fig, ax = plt.subplots(figsize=(8, 4))
        for label, points in {**series, "TOP": top}.items():
            if points:
                ax.plot([d for d, _ in points], [s for _, s in points],
                        marker="o", label=label)
        ax.legend()
        ax.set_ylabel("publicScore (ELO)")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(args.plot)
        print(f"plot salvo: {args.plot}")


if __name__ == "__main__":
    main()
