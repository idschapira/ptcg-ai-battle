"""IC do AGREGADO ponderado — o número em que o veredito se apoia.

`pilot_ab` imprime o delta ponderado pelo campo real, mas sem intervalo:
sem ele não dá para separar "+1,3pp" de zero, e o motor não é semeável
(stdev medido de 3,7pp a 150 jogos, 28/Jul), então um agregado sem IC é
exatamente o tipo de número que já produziu 3 falsos-positivos.

O agregado é Σ wᵢ·dᵢ com pesos wᵢ normalizados sobre as células jogadas.
Cada dᵢ é uma diferença de duas binomiais independentes (braços rodados
em processos separados), logo

    Var(dᵢ) = pᴬ(1-pᴬ)/nᴬ + pᴮ(1-pᴮ)/nᴮ
    Var(Σ wᵢdᵢ) = Σ wᵢ² Var(dᵢ)      (células independentes)

e o IC95 normal sobre essa variância. É aproximação (não é Newcombe),
mas o agregado soma milhares de jogos por braço, onde o normal é honesto;
os ICs POR CÉLULA continuam sendo os de Newcombe do pilot_ab.

Rodar da raiz do repo:
    python -m src.analysis.pilot_ab_aggregate data/processed/pilot_ab/*.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Final

Z_95: Final[float] = 1.959964


def aggregate(payload: dict) -> dict:
    """Delta ponderado + IC95 (None-safe: célula sem jogos é ignorada)."""
    rows = payload.get("rows") or []
    total_w = 0.0
    delta = 0.0
    variance = 0.0
    cells = 0
    for row in rows:
        a, b = row.get("a") or {}, row.get("b") or {}
        na, nb = a.get("decided") or 0, b.get("decided") or 0
        share = row.get("share") or 0.0
        if not na or not nb or share <= 0:
            continue
        pa, pb = a.get("winrate") or 0.0, b.get("winrate") or 0.0
        total_w += share
        delta += share * (pa - pb)
        variance += (share ** 2) * (pa * (1 - pa) / na + pb * (1 - pb) / nb)
        cells += 1
    if total_w <= 0:
        return {"delta": 0.0, "lo": 0.0, "hi": 0.0, "cells": 0, "weight": 0.0}
    delta /= total_w
    stderr = math.sqrt(variance) / total_w
    return {"delta": delta, "lo": delta - Z_95 * stderr,
            "hi": delta + Z_95 * stderr, "stderr": stderr,
            "cells": cells, "weight": total_w,
            "games_per_arm": sum((r.get("a") or {}).get("decided") or 0
                                 for r in rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()

    print(f"  {'variante':34s}{'delta':>9s}{'IC95 do agregado':>22s}"
          f"{'jogos/braco':>13s}  veredito")
    print("  " + "-" * 86)
    for path in args.files:
        try:
            with open(path, encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  {path.name}: ilegivel ({exc})")
            continue
        agg = aggregate(payload)
        name = Path(payload.get("a", path.stem)).name
        verdict = ("BATE o ship" if agg["lo"] > 0 else
                   "PIOR que o ship" if agg["hi"] < 0 else "NULO (IC cruza 0)")
        print(f"  {name:34s}{agg['delta'] * 100:+9.2f}"
              f"   [{agg['lo'] * 100:+6.2f}, {agg['hi'] * 100:+6.2f}]"
              f"{agg['games_per_arm']:13d}  {verdict}")


if __name__ == "__main__":
    main()
