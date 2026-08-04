"""Unit tests das funções puras do portfolio watch (limiares/alertas).

Rodar da raiz do repo:  python -m unittest tests.test_portfolio_watch
"""

from __future__ import annotations

import unittest

from src.analysis.portfolio_watch import (ACTIVE_REFS, FINALS, DaySnapshot,
                                          EloSeries, WatchConfig,
                                          evaluate_alerts, eviction_guard)


def _snap(day: str, shares: dict[str, float],
          ignore_fx: float = 0.0) -> DaySnapshot:
    return DaySnapshot(day=day, total=200, shares=shares,
                       ignore_fx_share=ignore_fx)


BASELINE = {"Alakazam box (non-ex)": 0.65,
            "Team Rocket Spidops (non-ex)": 0.12,
            "Crustle + Mega Kangaskhan stall": 0.15}


class TestRadarAlerts(unittest.TestCase):

    def test_quiet_meta_fires_nothing(self) -> None:
        snaps = [_snap(f"2026-07-{d:02d}", BASELINE, 0.01)
                 for d in range(10, 17)]
        self.assertEqual(evaluate_alerts(snaps), [])

    def test_empty_history_flags_harvest(self) -> None:
        alerts = evaluate_alerts([])
        self.assertEqual(len(alerts), 1)
        self.assertIn("harvest", alerts[0])

    def test_ignore_fx_needs_sustained_days(self) -> None:
        # 1 dia alto (burst tipo 07-12) NÃO dispara…
        snaps = [_snap("2026-07-11", BASELINE, 0.01),
                 _snap("2026-07-12", BASELINE, 0.23),
                 _snap("2026-07-13", BASELINE, 0.01)]
        self.assertEqual(evaluate_alerts(snaps), [])
        # …2 dias consecutivos disparam
        snaps.append(_snap("2026-07-14", BASELINE, 0.08))
        snaps.append(_snap("2026-07-15", BASELINE, 0.09))
        alerts = evaluate_alerts(snaps)
        self.assertTrue(any("IGNORE-EFFECTS" in a for a in alerts), alerts)

    def test_uncovered_archetype_needs_share_and_growth(self) -> None:
        rising = dict(BASELINE)
        rising["Archaludon ex box"] = 0.12
        flat_before = dict(BASELINE)
        flat_before["Archaludon ex box"] = 0.12
        # mesmo share do dia anterior (não cresce) -> silêncio
        self.assertEqual(
            evaluate_alerts([_snap("2026-07-14", flat_before),
                             _snap("2026-07-15", rising)]), [])
        # crescendo -> dispara
        smaller = dict(BASELINE)
        smaller["Archaludon ex box"] = 0.06
        alerts = evaluate_alerts([_snap("2026-07-14", smaller),
                                  _snap("2026-07-15", rising)])
        self.assertTrue(any("NAO-COBERTO" in a for a in alerts), alerts)

    def test_pillar_shifts(self) -> None:
        collapsed = {"Alakazam box (non-ex)": 0.30,
                     "Crustle + Mega Kangaskhan stall": 0.35}
        snaps = [_snap(f"2026-07-{d:02d}", collapsed) for d in (14, 15, 16)]
        alerts = evaluate_alerts(snaps)
        self.assertTrue(any("Alakazam" in a for a in alerts), alerts)
        self.assertTrue(any("Kangaskhan" in a for a in alerts), alerts)


class TestEvictionGuard(unittest.TestCase):
    """As séries são montadas a partir de ACTIVE_REFS, nunca de refs
    cravados: o portfólio roda (submissão nova evicta a mais antiga por
    data), e fixtures com ref literal passam a testar submissão
    aposentada — a guarda devolve [] e o teste quebra sem nenhum bug.
    Foi exatamente o que aconteceu com 54791820/54667957."""

    def _series(self, first_scores: list[float],
                second_scores: list[float]) -> EloSeries:
        series = EloSeries()
        days = [f"2026-07-{d:02d}" for d in range(10, 10 + len(first_scores))]
        series.by_ref[ACTIVE_REFS[0]] = list(zip(days, first_scores))
        series.by_ref[ACTIVE_REFS[1]] = list(zip(days, second_scores))
        return series

    def test_static_final_while_other_moves_fires(self) -> None:
        series = self._series([600.0, 600.0, 600.0], [860.0, 861.0, 862.0])
        alerts = eviction_guard(series)
        self.assertTrue(
            any("GUARDA" in a and FINALS[ACTIVE_REFS[0]] in a
                for a in alerts), alerts)

    def test_everything_static_is_quiet(self) -> None:
        # ninguém se moveu (ex.: ladder parado) — sem evidência de eviction
        series = self._series([600.0, 600.0, 600.0], [861.0, 861.0, 861.0])
        self.assertEqual(eviction_guard(series), [])

    def test_short_series_is_quiet(self) -> None:
        series = self._series([600.0], [861.0])
        self.assertEqual(eviction_guard(series), [])

    def test_static_other_final_fires_not_the_moving_one(self) -> None:
        # a 2ª subindo, a 1ª estática -> a guarda acusa exatamente a 1ª
        series = self._series([600.0, 640.0, 700.0], [861.0, 861.0, 861.0])
        alerts = eviction_guard(series)
        self.assertTrue(
            any(FINALS[ACTIVE_REFS[1]] in a for a in alerts), alerts)
        self.assertFalse(
            any(FINALS[ACTIVE_REFS[0]] in a for a in alerts), alerts)

    def test_every_active_ref_has_a_label(self) -> None:
        # a guarda indexa FINALS[ref] ao disparar: um ref ativo sem label
        # seria KeyError no exato momento do alerta.
        for ref in ACTIVE_REFS:
            self.assertIn(ref, FINALS, ref)


if __name__ == "__main__":
    unittest.main()
