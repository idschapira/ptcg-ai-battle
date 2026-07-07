"""Empirical contract test: Tera Pokémon take no attack damage on the Bench.

Verified engine behavior (this test re-verifies it on every run):
  - the engine OFFERS a benched Tera as a damage target (option is legal),
  - but the damage does NOT land (HP unchanged),
  - while the same attack damages a non-Tera benched Pokémon normally.

Scenario: Stonjourner's "Stony Kick" (attack 987, unconditional
"also does 20 damage to 1 of your opponent's Benched Pokémon") against a
bench holding Teal Mask Ogerpon ex (Tera) and Glastrier (control).
The engine has no seed API, so each case replays games until the bench
scenario materializes (empirically < 25 attempts; capped at 80).

Run from the repo root:  python -m unittest tests.test_tera_bench_immunity
"""

from __future__ import annotations

import unittest
from typing import Final

from cg import game
from cg.api import OptionType, SelectContext

from src.environment_wrapper.wrapper import EnvironmentWrapper
from src.ingestion.card_index import CardIndex

STONJOURNER: Final[int] = 682
F_ENERGY: Final[int] = 6
STONY_KICK: Final[int] = 987
OGERPON_TERA: Final[int] = 96
CONTROL: Final[int] = 867  # Glastrier
DEF_ACTIVE: Final[int] = 41  # Ting-Lu
W_ENERGY: Final[int] = 3

ATTACKER_DECK: Final[list[int]] = [STONJOURNER] * 4 + [F_ENERGY] * 56
DEFENDER_DECK: Final[list[int]] = (
    [OGERPON_TERA] * 4 + [CONTROL] * 4 + [DEF_ACTIVE] * 4 + [W_ENERGY] * 48
)

MAX_ATTEMPTS: Final[int] = 80


class _ScriptedProbe:
    """Drives both players toward a Stony Kick on a chosen bench target."""

    def __init__(self, wrapper: EnvironmentWrapper, prefer_target: int) -> None:
        self._wrapper = wrapper
        self._prefer_target = prefer_target

    @staticmethod
    def _bench(obs_dict: dict) -> list[tuple[int, int]]:
        return [(p["id"], p["hp"]) for p in obs_dict["current"]["players"][1]["bench"]]

    def _choose(self, obs_dict: dict) -> tuple[list[int], bool]:
        obs = self._wrapper.parse(obs_dict)
        select = obs.select
        assert select is not None
        acting = obs.current.yourIndex
        options = select.option
        ctx = select.context
        ids = [self._wrapper.resolve_card_id(obs, o) for o in options]

        if ctx == SelectContext.IS_FIRST:
            yes = next(i for i, o in enumerate(options) if o.type == OptionType.YES)
            no = next(i for i, o in enumerate(options) if o.type == OptionType.NO)
            return ([no] if acting == 0 else [yes]), False
        if ctx == SelectContext.MULLIGAN:
            return [next((i for i, o in enumerate(options) if o.type == OptionType.NO), 0)], False
        if ctx == SelectContext.SETUP_ACTIVE_POKEMON:
            if acting == 1:
                for want in (DEF_ACTIVE, CONTROL):
                    for i, cid in enumerate(ids):
                        if cid == want:
                            return [i], False
            return [0], False
        if ctx == SelectContext.SETUP_BENCH_POKEMON:
            if acting == 0:
                return [], False
            picks: list[int] = []
            for want in (OGERPON_TERA, CONTROL):
                for i, cid in enumerate(ids):
                    if cid == want and i not in picks:
                        picks.append(i)
                        break
            picks = picks[: select.maxCount]
            while len(picks) < select.minCount:
                picks.append(next(i for i in range(len(options)) if i not in picks))
            return picks, False
        if ctx == SelectContext.MAIN and acting == 0:
            for i, o in enumerate(options):
                if o.type == OptionType.ATTACK and o.attackId == STONY_KICK:
                    return [i], False
            for i, o in enumerate(options):
                if o.type == OptionType.ATTACH:
                    return [i], False
            return [next(i for i, o in enumerate(options) if o.type == OptionType.END)], False
        if ctx == SelectContext.MAIN and acting == 1:
            return [next(i for i, o in enumerate(options) if o.type == OptionType.END)], False
        if ctx == SelectContext.DAMAGE and acting == 0:
            for i, cid in enumerate(ids):
                if cid == self._prefer_target:
                    return [i], True
            return [0], True
        return list(range(select.minCount)), False

    def measure_damage_pick(self) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
        """Play games until a Stony Kick resolves on a full (Tera+control) bench.

        Returns (bench_before, bench_after) around that resolution, or None
        if the scenario did not materialize in this game.
        """
        obs_dict, start = game.battle_start(list(ATTACKER_DECK), list(DEFENDER_DECK))
        if obs_dict is None:
            raise RuntimeError(f"battle_start failed: errorType={start.errorType}")
        try:
            for _ in range(3000):
                if obs_dict["current"]["result"] != -1:
                    return None
                before = self._bench(obs_dict)
                answer, is_damage_pick = self._choose(obs_dict)
                obs_dict = game.battle_select(answer)
                if is_damage_pick:
                    after = self._bench(obs_dict)
                    has_tera = any(c == OGERPON_TERA for c, _ in before)
                    has_ctrl = any(c == CONTROL for c, _ in before)
                    picked = dict(before).keys() == dict(after).keys()
                    if has_tera and has_ctrl and picked:
                        return before, after
            return None
        finally:
            game.battle_finish()


class TestTeraBenchImmunity(unittest.TestCase):
    wrapper: EnvironmentWrapper

    @classmethod
    def setUpClass(cls) -> None:
        cls.wrapper = EnvironmentWrapper(CardIndex())

    def _run_case(self, prefer_target: int) -> dict[int, int]:
        probe = _ScriptedProbe(self.wrapper, prefer_target)
        for _ in range(MAX_ATTEMPTS):
            measured = probe.measure_damage_pick()
            if measured is not None:
                before, after = measured
                return {cid: h1 - h0 for (cid, h0), (_, h1) in zip(before, after)}
        self.fail(f"scenario never materialized in {MAX_ATTEMPTS} games")

    def test_benched_tera_takes_no_attack_damage(self) -> None:
        deltas = self._run_case(prefer_target=OGERPON_TERA)
        self.assertEqual(deltas[OGERPON_TERA], 0, "engine no longer grants Tera bench immunity")
        self.assertEqual(deltas[CONTROL], 0, "control was not targeted, must be untouched")

    def test_benched_non_tera_takes_damage_normally(self) -> None:
        deltas = self._run_case(prefer_target=CONTROL)
        self.assertEqual(deltas[CONTROL], -20, "Stony Kick must deal 20 to a non-Tera bench")
        self.assertEqual(deltas[OGERPON_TERA], 0)


if __name__ == "__main__":
    unittest.main()
