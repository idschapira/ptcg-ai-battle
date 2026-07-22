"""Contract tests for the Grimmsnarl module.

Every rule is asserted as a DIFFERENCE against the generic
HeuristicAgent on the identical observation — a module that merely
tracks the generic score is not a strategy. The last group asserts the
property that makes the module safe to put in a co-evolutionary league:
piloting a deck it knows nothing about, it degrades to generic play and
still answers legally, with zero exceptions.

Run from the repo root:
    python -m unittest tests.test_league_grimmsnarl
"""

from __future__ import annotations

import copy
import unittest
from typing import Final

from cg import game
from cg.api import AreaType, OptionType
from cg.api import Option as ApiOption

from src.agent_heuristics.heuristic_agent import HeuristicAgent
from src.deckbuilding.legality import read_deck_ids
from src.ingestion.build_card_model import REPO_ROOT
from src.ingestion.build_effect_model import EffectIndex
from src.ingestion.card_index import CardIndex
from src.league.modules.grimmsnarl import (BOSS_ORDERS, DARK_ENERGY,
                                           GRIMMSNARL_EX, IMPIDIMP,
                                           MORGREM, MUNKIDORI, RARE_CANDY,
                                           SNORUNT, GrimmsnarlModule)
from src.league.parametric_agent import ParametricHeuristicAgent

DECKS_DIR: Final = REPO_ROOT / "data" / "decks"
POKEGEAR: Final[int] = 1122


def _capture_main_obs(deck: list[int], opponent: list[int]) -> dict:
    """First MAIN observation of a real game with this deck in seat 0."""
    index = CardIndex()
    pilot = HeuristicAgent(seed=0, index=index)
    obs_dict, _ = game.battle_start(list(deck), list(opponent))
    try:
        for _ in range(20_000):
            state = obs_dict["current"]
            if state["result"] != -1:
                break
            select = obs_dict.get("select")
            if (state["yourIndex"] == 0 and select is not None
                    and select.get("context") == 0):
                return copy.deepcopy(obs_dict)
            obs_dict = game.battle_select(pilot(obs_dict))
    finally:
        game.battle_finish()
    raise RuntimeError("no MAIN observation captured")


class _GrimmsnarlCase(unittest.TestCase):
    base_obs: dict

    @classmethod
    def setUpClass(cls) -> None:
        cls.index = CardIndex()
        cls.effects = EffectIndex()
        cls.deck = read_deck_ids(DECKS_DIR / "meta_grimmsnarl.csv")
        cls.base_obs = _capture_main_obs(
            cls.deck, read_deck_ids(DECKS_DIR / "meta_spidops.csv"))
        cls.generic = HeuristicAgent(index=cls.index, effects=cls.effects)
        cls.agent = ParametricHeuristicAgent(
            module=GrimmsnarlModule(), index=cls.index, effects=cls.effects)
        cls.theta = cls.agent.theta

    def _mutated(self, hand0_id: int | None = None,
                 my_active: dict | None = None,
                 my_bench: list[dict] | None = None,
                 opp_active: dict | None = None,
                 opp_bench: list[dict] | None = None):
        obs_dict = copy.deepcopy(self.base_obs)
        state = obs_dict["current"]
        me = state["yourIndex"]
        mine, theirs = state["players"][me], state["players"][1 - me]
        my_tpl = dict(mine["active"][0])
        opp_tpl = dict(theirs["active"][0])
        if hand0_id is not None:
            mine["hand"][0]["id"] = hand0_id
        if my_active is not None:
            mine["active"][0].update(my_active)
        if my_bench is not None:
            mine["bench"] = [dict(my_tpl, **b) for b in my_bench]
        if opp_active is not None:
            theirs["active"][0].update(opp_active)
        if opp_bench is not None:
            theirs["bench"] = [dict(opp_tpl, **b) for b in opp_bench]
        return self.agent._wrapper.parse(obs_dict)

    @staticmethod
    def _play0() -> ApiOption:
        return ApiOption(type=OptionType.PLAY, area=AreaType.HAND, index=0)


class TestGrimmsnarlSetupRules(_GrimmsnarlCase):
    """Rule group 1: the stage-2 climb outranks everything generic."""

    def test_rare_candy_beats_the_generic_trainer_band(self) -> None:
        obs = self._mutated(hand0_id=RARE_CANDY)
        option = self._play0()
        module = self.agent._main_score(obs, option)
        generic = self.generic._main_score(obs, option)
        self.assertEqual(module, self.theta["rare_candy_score"])
        self.assertGreater(module, generic,
                           "Rare Candy is the shortcut to Punk Up: it must "
                           "outrank a plain trainer")

    def test_rare_candy_stays_below_actually_evolving(self) -> None:
        # invariant: the shortcut must never outrank the real thing
        self.assertLess(self.theta["rare_candy_score"], 80.0)

    def test_evolving_into_grimmsnarl_beats_any_other_evolution(self) -> None:
        obs = self._mutated(hand0_id=GRIMMSNARL_EX)
        evolve = ApiOption(type=OptionType.EVOLVE, area=AreaType.HAND,
                           index=0, playerIndex=obs.current.yourIndex)
        into_grimm = self.agent._main_score(obs, evolve)

        obs2 = self._mutated(hand0_id=MORGREM)
        into_morgrem = self.agent._main_score(obs2, evolve)

        self.assertGreater(into_grimm, into_morgrem,
                           "Punk Up (5 energy) is the reason to evolve")
        self.assertGreater(into_grimm, self.generic._main_score(obs, evolve),
                           "must differ from the generic evolve band")
        self.assertEqual(
            into_grimm - self.generic._main_score(obs, evolve),
            self.theta["evolve_grimmsnarl_bonus"])


class TestGrimmsnarlAbilityEngine(_GrimmsnarlCase):
    """Rule group 2: abilities are the engine (BC prior: 80.8%)."""

    def test_ability_outranks_the_generic_band_with_munkidori(self) -> None:
        obs = self._mutated(my_bench=[{"id": MUNKIDORI,
                                       "energies": [DARK_ENERGY]}])
        option = ApiOption(type=OptionType.ABILITY)
        module = self.agent._main_score(obs, option)
        generic = self.generic._main_score(obs, option)
        self.assertGreaterEqual(module, self.theta["ability_band"])
        self.assertGreater(module, generic,
                           "generic parks every ability at 40.0")

    def test_ability_untouched_without_munkidori(self) -> None:
        """An ability we have no rule for keeps the generic band —
        this is the graceful-degradation half of the rule."""
        obs = self._mutated(my_active={"id": SNORUNT},
                            my_bench=[{"id": SNORUNT}])
        option = ApiOption(type=OptionType.ABILITY)
        self.assertEqual(self.agent._main_score(obs, option),
                         self.generic._main_score(obs, option))

    def test_live_counters_raise_the_ability_further(self) -> None:
        option = ApiOption(type=OptionType.ABILITY)
        idle = self._mutated(my_bench=[{"id": MUNKIDORI,
                                        "energies": [DARK_ENERGY]}],
                             my_active={"hp": 320, "maxHp": 320})
        live = self._mutated(my_bench=[{"id": MUNKIDORI,
                                        "energies": [DARK_ENERGY]}],
                             my_active={"hp": 200, "maxHp": 320})
        self.assertGreater(self.agent._main_score(live, option),
                           self.agent._main_score(idle, option),
                           "damage on our side is what Adrena-Brain moves")

    def test_ability_stays_under_the_attach_band(self) -> None:
        """Adrena-Brain only works with {D} attached, so attaching must
        sequence first. Enforced by the SCHEMA, not just the defaults —
        no reachable genome may invert it."""
        schema = GrimmsnarlModule.schema
        self.assertLess(schema.spec("ability_band").high
                        + schema.spec("adrena_brain_live_bonus").high, 55.0)
        self.assertLess(self.theta["ability_band"]
                        + self.theta["adrena_brain_live_bonus"], 55.0)


class TestGrimmsnarlCounterEngine(_GrimmsnarlCase):
    """Rule group 3: counters convert into prizes."""

    def _dest(self, obs, index: int, area=AreaType.BENCH) -> float:
        option = ApiOption(type=OptionType.CARD, area=area, index=index,
                           playerIndex=1 - obs.current.yourIndex)
        return self.agent._module._counter_destination(
            self.agent._view(obs), option)

    def test_counters_go_where_they_kill(self) -> None:
        obs = self._mutated(opp_bench=[{"hp": 20, "maxHp": 130},
                                       {"hp": 120, "maxHp": 130}])
        self.assertGreater(self._dest(obs, 0), self._dest(obs, 1),
                           "30 damage finishes the 20 HP body")

    def test_ties_broken_toward_damage_already_dealt(self) -> None:
        obs = self._mutated(opp_bench=[{"hp": 60, "maxHp": 130},
                                       {"hp": 130, "maxHp": 130}])
        self.assertGreater(self._dest(obs, 0), self._dest(obs, 1))

    def test_counters_never_go_to_a_benched_tera(self) -> None:
        # Tera on the bench takes 0 damage (test_tera_bench_immunity)
        tera_id = next(c.card_id for c in self.index.cards.values()
                       if c.is_tera and c.hp is not None)
        obs = self._mutated(opp_bench=[{"id": tera_id, "hp": 20, "maxHp": 200},
                                       {"hp": 130, "maxHp": 130}])
        self.assertLess(self._dest(obs, 0), self._dest(obs, 1),
                        "a lethal-looking Tera bench target is a trap")

    def test_source_prefers_the_big_ex(self) -> None:
        obs = self._mutated(my_active={"id": GRIMMSNARL_EX, "hp": 200,
                                       "maxHp": 320},
                            my_bench=[{"id": IMPIDIMP, "hp": 40,
                                       "maxHp": 70}])
        view = self.agent._view(obs)
        me = obs.current.yourIndex
        active = self.agent._module._counter_source(
            view, ApiOption(type=OptionType.CARD, area=AreaType.ACTIVE,
                            index=0, playerIndex=me))
        bench = self.agent._module._counter_source(
            view, ApiOption(type=OptionType.CARD, area=AreaType.BENCH,
                            index=0, playerIndex=me))
        self.assertGreater(active, bench,
                           "unload the 320 HP two-prize body first")


class TestGrimmsnarlTargeting(_GrimmsnarlCase):
    """Rule group 4: Boss's Orders drags a prize (contrast Crustle)."""

    def test_boss_is_dead_weight_without_a_target(self) -> None:
        obs = self._mutated(hand0_id=BOSS_ORDERS,
                            opp_bench=[{"hp": 330, "maxHp": 330}])
        self.assertEqual(self.agent._main_score(obs, self._play0()),
                         self.theta["boss_idle_score"])

    def test_boss_is_urgent_with_a_reachable_target(self) -> None:
        obs = self._mutated(hand0_id=BOSS_ORDERS,
                            opp_bench=[{"hp": 60, "maxHp": 130}])
        module = self.agent._main_score(obs, self._play0())
        self.assertEqual(module, self.theta["boss_live_score"])
        self.assertGreater(module, self.generic._main_score(obs, self._play0()))

    def test_drag_target_is_the_kill_not_the_wall(self) -> None:
        """The strategic inversion vs the Crustle module, which drags the
        heaviest UNPOWERED body to trap it. Here we drag what we can KO."""
        obs = self._mutated(my_active={"id": GRIMMSNARL_EX,
                                       "energies": [DARK_ENERGY] * 4},
                            opp_bench=[{"hp": 60, "maxHp": 130},
                                       {"hp": 320, "maxHp": 330}])
        them = 1 - obs.current.yourIndex
        view = self.agent._view(obs)
        killable = self.agent._module._drag_target_score(
            view, ApiOption(type=OptionType.CARD, area=AreaType.BENCH,
                            index=0, playerIndex=them))
        healthy = self.agent._module._drag_target_score(
            view, ApiOption(type=OptionType.CARD, area=AreaType.BENCH,
                            index=1, playerIndex=them))
        self.assertGreater(killable, healthy)

    def test_first_dark_energy_on_munkidori_is_prioritized(self) -> None:
        obs = self._mutated(my_active={"id": GRIMMSNARL_EX,
                                       "energies": [DARK_ENERGY]},
                            my_bench=[{"id": MUNKIDORI, "energies": []}])
        view = self.agent._view(obs)
        module = self.agent._module
        to_munkidori = module.attach_score(
            view, ApiOption(type=OptionType.ATTACH, area=AreaType.HAND,
                            index=0, inPlayArea=AreaType.BENCH,
                            inPlayIndex=0), 0.0)
        to_active = module.attach_score(
            view, ApiOption(type=OptionType.ATTACH, area=AreaType.HAND,
                            index=0, inPlayArea=AreaType.ACTIVE,
                            inPlayIndex=0), 0.0)
        self.assertGreater(to_munkidori, to_active,
                           "one {D} switches Adrena-Brain on for the game")


class TestGrimmsnarlSearch(_GrimmsnarlCase):
    """Rule group 5: search picks are board-aware."""

    def _search(self, obs, card_id: int) -> float:
        obs.current.players[obs.current.yourIndex].hand[0].id = card_id
        option = ApiOption(type=OptionType.CARD, area=AreaType.HAND,
                           index=0, playerIndex=obs.current.yourIndex)
        return self.agent._module._search_value(self.agent._view(obs), option)

    def test_grimmsnarl_is_the_pick_when_a_morgrem_waits(self) -> None:
        waiting = self._mutated(my_bench=[{"id": MORGREM}])
        empty = self._mutated(my_bench=[{"id": SNORUNT}])
        self.assertGreater(self._search(waiting, GRIMMSNARL_EX),
                           self._search(empty, GRIMMSNARL_EX))

    def test_basics_are_worth_less_once_we_have_them(self) -> None:
        have = self._mutated(my_active={"id": SNORUNT},
                             my_bench=[{"id": IMPIDIMP}, {"id": MUNKIDORI}])
        lack = self._mutated(my_active={"id": SNORUNT},
                             my_bench=[{"id": SNORUNT}])
        self.assertGreater(self._search(lack, IMPIDIMP),
                           self._search(have, IMPIDIMP))
        self.assertGreater(self._search(lack, MUNKIDORI),
                           self._search(have, MUNKIDORI))

    def test_rare_candy_outranks_a_generic_item(self) -> None:
        obs = self._mutated()
        self.assertGreater(self._search(obs, RARE_CANDY),
                           self._search(obs, POKEGEAR))


class TestGrimmsnarlRobustness(_GrimmsnarlCase):
    """The league property: safe on a deck the module never saw."""

    def test_legal_and_exception_free_on_a_foreign_deck(self) -> None:
        """Pilot the Crustle deck with the Grimmsnarl module: none of its
        card ids are present, so every rule must decline and the agent
        must still finish real games without a single exception."""
        crustle = read_deck_ids(DECKS_DIR / "candidate_crustle_e10.csv")
        spidops = read_deck_ids(DECKS_DIR / "meta_spidops.csv")
        decisions = 0
        for game_index in range(3):
            agent = ParametricHeuristicAgent(module=GrimmsnarlModule(),
                                             index=self.index,
                                             effects=self.effects)
            foil = HeuristicAgent(seed=game_index, index=self.index,
                                  effects=self.effects)
            obs_dict, _ = game.battle_start(list(crustle), list(spidops))
            self.assertIsNotNone(obs_dict)
            try:
                for _ in range(20_000):
                    state = obs_dict["current"]
                    if state["result"] != -1:
                        break
                    who = state["yourIndex"]
                    if who == 0:
                        answer = agent(obs_dict)
                        select = obs_dict.get("select")
                        if select is not None:
                            n = len(select.get("option") or [])
                            self.assertTrue(
                                all(isinstance(i, int) and 0 <= i < n
                                    for i in answer),
                                f"illegal answer {answer} for {n} options")
                            self.assertEqual(len(set(answer)), len(answer),
                                             "duplicate indices")
                            decisions += 1
                    else:
                        answer = foil(obs_dict)
                    obs_dict = game.battle_select(answer)
            finally:
                game.battle_finish()
        self.assertGreater(decisions, 100)

    def test_garbage_observations(self) -> None:
        for junk in ({"current": None}, {}, {"select": {"option": []}},
                     {"current": {"players": []}, "select": None}):
            self.assertIsInstance(self.agent(copy.deepcopy(junk)), list)

    def test_theta_bounds_preserve_the_development_invariant(self) -> None:
        """No reachable genome may let an ability outrank attaching, or
        Rare Candy outrank evolving — the Sprint-3 lesson, enforced by
        the schema rather than by rule code."""
        schema = GrimmsnarlModule.schema
        self.assertLess(schema.spec("ability_band").high
                        + schema.spec("adrena_brain_live_bonus").high, 55.0)
        self.assertLess(schema.spec("rare_candy_score").high, 80.0)
        self.assertLess(schema.spec("rebuild_score").high, 55.0)


if __name__ == "__main__":
    unittest.main()
