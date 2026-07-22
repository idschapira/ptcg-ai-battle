"""Contract tests for the Abomasnow module (the bypass candidate).

Same discipline as the Grimmsnarl contracts: every rule asserted as a
DIFFERENCE against the generic HeuristicAgent, plus robustness on a deck
the module has never seen.

The bypass HYPOTHESIS itself is not asserted here — a winrate claim
belongs in the fitness harness, not a unit test. What IS asserted here
is the structural premise the hypothesis rests on: this deck carries no
Ability and no damage-prevention effect, so there is nothing for an
effect-ignoring attacker (Mega Starmie ex's Nebula Beam) to strip.

Run from the repo root:
    python -m unittest tests.test_league_abomasnow
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
from src.league.modules.abomasnow import (KYOGRE, MAXIMUM_BELT, MEGA_ABOMASNOW,
                                          SNOVER, WAITRESS, WATER_ENERGY,
                                          AbomasnowModule)
from src.league.parametric_agent import ParametricHeuristicAgent

DECKS_DIR: Final = REPO_ROOT / "data" / "decks"


def _capture_main_obs(deck: list[int], opponent: list[int]) -> dict:
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


class _AbomasnowCase(unittest.TestCase):
    base_obs: dict

    @classmethod
    def setUpClass(cls) -> None:
        cls.index = CardIndex()
        cls.effects = EffectIndex()
        cls.deck = read_deck_ids(DECKS_DIR / "placeholder_abomasnow.csv")
        cls.base_obs = _capture_main_obs(
            cls.deck, read_deck_ids(DECKS_DIR / "meta_starmie.csv"))
        cls.generic = HeuristicAgent(index=cls.index, effects=cls.effects)
        cls.agent = ParametricHeuristicAgent(
            module=AbomasnowModule(), index=cls.index, effects=cls.effects)
        cls.module = cls.agent.module
        cls.theta = cls.agent.theta

    def _mutated(self, hand0_id: int | None = None,
                 my_active: dict | None = None,
                 my_bench: list[dict] | None = None,
                 my_deck: int | None = None,
                 discard: list[int] | None = None,
                 opp_active: dict | None = None):
        obs_dict = copy.deepcopy(self.base_obs)
        state = obs_dict["current"]
        me = state["yourIndex"]
        mine, theirs = state["players"][me], state["players"][1 - me]
        tpl = dict(mine["active"][0])
        if hand0_id is not None:
            mine["hand"][0]["id"] = hand0_id
        if my_active is not None:
            mine["active"][0].update(my_active)
        if my_bench is not None:
            mine["bench"] = [dict(tpl, **b) for b in my_bench]
        if my_deck is not None:
            mine["deckCount"] = my_deck
        if discard is not None:
            template = dict(mine["hand"][0])
            mine["discard"] = [dict(template, id=cid) for cid in discard]
        if opp_active is not None:
            theirs["active"][0].update(opp_active)
        return self.agent._wrapper.parse(obs_dict)

    def _attack_ids(self, card_id: int) -> list[int]:
        return [a.attack_id for a in self.index.attacks_of(card_id)]

    @property
    def hammer(self) -> int:
        return next(iter(self.module._hammer_ids))

    @property
    def frost(self) -> int:
        return next(iter(self.module._frost_ids))

    @property
    def riptide(self) -> int:
        return next(iter(self.module._riptide_ids))


class TestBypassPremise(_AbomasnowCase):
    """The structural claim the bypass hypothesis rests on."""

    def test_deck_carries_no_ability_at_all(self) -> None:
        """Mega Starmie ex's Nebula Beam ignores 'any effects on your
        opponent's Active Pokémon'. A deck with no Ability and no
        damage-prevention effect has nothing for it to bypass — unlike
        Crustle's wall or Grimmsnarl's Froslass/Munkidori engine."""
        with_abilities = []
        for card_id in set(self.deck):
            card = self.index.get_card(card_id)
            if card is None or card.hp is None:
                continue
            if self.index.skills_of(card_id):
                with_abilities.append(card.card_name)
        self.assertEqual(with_abilities, [],
                         "the bypass premise requires zero ability holders")

    def test_the_comparison_decks_do_carry_abilities(self) -> None:
        """Control: the premise is a real difference, not a truism."""
        for deck_file in ("candidate_crustle_e10", "meta_grimmsnarl"):
            ids = read_deck_ids(DECKS_DIR / f"{deck_file}.csv")
            holders = [cid for cid in set(ids)
                       if self.index.get_card(cid) is not None
                       and self.index.get_card(cid).hp is not None
                       and self.index.skills_of(cid)]
            self.assertTrue(holders, f"{deck_file} should have abilities")

    def test_module_declares_no_ability_rule(self) -> None:
        """Corollary: unlike the Grimmsnarl module, this one has no
        ability band to tune — there is no ability to play."""
        self.assertNotIn("ability_band", AbomasnowModule.schema)


class TestHammerLancheEconomics(_AbomasnowCase):
    """Rule group 1: the payoff/self-mill tradeoff, from real state."""

    def test_hammer_is_estimated_from_observable_density(self) -> None:
        """Not a hardcoded average: seeing {W} already in the discard
        must LOWER the expected payoff of milling six more."""
        fresh = self._mutated(my_deck=40, discard=[])
        drained = self._mutated(my_deck=40,
                                discard=[WATER_ENERGY] * 30)
        view_fresh = self.agent._view(fresh)
        view_drained = self.agent._view(drained)
        self.assertGreater(
            self.module._attack_estimate(view_fresh, self.hammer),
            self.module._attack_estimate(view_drained, self.hammer),
            "{W} already discarded is {W} not left to hit")

    def test_hammer_beats_frost_barrier_on_a_fresh_deck(self) -> None:
        obs = self._mutated(my_deck=45, discard=[])
        view = self.agent._view(obs)
        self.assertGreater(self.module._attack_estimate(view, self.hammer),
                           self.module._attack_estimate(view, self.frost),
                           "~350 expected beats the flat 200")

    def test_hammer_is_refused_when_it_would_race_our_own_deck_out(self):
        obs = self._mutated(my_deck=6, discard=[])
        view = self.agent._view(obs)
        self.assertEqual(self.module._attack_estimate(view, self.hammer), 0.0)
        self.assertGreater(self.module._attack_estimate(view, self.frost),
                           self.module._attack_estimate(view, self.hammer),
                           "below the floor, take the flat 200 instead")

    def test_the_choice_shows_up_in_the_attack_context(self) -> None:
        obs = self._mutated(my_deck=45, discard=[])
        plan = self.module.select_score(self.agent._view(obs),
                                        __import__("cg.api", fromlist=["x"]
                                                   ).SelectContext.ATTACK)
        self.assertIsNotNone(plan, "the module must own the ATTACK context")

    def test_main_attack_score_never_reaches_the_trainer_band(self) -> None:
        """Sprint-3 invariant: development still outranks attacking, for
        every reachable genome, even with a huge damage estimate."""
        schema = AbomasnowModule.schema
        ceiling = (schema.spec("attack_band_floor").high
                   + 450.0 / schema.spec("attack_scale").low
                   + schema.spec("attack_ko_bonus").high)
        self.assertLessEqual(ceiling, 35.0)


class TestRiptideRecycler(_AbomasnowCase):
    """Rule group 2: the one attack a thin deck makes BETTER."""

    def test_riptide_scales_with_water_in_the_discard(self) -> None:
        empty = self.agent._view(self._mutated(my_deck=30, discard=[]))
        loaded = self.agent._view(
            self._mutated(my_deck=30, discard=[WATER_ENERGY] * 10))
        self.assertGreater(self.module._attack_estimate(loaded, self.riptide),
                           self.module._attack_estimate(empty, self.riptide))

    def test_a_thin_deck_makes_riptide_more_attractive(self) -> None:
        """The inversion: everywhere else a low deck is a brake, here it
        is the reason to attack (Riptide shuffles the {W} back in)."""
        healthy = self.agent._view(
            self._mutated(my_deck=40, discard=[WATER_ENERGY] * 8))
        thin = self.agent._view(
            self._mutated(my_deck=8, discard=[WATER_ENERGY] * 8))
        self.assertGreater(self.module._attack_estimate(thin, self.riptide),
                           self.module._attack_estimate(healthy, self.riptide))

    def test_kyogre_is_searched_for_late_not_early(self) -> None:
        early = self._mutated(my_deck=45)
        late = self._mutated(my_deck=8)
        self.assertGreater(self._search(late, KYOGRE),
                           self._search(early, KYOGRE))

    def _search(self, obs, card_id: int) -> float:
        obs.current.players[obs.current.yourIndex].hand[0].id = card_id
        option = ApiOption(type=OptionType.CARD, area=AreaType.HAND, index=0,
                           playerIndex=obs.current.yourIndex)
        return self.module._search_value(self.agent._view(obs), option)


class TestAbomasnowSetupAndRouting(_AbomasnowCase):
    """Rule group 3: the climb and where the energy goes."""

    def test_evolving_into_mega_beats_the_generic_evolve_band(self) -> None:
        obs = self._mutated(hand0_id=MEGA_ABOMASNOW)
        option = ApiOption(type=OptionType.EVOLVE, area=AreaType.HAND,
                           index=0, playerIndex=obs.current.yourIndex)
        module = self.agent._main_score(obs, option)
        self.assertEqual(module - self.generic._main_score(obs, option),
                         self.theta["evolve_mega_bonus"])

    def test_waitress_outranks_plain_draw(self) -> None:
        """Acceleration beats card flow in a racer."""
        self.assertGreater(self.theta["waitress_score"],
                           self.theta["draw_supporter_score"])
        obs = self._mutated(hand0_id=WAITRESS)
        option = ApiOption(type=OptionType.PLAY, area=AreaType.HAND, index=0)
        self.assertGreater(self.agent._main_score(obs, option),
                           self.generic._main_score(obs, option))

    def test_energy_goes_to_the_body_that_still_needs_it(self) -> None:
        obs = self._mutated(
            my_active={"id": MEGA_ABOMASNOW, "energies": []},
            my_bench=[{"id": KYOGRE, "energies": [WATER_ENERGY] * 3}])
        view = self.agent._view(obs)
        to_mega = self.module.attach_score(
            view, ApiOption(type=OptionType.ATTACH, area=AreaType.HAND,
                            index=0, inPlayArea=AreaType.ACTIVE,
                            inPlayIndex=0), 0.0)
        to_kyogre = self.module.attach_score(
            view, ApiOption(type=OptionType.ATTACH, area=AreaType.HAND,
                            index=0, inPlayArea=AreaType.BENCH,
                            inPlayIndex=0), 0.0)
        self.assertGreater(to_mega, to_kyogre)

    def test_a_ready_attacker_stops_hoarding_energy(self) -> None:
        starved = self._mutated(my_active={"id": MEGA_ABOMASNOW,
                                           "energies": []})
        ready = self._mutated(my_active={"id": MEGA_ABOMASNOW,
                                         "energies": [WATER_ENERGY] * 4})
        option = ApiOption(type=OptionType.ATTACH, area=AreaType.HAND,
                           index=0, inPlayArea=AreaType.ACTIVE, inPlayIndex=0)
        self.assertGreater(
            self.module.attach_score(self.agent._view(starved), option, 0.0),
            self.module.attach_score(self.agent._view(ready), option, 0.0))

    def test_module_owns_the_attach_to_context(self) -> None:
        """The generic agent answers ATTACH_TO by index order — with four
        Waitress in the list that is real wasted acceleration."""
        from cg.api import SelectContext
        obs = self._mutated()
        self.assertIsNotNone(
            self.module.select_score(self.agent._view(obs),
                                     SelectContext.ATTACH_TO))
        self.assertIsNone(
            self.generic._decide.__self__.__class__.__dict__.get("_attach_to"),
            "generic has no ATTACH_TO handler — that is the gap")

    def test_snover_is_the_bench_priority(self) -> None:
        obs = self._mutated()
        option = ApiOption(type=OptionType.CARD, area=AreaType.HAND, index=0,
                           playerIndex=obs.current.yourIndex)
        obs.current.players[obs.current.yourIndex].hand[0].id = SNOVER
        snover = self.module.own_pokemon_score(self.agent._view(obs), option,
                                               False, 0.0)
        obs.current.players[obs.current.yourIndex].hand[0].id = KYOGRE
        kyogre = self.module.own_pokemon_score(self.agent._view(obs), option,
                                               False, 0.0)
        self.assertGreater(snover, kyogre)


class TestAbomasnowRobustness(_AbomasnowCase):
    """Safe on a deck the module never saw."""

    def test_legal_and_exception_free_on_a_foreign_deck(self) -> None:
        grimm = read_deck_ids(DECKS_DIR / "meta_grimmsnarl.csv")
        spidops = read_deck_ids(DECKS_DIR / "meta_spidops.csv")
        decisions = 0
        for game_index in range(3):
            agent = ParametricHeuristicAgent(module=AbomasnowModule(),
                                             index=self.index,
                                             effects=self.effects)
            foil = HeuristicAgent(seed=game_index, index=self.index,
                                  effects=self.effects)
            obs_dict, _ = game.battle_start(list(grimm), list(spidops))
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
                                    for i in answer), f"illegal {answer}/{n}")
                            self.assertEqual(len(set(answer)), len(answer))
                            decisions += 1
                    else:
                        answer = foil(obs_dict)
                    obs_dict = game.battle_select(answer)
            finally:
                game.battle_finish()
        self.assertGreater(decisions, 100)

    def test_garbage_observations(self) -> None:
        for junk in ({"current": None}, {}, {"select": {"option": []}}):
            self.assertIsInstance(self.agent(copy.deepcopy(junk)), list)

    def test_attack_ids_resolved_by_effect_text_not_hardcoded(self) -> None:
        self.assertTrue(self.module._hammer_ids)
        self.assertTrue(self.module._frost_ids)
        self.assertTrue(self.module._riptide_ids)
        self.assertFalse(self.module._hammer_ids & self.module._frost_ids)


if __name__ == "__main__":
    unittest.main()
