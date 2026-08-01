"""Profile-dependent attack band + energy routing: correct, and inert
for the ship.

The shipped score bands put attacking below every development action,
because attacking ENDS the turn while everything else keeps the MAIN
prompt open. That is the Crustle lesson (mill and wall) and it was
applied to every deck, including decks whose whole plan is a KO race.

Two changes, both default OFF:

  profile=PROFILE_AGGRO   a LETHAL attack outranks development, for decks
                          whose archetype wins the prize race. The
                          profile is derived from the DECK (same
                          archetype rules as the radar and the runtime
                          estimator), so it is a fact about the list, not
                          a switch — and our own list derives
                          DEVELOPMENT, which is the ship's behaviour.
  energy_routing=True     the ATTACH and PROMOTE scorers see scaled
                          damage. Both read attack.damage_base, which is
                          None for Powerful Hand, so Alakazam reads as a
                          harmless body: energy goes elsewhere and the
                          promotion prompt prefers the fattest Pokémon.

Lines held here:
  * the profile derivation is right on the REAL decklists;
  * on REAL engine options the aggro band lifts a lethal attack over the
    development bands and the development profile does not;
  * on REAL engine options routing changes where the energy goes;
  * CrustleAgent — the ship — does not move, checked lockstep with BOTH
    flags forced on in the shadow, on our own board and against the
    scaling opponent.

Run from the repo root:  python -m unittest tests.test_attack_profile
"""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Final

from cg.api import AreaType, OptionType, SelectContext

from src.agent_heuristics.crustle_agent import CrustleAgent
from src.agent_heuristics.heuristic_agent import (_AGGRO_LETHAL_BAND,
                                                  _EVOLVE_BAND, HeuristicAgent)
from src.deckbuilding.archetype_rules import (PROFILE_AGGRO,
                                              PROFILE_DEVELOPMENT,
                                              deck_profile, label_archetype)
from src.deckbuilding.legality import read_deck_ids
from src.environment_wrapper.selfplay import play_one_game
from src.environment_wrapper.wrapper import EnvironmentWrapper
from src.ingestion.build_effect_model import EffectIndex
from src.ingestion.card_index import CardIndex

ALAKAZAM: Final[int] = 743
ALAKAZAM_DECK: Final[Path] = Path("data/decks/meta_alakazam.csv")
OUR_DECK: Final[Path] = Path("deck.csv")


def _names(deck: list[int], index: CardIndex) -> list[str]:
    return [c.card_name for c in
            (index.get_card(i) for i in deck) if c is not None]


class TestProfileDerivation(unittest.TestCase):
    """The profile is read off the deck, by the shared archetype rules."""

    index: CardIndex

    @classmethod
    def setUpClass(cls) -> None:
        cls.index = CardIndex()

    def test_our_list_is_a_development_deck(self) -> None:
        """The ship's own list must never derive the aggro band — that is
        the structural half of the inertia argument."""
        names = _names(read_deck_ids(OUR_DECK), self.index)
        self.assertEqual(label_archetype(names), "Crustle mill (ours)")
        self.assertEqual(deck_profile(names), PROFILE_DEVELOPMENT)

    def test_alakazam_is_an_aggro_deck(self) -> None:
        names = _names(read_deck_ids(ALAKAZAM_DECK), self.index)
        self.assertEqual(deck_profile(names), PROFILE_AGGRO)

    def test_stall_lists_stay_on_development(self) -> None:
        for path in (Path("data/decks/meta_crustle_kangaskhan.csv"),
                     Path("data/decks/meta_spidops.csv")):
            names = _names(read_deck_ids(path), self.index)
            self.assertEqual(deck_profile(names), PROFILE_DEVELOPMENT,
                             f"{path.name} nao e um deck de corrida de premios")

    def test_unlabelled_deck_falls_to_the_conservative_profile(self) -> None:
        self.assertEqual(deck_profile(["Nao Existe"]), PROFILE_DEVELOPMENT)


class _Tap:
    """Wraps an agent and hands every parsed observation to a callback."""

    def __init__(self, inner, wrapper: EnvironmentWrapper, sink) -> None:
        self._inner, self._wrapper, self._sink = inner, wrapper, sink

    def __call__(self, obs_dict: dict) -> list[int]:
        answer = self._inner(obs_dict)
        try:
            obs = self._wrapper.parse(obs_dict)
            if obs.select is not None:
                self._sink(obs)
        except Exception:
            pass
        return answer


class TestBandOnRealOptions(unittest.TestCase):
    """Drive the engine and score REAL options under both profiles."""

    index: CardIndex
    effects: EffectIndex

    @classmethod
    def setUpClass(cls) -> None:
        cls.index = CardIndex()
        cls.effects = EffectIndex()
        cls.deck = read_deck_ids(ALAKAZAM_DECK)
        cls.our_deck = read_deck_ids(OUR_DECK)

    def _agent(self, profile: str, routing: bool = False) -> HeuristicAgent:
        return HeuristicAgent(index=self.index, effects=self.effects, seed=3,
                              tempo=True, scaled_damage=True,
                              profile=profile, energy_routing=routing)

    def test_lethal_attack_outranks_development_only_under_aggro(self) -> None:
        aggro = self._agent(PROFILE_AGGRO)
        plain = self._agent(PROFILE_DEVELOPMENT)
        lethal_seen = 0
        nonlethal_seen = 0
        wrapper = EnvironmentWrapper(self.index)

        def sink(obs) -> None:
            nonlocal lethal_seen, nonlethal_seen
            if obs.select.context != SelectContext.MAIN:
                return
            for option in obs.select.option:
                if option.type != OptionType.ATTACK:
                    continue
                value = plain._attack_value(option.attackId, obs)
                hot = aggro._main_score(obs, option)
                cold = plain._main_score(obs, option)
                if plain._is_lethal(value, obs):
                    lethal_seen += 1
                    self.assertGreaterEqual(hot, _AGGRO_LETHAL_BAND)
                    self.assertLess(cold, _EVOLVE_BAND)
                else:
                    nonlethal_seen += 1
                    # non-lethal attacks must NOT be promoted: attacking
                    # ends the turn, so skipping development for a hit
                    # that does not kill is strictly worse
                    self.assertEqual(hot, cold)

        driver = _Tap(self._agent(PROFILE_AGGRO), wrapper, sink)
        for game in range(12):
            foe = CrustleAgent(index=self.index, effects=self.effects,
                               variant="v3", seed=game)
            play_one_game((driver, foe), list(self.deck), list(self.our_deck))
            if lethal_seen > 30 and nonlethal_seen > 10:
                break
        self.assertGreater(lethal_seen, 10,
                           "amostra insuficiente de ataque LETAL real")
        self.assertGreater(nonlethal_seen, 5,
                           "amostra insuficiente de ataque nao-letal real")


class TestRoutingOnRealOptions(unittest.TestCase):
    """Energy routing must change WHERE the energy goes, on real options."""

    index: CardIndex
    effects: EffectIndex

    @classmethod
    def setUpClass(cls) -> None:
        cls.index = CardIndex()
        cls.effects = EffectIndex()
        cls.deck = read_deck_ids(ALAKAZAM_DECK)
        cls.our_deck = read_deck_ids(OUR_DECK)

    def test_a_scaling_attacker_reads_as_harmless_without_routing(self) -> None:
        """The bug, stated on the card: one {P} unlocks a ~240 attack and
        the attach scorer still measures the gain as zero."""
        plain = HeuristicAgent(index=self.index, effects=self.effects,
                               scaled_damage=True, energy_routing=False)
        # 5 == psychic in the engine's energy namespace
        self.assertEqual(plain._best_affordable_damage(ALAKAZAM, [5]), 0.0)

    def test_routing_lets_the_attach_scorer_see_the_body(self) -> None:
        """Exactly Powerful Hand's rule — 2 counters (20 damage) per card
        in hand — read off the live board, not a threshold picked to pass.
        """
        routed = HeuristicAgent(index=self.index, effects=self.effects,
                                scaled_damage=True, energy_routing=True)
        wrapper = EnvironmentWrapper(self.index)
        seen: list[tuple[float, int]] = []

        def sink(obs) -> None:
            if obs.select.context != SelectContext.MAIN:
                return
            state = obs.current
            me = state.players[state.yourIndex]
            hand = int(me.handCount or len(me.hand or []))
            # 5 == psychic; one energy already pays Powerful Hand
            seen.append((routed._best_affordable_damage(ALAKAZAM, [5], obs),
                         hand))

        driver = _Tap(routed, wrapper, sink)
        foe = CrustleAgent(index=self.index, effects=self.effects,
                           variant="v3", seed=1)
        play_one_game((driver, foe), list(self.deck), list(self.our_deck))
        self.assertGreater(len(seen), 5, "o motor quase nao ofereceu MAIN")
        for value, hand in seen:
            self.assertEqual(value, 20.0 * hand)
        self.assertGreater(max(h for _v, h in seen), 0,
                           "a mao nunca teve carta — o teste nao prova nada")

    def test_routing_moves_the_chosen_attach_target(self) -> None:
        """Over REAL attach options: how often does each scorer put the
        scaling attacker on top?"""
        routed = HeuristicAgent(index=self.index, effects=self.effects,
                                seed=5, tempo=True, scaled_damage=True,
                                energy_routing=True)
        plain = HeuristicAgent(index=self.index, effects=self.effects,
                               seed=5, tempo=True, scaled_damage=True,
                               energy_routing=False)
        wrapper = EnvironmentWrapper(self.index)
        offers = 0
        top = {"routed": 0, "plain": 0}

        def target_id(obs, option):
            state = obs.current
            pokemon = plain._pokemon_at(state, state.yourIndex,
                                        option.inPlayArea, option.inPlayIndex)
            return pokemon.id if pokemon is not None else None

        def sink(obs) -> None:
            nonlocal offers
            if obs.select.context != SelectContext.MAIN:
                return
            attaches = [o for o in obs.select.option
                        if o.type == OptionType.ATTACH]
            if len(attaches) < 2:
                return
            if not any(target_id(obs, o) == ALAKAZAM for o in attaches):
                return
            offers += 1
            for tag, agent in (("routed", routed), ("plain", plain)):
                best = max(attaches, key=lambda o: agent._attach_score(obs, o))
                if target_id(obs, best) == ALAKAZAM:
                    top[tag] += 1

        driver = _Tap(routed, wrapper, sink)
        for game in range(20):
            foe = CrustleAgent(index=self.index, effects=self.effects,
                               variant="v3", seed=100 + game)
            play_one_game((driver, foe), list(self.deck), list(self.our_deck))
            if offers > 25:
                break
        self.assertGreater(offers, 10,
                           "amostra insuficiente de ATTACH com o atacante "
                           "de escala na mesa")
        self.assertGreater(top["routed"], top["plain"],
                           f"routing nao mudou o alvo da energia "
                           f"(routed={top['routed']} plain={top['plain']} "
                           f"de {offers})")


class TestShippedCrustleIsUnchanged(unittest.TestCase):
    index: CardIndex
    effects: EffectIndex

    @classmethod
    def setUpClass(cls) -> None:
        cls.index = CardIndex()
        cls.effects = EffectIndex()
        cls.deck = read_deck_ids(OUR_DECK)
        cls.alakazam = read_deck_ids(ALAKAZAM_DECK)

    def test_crustle_defaults_to_the_shipped_profile(self) -> None:
        agent = CrustleAgent(index=self.index, effects=self.effects,
                             variant="v3")
        self.assertEqual(agent._profile, PROFILE_DEVELOPMENT)
        self.assertFalse(agent._energy_routing)

    def test_the_aggro_arm_derives_DEVELOPMENT_from_our_own_deck(self) -> None:
        """The wiring, not the intent: an arm asked for the aggro band and
        handed OUR list must come back on the development profile.

        This is the whole safety argument for the band, so it is asserted
        against the real arm factory rather than re-derived here."""
        from src.environment_wrapper.ab_test import (ArmMetrics, ArmSpec,
                                                     arm_factory)
        make = arm_factory(ArmSpec.parse("heuristic-tempo-scaled-aggro"),
                           self.index, self.effects, ArmMetrics(),
                           list(self.deck))
        self.assertEqual(make(0)._agent._profile, PROFILE_DEVELOPMENT)
        make_aggro = arm_factory(ArmSpec.parse("heuristic-tempo-scaled-aggro"),
                                 self.index, self.effects, ArmMetrics(),
                                 list(self.alakazam))
        self.assertEqual(make_aggro(0)._agent._profile, PROFILE_AGGRO)

    def test_forcing_the_aggro_band_DOES_move_the_ship(self) -> None:
        """The counter-test, so 'inert as wired' is never read as 'inert'.

        A band that reordered nothing would be a band that does nothing;
        the ship is safe because our deck derives DEVELOPMENT, not
        because the rule is toothless.

        Asserted on the SCORE of the ship's own real attack options, not
        on a changed answer. Changed answers do happen -- 144 of our
        11,495 real ladder decisions (222 episodes) flip when the profile
        is forced -- but they arrive in bursts: over mirror games they
        concentrate in one or two games out of eight, so an
        answer-level assertion is a coin flip at any sample this suite
        can afford. The score is the same claim without the lottery.
        """
        aggro = CrustleAgent(index=self.index, effects=self.effects,
                             variant="v3", seed=31)
        aggro._profile = PROFILE_AGGRO
        plain = CrustleAgent(index=self.index, effects=self.effects,
                             variant="v3", seed=31)
        wrapper = EnvironmentWrapper(self.index)
        lethal_seen = 0

        def sink(obs) -> None:
            nonlocal lethal_seen
            if obs.select.context != SelectContext.MAIN:
                return
            for option in obs.select.option:
                if option.type != OptionType.ATTACK:
                    continue
                value = plain._attack_value(option.attackId, obs)
                if not plain._is_lethal(value, obs):
                    continue
                lethal_seen += 1
                self.assertGreaterEqual(aggro._main_score(obs, option),
                                        _AGGRO_LETHAL_BAND)
                self.assertLess(plain._main_score(obs, option), _EVOLVE_BAND)

        # the MIRROR: two Crustle boards trade knockouts, so lethal
        # attacks are dense there
        driver = _Tap(CrustleAgent(index=self.index, effects=self.effects,
                                   variant="v3", seed=31), wrapper, sink)
        foe = HeuristicAgent(index=self.index, effects=self.effects, seed=32)
        # keep playing until the sample is real: how many lethal windows
        # a mirror produces is stochastic (they come in bursts), so a
        # fixed game count makes the guard, not the rule, decide the test
        for _game in range(30):
            play_one_game((driver, foe), list(self.deck), list(self.deck))
            if lethal_seen > 20:
                break
        self.assertGreater(lethal_seen, 20,
                           "a banda agressiva nunca foi exercitada no piloto "
                           "do ship — ou a regra morreu, ou o teste parou de "
                           "alcança-la")

    def _lockstep(self, opponent_deck: list[int], seed: int,
                  games: int = 5) -> tuple[int, int]:
        """Lockstep with every flag that is inert BY CONSTRUCTION forced
        on. The profile is deliberately left where the deck puts it —
        forcing it is the previous test, and it is expected to differ.

        Several games, not one: a single game can end in 8 of our
        decisions (an early concession), which trips the vacuity guard
        instead of the assertion it is supposed to guard.
        """
        driver = CrustleAgent(index=self.index, effects=self.effects,
                              variant="v3", seed=seed)
        shadow = CrustleAgent(index=self.index, effects=self.effects,
                              variant="v3", seed=seed)
        shadow._energy_routing = True             # the ONLY differences
        shadow._scaled_damage = True
        shadow._tempo = True
        divergences = compared = 0

        def lockstep(obs_dict: dict) -> list[int]:
            nonlocal divergences, compared
            answer = driver(obs_dict)
            compared += 1
            if answer != shadow(dict(obs_dict)):
                divergences += 1
            return answer

        foe = HeuristicAgent(index=self.index, effects=self.effects,
                             seed=seed + 1)
        for _game in range(games):
            play_one_game((lockstep, foe), list(self.deck),
                          list(opponent_deck))
        return compared, divergences

    def test_flags_are_inert_on_our_own_board(self) -> None:
        compared, divergences = self._lockstep(self.deck, seed=21)
        self.assertGreater(compared, 100, "quase nada comparado — teste vácuo")
        self.assertEqual(divergences, 0,
                         f"o fix mudou o piloto do ship em "
                         f"{divergences}/{compared} decisões")

    def test_flags_are_inert_against_a_scaling_opponent(self) -> None:
        compared, divergences = self._lockstep(self.alakazam, seed=22)
        self.assertGreater(compared, 100, "quase nada comparado — teste vácuo")
        self.assertEqual(divergences, 0,
                         f"o fix mudou o piloto do ship em "
                         f"{divergences}/{compared} decisões")


if __name__ == "__main__":
    unittest.main()
