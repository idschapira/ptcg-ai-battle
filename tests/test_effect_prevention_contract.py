"""Empirical contract test: which energies stop a damage-COUNTER effect.

This is the fact the Rock-Fighting slot argument stands on, so it is
verified against the ENGINE, not against the card text and not against
our effect model.

Both Mist Energy (11) and Rock Fighting Energy (20) read "Prevent all
effects of attacks used by your opponent's Pokémon done to the Pokémon
this card is attached to. (Damage is not an effect.)" — but Rock Fighting
adds "…done to the {F} Pokémon this card is attached to". If the engine
honours that type restriction, then Rock Fighting is structurally dead on
our {G} wall (Crustle/Dwebble) while Mist covers everything.

Probe: Uxie's "Painful Memories" (attack 289, cost {P}, damage 0, "Put 2
damage counters on each of your opponent's Pokémon"). Placing damage
counters is an EFFECT, not damage, so a live prevention clause must zero
it out entirely — which makes the read unambiguous: HP either drops 20 or
does not move at all.

Matrix (one game per case, single variable = the attached energy):
    Great Tusk {F} + Mist            -> prevented
    Great Tusk {F} + Rock Fighting   -> prevented
    Great Tusk {F} + Basic {F}       -> NOT prevented  (control)
    Dwebble    {G} + Mist            -> prevented
    Dwebble    {G} + Rock Fighting   -> NOT prevented  (the type clause)

The engine has no seed API, so each case replays games until the
scenario materializes.

Run from the repo root:
    python -m unittest tests.test_effect_prevention_contract
"""

from __future__ import annotations

import unittest
from typing import Final

from cg import game
from cg.api import AreaType, OptionType, SelectContext

from src.environment_wrapper.wrapper import EnvironmentWrapper
from src.ingestion.card_index import CardIndex

UXIE: Final[int] = 215
PAINFUL_MEMORIES: Final[int] = 289
P_ENERGY: Final[int] = 5

GREAT_TUSK: Final[int] = 58        # {F}, Basic, HP 140
DWEBBLE: Final[int] = 344         # {G}, Basic, HP 70
ROCK_FIGHTING: Final[int] = 20
MIST: Final[int] = 11
BASIC_F: Final[int] = 6

COUNTERS_DAMAGE: Final[int] = 20  # 2 damage counters

ATTACKER_DECK: Final[list[int]] = [UXIE] * 4 + [P_ENERGY] * 56
DEFENDER_DECK: Final[list[int]] = (
    [GREAT_TUSK] * 4 + [DWEBBLE] * 4
    + [ROCK_FIGHTING] * 4 + [MIST] * 4
    + [BASIC_F] * 44
)

MAX_ATTEMPTS: Final[int] = 60
MAX_STEPS: Final[int] = 4000


class _Probe:
    """Drives a game to 'Uxie attacks a lone host holding one energy'."""

    def __init__(self, wrapper: EnvironmentWrapper, host: int,
                 energy: int) -> None:
        self._wrapper = wrapper
        self._host = host
        self._energy = energy

    def _defender_active(self, obs_dict: dict) -> dict | None:
        players = (obs_dict.get("current") or {}).get("players") or []
        if len(players) < 2:
            return None
        active = players[1].get("active") or []
        first = next(iter(active), None)
        return first if isinstance(first, dict) else None

    def _host_is_ready(self, obs_dict: dict) -> bool:
        """Host is active and holds EXACTLY the energy under test."""
        active = self._defender_active(obs_dict)
        if active is None or active.get("id") != self._host:
            return False
        attached = [c.get("id") for c in (active.get("energyCards") or [])
                    if isinstance(c, dict)]
        return attached == [self._energy]

    @staticmethod
    def _attach_ids(obs_dict: dict, options: list) -> list[int | None]:
        """Card id behind each option, resolving ATTACH from the hand.

        EnvironmentWrapper.resolve_card_id cannot do this: an ATTACH
        option carries area=HAND and an index but playerIndex=None, and
        the shared resolver refuses to guess whose hand it is. Here the
        acting player's hand is the only possibility.
        """
        current = obs_dict.get("current") or {}
        seat = current.get("yourIndex")
        players = current.get("players") or []
        hand = (players[seat].get("hand") or []
                if isinstance(seat, int) and seat < len(players) else [])
        out: list[int | None] = []
        for option in options:
            index = getattr(option, "index", None)
            if (getattr(option, "area", None) == int(AreaType.HAND)
                    and isinstance(index, int) and index < len(hand)
                    and isinstance(hand[index], dict)):
                out.append(hand[index].get("id"))
            else:
                out.append(None)
        return out

    def _choose(self, obs_dict: dict) -> tuple[list[int], bool]:
        obs = self._wrapper.parse(obs_dict)
        select = obs.select
        assert select is not None
        acting = obs.current.yourIndex
        options = select.option
        ctx = select.context
        ids = [resolved if resolved is not None
               else self._wrapper.resolve_card_id(obs, option)
               for option, resolved
               in zip(options, self._attach_ids(obs_dict, list(options)))]

        def pick(option_type: OptionType) -> int | None:
            return next((i for i, o in enumerate(options)
                         if o.type == option_type), None)

        if ctx == SelectContext.IS_FIRST:
            # attacker (0) goes first so it can attack on its second turn
            yes, no = pick(OptionType.YES), pick(OptionType.NO)
            return ([yes] if acting == 0 else [no]), False  # type: ignore[list-item]
        if ctx == SelectContext.MULLIGAN:
            return [pick(OptionType.NO) or 0], False
        if ctx == SelectContext.SETUP_ACTIVE_POKEMON:
            if acting == 1:
                for i, cid in enumerate(ids):
                    if cid == self._host:
                        return [i], False
            return [0], False
        if ctx == SelectContext.SETUP_BENCH_POKEMON:
            # keep both boards to a single Pokémon: no bench, so the
            # counters have exactly one place to land
            return ([] if select.minCount == 0
                    else list(range(select.minCount))), False
        if ctx == SelectContext.MAIN:
            if acting == 0:
                attack = next((i for i, o in enumerate(options)
                               if o.type == OptionType.ATTACK
                               and o.attackId == PAINFUL_MEMORIES), None)
                if attack is not None:
                    return [attack], True
                for i, option in enumerate(options):
                    if (option.type == OptionType.ATTACH
                            and ids[i] == P_ENERGY):
                        return [i], False
                return [pick(OptionType.END) or 0], False
            # defender: attach ONLY the energy under test, then pass
            active = self._defender_active(obs_dict)
            held = len((active or {}).get("energyCards") or [])
            if held == 0:
                for i, option in enumerate(options):
                    if (option.type == OptionType.ATTACH
                            and ids[i] == self._energy):
                        return [i], False
            return [pick(OptionType.END) or 0], False
        return list(range(select.minCount)), False

    def run(self) -> tuple[int, int] | None:
        """(hp_before, hp_after) around a Painful Memories on the host."""
        obs_dict, start = game.battle_start(list(ATTACKER_DECK),
                                            list(DEFENDER_DECK))
        if obs_dict is None:
            raise RuntimeError(f"battle_start failed: {start.errorType}")
        try:
            for _ in range(MAX_STEPS):
                if obs_dict["current"]["result"] != -1:
                    return None
                ready = self._host_is_ready(obs_dict)
                active = self._defender_active(obs_dict)
                before = (active or {}).get("hp")
                answer, is_attack = self._choose(obs_dict)
                obs_dict = game.battle_select(answer)
                if not (is_attack and ready and isinstance(before, int)):
                    continue
                active = self._defender_active(obs_dict)
                after = (active or {}).get("hp")
                if isinstance(after, int):
                    return before, after
                # host left the board (KO'd) -> treat as full damage
                return before, 0
            return None
        finally:
            game.battle_finish()


class TestEffectPreventionContract(unittest.TestCase):
    wrapper: EnvironmentWrapper

    @classmethod
    def setUpClass(cls) -> None:
        cls.wrapper = EnvironmentWrapper(CardIndex())

    def _damage(self, host: int, energy: int) -> int:
        probe = _Probe(self.wrapper, host, energy)
        for _ in range(MAX_ATTEMPTS):
            measured = probe.run()
            if measured is not None:
                before, after = measured
                return before - after
        self.fail(f"scenario (host={host}, energy={energy}) never "
                  f"materialized in {MAX_ATTEMPTS} games")

    # --- control: with a plain energy the counters land -----------------
    def test_plain_energy_does_not_prevent_counters(self) -> None:
        self.assertEqual(
            self._damage(GREAT_TUSK, BASIC_F), COUNTERS_DAMAGE,
            "Painful Memories must place its 2 counters when the host "
            "holds no prevention energy — otherwise this whole probe is "
            "measuring nothing")

    # --- Mist protects ANY host ----------------------------------------
    def test_mist_prevents_counters_on_fighting_host(self) -> None:
        self.assertEqual(self._damage(GREAT_TUSK, MIST), 0)

    def test_mist_prevents_counters_on_grass_host(self) -> None:
        self.assertEqual(
            self._damage(DWEBBLE, MIST), 0,
            "Mist Energy has no type restriction — it must cover the {G} "
            "line (Dwebble/Crustle) too")

    # --- Rock Fighting only protects a {F} host ------------------------
    def test_rock_fighting_prevents_counters_on_fighting_host(self) -> None:
        self.assertEqual(
            self._damage(GREAT_TUSK, ROCK_FIGHTING), 0,
            "Rock Fighting's clause must be LIVE on a {F} host — Great "
            "Tusk is our most common active, so this is the slot's whole "
            "justification")

    def test_rock_fighting_does_not_protect_grass_host(self) -> None:
        self.assertEqual(
            self._damage(DWEBBLE, ROCK_FIGHTING), COUNTERS_DAMAGE,
            "Rock Fighting says '…done to the {F} Pokémon this card is "
            "attached to' — on our {G} line the clause must be dead")


if __name__ == "__main__":
    unittest.main()
