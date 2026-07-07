"""Typed convenience layer over the official cg engine observations.

EnvironmentWrapper never talks to the engine binary itself — it only
interprets the observation dicts the engine hands to agents, and joins
option references back to the CardIndex when one is available.
"""

from __future__ import annotations

from dataclasses import dataclass

from cg.api import (
    AreaType,
    Observation,
    Option,
    OptionType,
    SelectContext,
    SelectType,
    State,
    to_observation_class,
)
from cg.api import Card as ObsCard

from ..ingestion.card_index import Attack as IndexAttack
from ..ingestion.card_index import Card as IndexCard
from ..ingestion.card_index import CardIndex


@dataclass(frozen=True, slots=True)
class EnrichedOption:
    """An engine option joined with CardIndex data (None when unresolvable)."""

    option: Option
    card_id: int | None
    card: IndexCard | None
    attack: IndexAttack | None


class EnvironmentWrapper:
    """Parses raw observation dicts and decodes/enriches their options."""

    __slots__ = ("_index",)

    def __init__(self, index: CardIndex | None = None) -> None:
        self._index = index

    # ------------------------------------------------------------------ #
    # Parsing / basic reads
    # ------------------------------------------------------------------ #

    @staticmethod
    def parse(obs_dict: dict) -> Observation:
        return to_observation_class(obs_dict)

    @staticmethod
    def is_initial_selection(obs: Observation) -> bool:
        """True for the pre-game deck selection (agent must return 60 card ids)."""
        return obs.select is None

    @staticmethod
    def legal_option_indices(obs: Observation) -> list[int]:
        """Every index an agent may legally include in its answer."""
        if obs.select is None:
            return []
        return list(range(len(obs.select.option)))

    # ------------------------------------------------------------------ #
    # Option -> card resolution
    # ------------------------------------------------------------------ #

    @staticmethod
    def _card_at(
        state: State | None, player_index: int | None, area: AreaType | None, index: int | None
    ) -> ObsCard | None:
        """The observation Card an (area, index) reference points at, else None."""
        if state is None or area is None or index is None:
            return None
        try:
            if area == AreaType.STADIUM:
                return state.stadium[0]
            if area == AreaType.LOOKING:
                return state.looking[index] if state.looking else None
            if player_index is None:
                return None
            player = state.players[player_index]
            if area == AreaType.HAND:
                return player.hand[index] if player.hand is not None else None
            if area == AreaType.DISCARD:
                return player.discard[index]
            if area == AreaType.PRIZE:
                return player.prize[index]
            if area == AreaType.ACTIVE:
                pokemon = player.active[index]
                return ObsCard(pokemon.id, pokemon.serial, player_index) if pokemon else None
            if area == AreaType.BENCH:
                pokemon = player.bench[index]
                return ObsCard(pokemon.id, pokemon.serial, player_index) if pokemon else None
        except (IndexError, TypeError):
            return None
        return None  # DECK and other face-down areas are not resolvable

    def resolve_card_id(self, obs: Observation, option: Option) -> int | None:
        """Best-effort card id an option refers to (None if face-down/unknown)."""
        if option.cardId is not None:  # SKILL options carry it directly
            return option.cardId if option.cardId != 0 else None
        state = obs.current
        if option.type == OptionType.PLAY:
            # PLAY carries only a hand index; the hand belongs to the acting player.
            if state is None:
                return None
            card = self._card_at(state, state.yourIndex, AreaType.HAND, option.index)
            return card.id if card else None
        if option.type == OptionType.ATTACK:
            if self._index is None or option.attackId is None:
                return None
            attack = self._index.get_attack(option.attackId)
            return attack.card_id if attack else None
        card = self._card_at(state, option.playerIndex, option.area, option.index)
        return card.id if card else None

    def enrich(self, obs: Observation, option: Option) -> EnrichedOption:
        """Join an option with CardIndex data; every field falls back to None."""
        attack = None
        if self._index is not None and option.attackId is not None:
            attack = self._index.get_attack(option.attackId)
        card_id = self.resolve_card_id(obs, option)
        card = None
        if self._index is not None and card_id is not None:
            card = self._index.get_card(card_id)
        return EnrichedOption(option=option, card_id=card_id, card=card, attack=attack)

    # ------------------------------------------------------------------ #
    # Debug output
    # ------------------------------------------------------------------ #

    def option_summary(self, obs: Observation) -> list[str]:
        """One human-readable line per option, for logging/debugging."""
        if obs.select is None:
            return ["<initial deck selection: return 60 card ids>"]
        select = obs.select
        header = (
            f"{SelectType(select.type).name}/{SelectContext(select.context).name} "
            f"count[{select.minCount}..{select.maxCount}]"
        )
        lines = [header]
        for i, option in enumerate(select.option):
            parts = [f"[{i}] {OptionType(option.type).name}"]
            if option.area is not None:
                parts.append(f"area={AreaType(option.area).name}")
            for attr in ("index", "playerIndex", "number", "count",
                         "toolIndex", "energyIndex", "attackId", "serial"):
                value = getattr(option, attr)
                if value is not None:
                    parts.append(f"{attr}={value}")
            if option.inPlayArea is not None:
                parts.append(
                    f"inPlay={AreaType(option.inPlayArea).name}[{option.inPlayIndex}]"
                )
            if option.specialConditionType is not None:
                parts.append(f"condition={option.specialConditionType}")
            enriched = self.enrich(obs, option)
            if enriched.card is not None:
                parts.append(f"card={enriched.card.card_name!r}")
            elif enriched.card_id is not None:
                parts.append(f"cardId={enriched.card_id}")
            if enriched.attack is not None:
                parts.append(f"attack={enriched.attack.move_name!r}")
            lines.append(" ".join(parts))
        return lines
