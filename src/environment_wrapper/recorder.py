"""Dev-only game recorder: serializes each decision point to JSON.

The output feeds viewer/battle_viewer.html. Everything is decoded through
CardIndex (names) and option_summary (readable options) and is None-safe:
a missing id or field records as null, never raises.

Schema "ptcg-devrecord-v1":
{
  "schema": "ptcg-devrecord-v1",
  "players": ["HeuristicAgent", "RandomAgent"],
  "result": 0 | 1 | 2,          # winner index, 2 = draw
  "turns": <final turn count>,
  "steps": [
    {
      "step": int, "turn": int, "acting": 0|1,
      "select": {"type": str, "context": str, "min": int, "max": int},
      "options": [str, ...],     # option_summary lines (no header)
      "scores": [float, ...] | null,   # agent.last_scores when exposed
      "chosen": [int, ...],
      "logs": [str, ...],        # events since the previous selection
      "state": {
        "stadium": {"id": int, "name": str} | null,
        "players": [
          {"active": [pokemon...], "bench": [pokemon...],
           "deckCount": int, "handCount": int, "prizeLeft": int,
           "discardCount": int, "hand": [card...] | null,
           "conditions": {"poisoned": bool, ...}},
          ...
        ]
      }
    }, ...
  ]
}
pokemon = {"id", "name", "hp", "maxHp", "energies": [code...], "tools": int}
card    = {"id", "name"}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

from cg.api import LogType, Observation

from ..environment_wrapper.wrapper import EnvironmentWrapper
from ..ingestion.card_index import CardIndex

SCHEMA: Final[str] = "ptcg-devrecord-v1"


class GameRecorder:
    """Accumulates decoded decision points for one game and saves JSON."""

    __slots__ = ("_index", "_wrapper", "_players", "_steps", "_step_counter")

    def __init__(self, index: CardIndex, player_names: tuple[str, str]) -> None:
        self._index = index
        self._wrapper = EnvironmentWrapper(index)
        self._players = list(player_names)
        self._steps: list[dict[str, Any]] = []
        self._step_counter = 0

    # ------------------------------------------------------------------ #
    # Decoding helpers (raw-dict based, defensive everywhere)
    # ------------------------------------------------------------------ #

    def _name_of(self, card_id: Any) -> str | None:
        if not isinstance(card_id, int):
            return None
        card = self._index.get_card(card_id)
        return card.card_name if card is not None else None

    def _pokemon_snapshot(self, pokemon: dict | None) -> dict[str, Any] | None:
        if not isinstance(pokemon, dict):
            return None
        return {
            "id": pokemon.get("id"),
            "name": self._name_of(pokemon.get("id")),
            "hp": pokemon.get("hp"),
            "maxHp": pokemon.get("maxHp"),
            "energies": pokemon.get("energies") or [],
            "tools": len(pokemon.get("tools") or []),
        }

    def _card_snapshot(self, card: dict | None) -> dict[str, Any] | None:
        if not isinstance(card, dict):
            return None
        return {"id": card.get("id"), "name": self._name_of(card.get("id"))}

    def _player_snapshot(self, player: dict) -> dict[str, Any]:
        hand = player.get("hand")
        return {
            "active": [self._pokemon_snapshot(p) for p in (player.get("active") or [])],
            "bench": [self._pokemon_snapshot(p) for p in (player.get("bench") or [])],
            "deckCount": player.get("deckCount"),
            "handCount": player.get("handCount"),
            "prizeLeft": len(player.get("prize") or []),
            "discardCount": len(player.get("discard") or []),
            "hand": ([self._card_snapshot(c) for c in hand]
                     if isinstance(hand, list) else None),
            "conditions": {
                condition: bool(player.get(condition))
                for condition in ("poisoned", "burned", "asleep", "paralyzed", "confused")
            },
        }

    def _state_snapshot(self, state: dict | None) -> dict[str, Any] | None:
        if not isinstance(state, dict):
            return None
        stadium_cards = state.get("stadium") or []
        stadium = self._card_snapshot(stadium_cards[0]) if stadium_cards else None
        return {
            "stadium": stadium,
            "players": [self._player_snapshot(p) for p in (state.get("players") or [])],
        }

    def _decode_log(self, log: dict) -> str:
        try:
            name = LogType(log.get("type", -1)).name
        except ValueError:
            name = f"LOG_{log.get('type')}"
        parts = [name]
        for key in ("playerIndex", "value", "result", "reason", "head"):
            if log.get(key) is not None:
                parts.append(f"{key}={log[key]}")
        for key in ("cardId", "cardIdTarget", "cardIdActive", "cardIdBench",
                    "cardIdBefore", "cardIdAfter"):
            if log.get(key) is not None:
                card_name = self._name_of(log[key])
                parts.append(f"{key.removeprefix('cardId') or 'card'}="
                             f"{card_name or log[key]}")
        if log.get("attackId") is not None:
            attack = self._index.get_attack(log["attackId"])
            parts.append(f"attack={attack.move_name if attack else log['attackId']}")
        return " ".join(str(p) for p in parts)

    # ------------------------------------------------------------------ #
    # Recording
    # ------------------------------------------------------------------ #

    def record_step(self, obs_dict: dict,
                    answer: list[int], scores: list[float] | None) -> None:
        try:
            obs: Observation = self._wrapper.parse(obs_dict)
            summary = self._wrapper.option_summary(obs)
            header, option_lines = (summary[0], summary[1:]) if summary else ("", [])
            select = obs_dict.get("select") or {}
            state = obs_dict.get("current") or {}
            select_parts = header.split(" ") if header else []
            type_context = (select_parts[0].split("/") + ["?", "?"])[:2]
            self._steps.append({
                "step": self._step_counter,
                "turn": state.get("turn"),
                "acting": state.get("yourIndex"),
                "select": {
                    "type": type_context[0],
                    "context": type_context[1],
                    "min": select.get("minCount"),
                    "max": select.get("maxCount"),
                },
                "options": option_lines,
                "scores": list(scores) if scores is not None else None,
                "chosen": list(answer),
                "logs": [self._decode_log(log) for log in (obs_dict.get("logs") or [])
                         if isinstance(log, dict)],
                "state": self._state_snapshot(state),
            })
        except Exception as exc:  # recording must never break a game
            self._steps.append({"step": self._step_counter,
                                "error": f"{type(exc).__name__}: {exc}"})
        finally:
            self._step_counter += 1

    def finish(self, result: int, turns: int) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "players": self._players,
            "result": result,
            "turns": turns,
            "steps": self._steps,
        }

    def save(self, path: Path, result: int, turns: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.finish(result, turns), fh, ensure_ascii=False, indent=1)
