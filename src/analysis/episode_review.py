"""Review OUR ladder episodes: reproduce decisions, classify HOW we lost.

For each episode JSON in viewer/episodes/ (fetched by
fetch_my_episodes.py):

(a) find OUR seat via info.TeamNames;
(b) replay every one of OUR decision points through the SHIPPED pilot
    (CrustleAgent, --variant selects which ship; deterministic given
    the observation), capturing
    last_scores and checking it REPRODUCES the recorded action —
    reproduction fidelity is a GATE: a low rate means state/parsing
    divergence and any reconstructed reasoning would be untrustworthy;
(c) emit the viewer format (score overlay) for selected episodes;
(d) classify the LOSS MECHANISM — our win condition is mill, so how we
    lose is the diagnosis:
      self-deck-out   we emptied our own deck while the opponent's was
                      still healthy (anti-self-mill failed somewhere),
      out-milled      both decks low: control mirror, we lost the race,
      prize-race      the opponent took all prizes (wall pierced),
      board-wipe      we ran out of Pokémon in play,
      action-cap/draw result 2;
    and flag suspicious decisions: tiny score margins, attacking with
    damage when the mill attack was live, skipping an available heal
    under damage, self-thinning while losing the mill race.

Decision pairing reuses replays_parse._iter_decisions (prev-step
convention, verified on real episodes; empty-list answers are not
yielded there, so pass-only prompts fall outside the fidelity count).

Run from the repo root:
    python -m src.analysis.episode_review
    python -m src.analysis.episode_review --render 85316134 85314533
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from cg.api import OptionType

from ..agent_heuristics.crustle_agent import (JUMBO_ICE_CREAM, LOW_DECK,
                                              SELF_THINNERS, CrustleAgent)
from ..environment_wrapper.recorder import GameRecorder
from ..ingestion.build_card_model import REPO_ROOT
from ..ingestion.build_effect_model import EffectIndex
from ..ingestion.card_index import CardIndex
from ..ingestion.replays_parse import _iter_decisions
from .fetch_my_episodes import EPISODES_DIR

RENDER_DIR: Final[Path] = REPO_ROOT / "viewer" / "recordings"
OUR_TEAM: Final[str] = "Ilan Schapira"
CLOSE_MARGIN: Final[float] = 1.5


@dataclass
class EpisodeReview:
    episode_id: int
    our_index: int
    opponent: str
    result: str                      # win / loss / draw
    mechanism: str
    turns: int | None
    our_deck_left: int | None
    opp_deck_left: int | None
    our_prizes_left: int | None
    opp_prizes_left: int | None
    decisions: int = 0
    reproduced: int = 0
    mismatches: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    @property
    def fidelity(self) -> float:
        return self.reproduced / self.decisions if self.decisions else 1.0


def _our_seat(replay: dict, team: str) -> int | None:
    names = ((replay.get("info") or {}).get("TeamNames")) or []
    for i, name in enumerate(names):
        if name == team:
            return i
    return None


def _final_current(replay: dict) -> dict | None:
    """FRESHEST final engine state across both agents (None-safe).

    In the last step the two views can differ by one action: the loser's
    observation often predates the terminal KO, while the winner's shows
    it (e.g. our side already at 0 pokémon). Picking the first view here
    used to misclassify real board-wipes as "unknown", so within the
    last step that has any state we take the view with the highest turn.
    """
    for step in reversed(replay.get("steps") or []):
        if not isinstance(step, list):
            continue
        candidates = []
        for entry in step:
            if not isinstance(entry, dict):
                continue
            current = (entry.get("observation") or {}).get("current")
            if isinstance(current, dict):
                candidates.append(current)
        if candidates:
            # tie-break on position: with equal turns the later entry is
            # the fresher view (the winner's, holding the terminal KO)
            return max(enumerate(candidates),
                       key=lambda ic: ((ic[1].get("turn") or 0), ic[0]))[1]
    return None


def _winner_index(replay: dict) -> int | None:
    rewards = replay.get("rewards")
    if isinstance(rewards, list) and len(rewards) == 2:
        if rewards[0] == rewards[1]:
            return None  # draw
        return 0 if rewards[0] > rewards[1] else 1
    return None


def _classify(current: dict | None, our_index: int,
              winner: int | None) -> tuple[str, str]:
    """(result, mechanism) from the final state (all None-safe)."""
    if winner is None:
        return "draw", "action-cap/draw"
    result = "win" if winner == our_index else "loss"
    if current is None:
        return result, "unknown (no final state)"
    players = current.get("players") or [{}, {}]
    ours = players[our_index] if our_index < len(players) else {}
    theirs = players[1 - our_index] if 1 - our_index < len(players) else {}
    our_deck = ours.get("deckCount")
    opp_deck = theirs.get("deckCount")
    loser = ours if result == "loss" else theirs
    winner_side = theirs if result == "loss" else ours

    if (loser.get("deckCount") or 0) == 0:
        other_deck = winner_side.get("deckCount")
        if other_deck is not None and other_deck <= LOW_DECK:
            return result, "out-milled (control mirror)"
        return result, ("self-deck-out" if result == "loss"
                        else "opponent decked out")
    if len(winner_side.get("prize") or [None]) == 0:
        return result, "prize-race (wall pierced)" if result == "loss" \
            else "prize-race win"
    active = loser.get("active") or []
    bench = loser.get("bench") or []
    loser_pokemon = len([p for p in list(active) + list(bench) if p])
    if loser_pokemon == 0:
        return result, "board-wipe (no pokémon left)"
    # The terminal action (the winner's last attack) frequently resolves
    # AFTER the freshest recorded observation, so infer it from how close
    # each win condition was: which one the unseen KO can still trigger.
    winner_prizes = len(winner_side.get("prize") or [None])
    if winner_prizes <= 1 and loser_pokemon > 1:
        return result, "prize-race (wall pierced)"
    if winner_prizes > 1 and loser_pokemon == 1:
        return result, "board-wipe (no pokémon left)"
    if winner_prizes <= 1 and loser_pokemon == 1:
        return result, "final-KO (wipe/prize ambíguo)"
    return result, "unknown"


def _flag_decision(agent: CrustleAgent, obs: Any, action: list[int],
                   turn: Any) -> list[str]:
    """Suspicious-decision heuristics over the RECORDED action."""
    flags: list[str] = []
    select = obs.select
    if select is None or not select.option:
        return flags
    options = select.option
    chosen = action[0] if action and 0 <= action[0] < len(options) else None
    if chosen is None:
        return flags
    state = obs.current
    me = state.players[state.yourIndex] if state is not None else None

    scores = agent.last_scores
    if scores and len(scores) == len(options):
        ranked = sorted(scores, reverse=True)
        if len(ranked) >= 2 and ranked[0] - ranked[1] < CLOSE_MARGIN:
            flags.append(f"t{turn}: margem pequena "
                         f"({ranked[0]:.1f} vs {ranked[1]:.1f})")

    picked = options[chosen]
    mill_ids = agent._mill_attack_ids
    if picked.type == OptionType.ATTACK and picked.attackId not in mill_ids:
        if any(o.type == OptionType.ATTACK and o.attackId in mill_ids
               for o in options):
            flags.append(f"t{turn}: atacou com dano tendo Land Collapse")
    if picked.type == OptionType.PLAY:
        card_id = agent._wrapper.resolve_card_id(obs, picked)
        if (card_id in SELF_THINNERS and me is not None
                and agent._losing_mill_race(obs)):
            flags.append(f"t{turn}: self-thinning perdendo a corrida "
                         f"(deck {me.deckCount})")
    if picked.type in (OptionType.END, OptionType.ATTACK):
        active = agent._my_active(obs)
        if (active is not None and active.maxHp
                and active.maxHp - active.hp >= 40):
            for i, option in enumerate(options):
                if (option.type == OptionType.PLAY and i != chosen
                        and agent._wrapper.resolve_card_id(obs, option)
                        == JUMBO_ICE_CREAM):
                    flags.append(f"t{turn}: não curou com {active.maxHp - active.hp} de dano")
                    break
    return flags


def review_episode(path: Path, agent: CrustleAgent, index: CardIndex,
                   team: str, render: bool) -> EpisodeReview | None:
    with open(path, encoding="utf-8") as fh:
        replay = json.load(fh)
    our_index = _our_seat(replay, team)
    if our_index is None:
        print(f"  {path.name}: nosso time não encontrado — pulado")
        return None
    names = ((replay.get("info") or {}).get("TeamNames")) or ["?", "?"]
    winner = _winner_index(replay)
    current = _final_current(replay)
    result, mechanism = _classify(current, our_index, winner)
    players = (current or {}).get("players") or [{}, {}]
    ours, theirs = players[our_index], players[1 - our_index]

    episode_id = (replay.get("info") or {}).get("EpisodeId")
    if not isinstance(episode_id, int):
        episode_id = int(path.stem) if path.stem.isdigit() else 0
    review = EpisodeReview(
        episode_id=episode_id,
        our_index=our_index,
        opponent=names[1 - our_index],
        result=result,
        mechanism=mechanism,
        turns=(current or {}).get("turn"),
        our_deck_left=ours.get("deckCount"),
        opp_deck_left=theirs.get("deckCount"),
        our_prizes_left=len(ours.get("prize") or []),
        opp_prizes_left=len(theirs.get("prize") or []),
    )

    recorder = GameRecorder(index, (names[0], names[1])) if render else None
    for agent_index, obs_dict, action in _iter_decisions(replay):
        scores = None
        if agent_index == our_index:
            answer = agent(dict(obs_dict))
            scores = agent.last_scores
            review.decisions += 1
            if answer == action or sorted(answer) == sorted(action):
                review.reproduced += 1
            else:
                turn = (obs_dict.get("current") or {}).get("turn")
                review.mismatches.append(
                    f"t{turn}: gravado={action} reproduzido={answer}")
            try:
                obs = agent._wrapper.parse(obs_dict)
                turn = (obs_dict.get("current") or {}).get("turn")
                review.flags.extend(_flag_decision(agent, obs, action, turn))
            except Exception:
                pass
        if recorder is not None:
            recorder.record_step(obs_dict, action, scores)
    if recorder is not None:
        RENDER_DIR.mkdir(parents=True, exist_ok=True)
        out = RENDER_DIR / f"ladder_{review.episode_id}.json"
        recorder.save(out, winner if winner is not None else 2,
                      review.turns or 0)
        print(f"  render: {out.relative_to(REPO_ROOT)}")
    return review


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-dir", type=Path, default=EPISODES_DIR)
    parser.add_argument("--team", type=str, default=OUR_TEAM)
    parser.add_argument("--render", type=int, nargs="*", default=[],
                        help="episode ids to emit for the viewer")
    parser.add_argument("--variant", choices=("v1", "v2", "v3"),
                        default="v3",
                        help="which shipped CrustleAgent to replay with "
                             "(must match the submission that played the "
                             "episodes, or fidelity drops)")
    args = parser.parse_args()

    index = CardIndex()
    effects = EffectIndex()
    agent = CrustleAgent(index=index, effects=effects,
                         variant=args.variant)

    paths = sorted(args.episodes_dir.glob("*.json"))
    if not paths:
        raise SystemExit(f"nenhum episódio em {args.episodes_dir} — rode "
                         f"python -m src.analysis.fetch_my_episodes antes")

    reviews: list[EpisodeReview] = []
    total_decisions = total_reproduced = 0
    for path in paths:
        review = review_episode(path, agent, index, args.team,
                                render=int(path.stem) in set(args.render))
        if review is None:
            continue
        reviews.append(review)
        total_decisions += review.decisions
        total_reproduced += review.reproduced
        print(f"{review.episode_id}  {review.result:4s}  "
              f"{review.mechanism:28s} turns={review.turns}  "
              f"deck nós/eles={review.our_deck_left}/{review.opp_deck_left}  "
              f"prizes nós/eles={review.our_prizes_left}/{review.opp_prizes_left}  "
              f"fidelidade {review.reproduced}/{review.decisions}  "
              f"vs {review.opponent}")
        for mismatch in review.mismatches[:4]:
            print(f"    MISMATCH {mismatch}")
        for flag in review.flags[:8]:
            print(f"    FLAG {flag}")

    fidelity = total_reproduced / total_decisions if total_decisions else 1.0
    print(f"\nfidelidade de reprodução GLOBAL: {total_reproduced}/"
          f"{total_decisions} = {fidelity:.2%}  (GATE: ~100%)")
    mechanisms: dict[str, int] = {}
    for review in reviews:
        if review.result == "loss":
            mechanisms[review.mechanism] = mechanisms.get(review.mechanism, 0) + 1
    print(f"mecanismos das derrotas: {mechanisms}")


if __name__ == "__main__":
    main()
