"""Contrafactual de piloto (semente do v4) via cg.search_begin.

Pergunta: nas derrotas residuais do e10, existia LINHA LEGAL alternativa
que teria virado o jogo? Não é caça de misplay conhecido (deu zero em
1699 decisões) — é avaliação contrafactual de duas hipóteses:

  (a) board-wipe / over-commitment: segurar a peça de board na mão
      (não baixar/evoluir para dentro do wipe) vira o jogo?
  (b) near-miss de mill (deck do oponente <=5 na derrota): existe
      timing de cura/retreat/ataque/END que compra o turno que falta?

Método: o motor é NÃO-reproduzível, então as derrotas do A/B interno
não podem ser rematerializadas — o modo `capture` joga um lote FRESCO
de e10 (CrustleAgent v3) vs o campo (HeuristicAgent, como no A/B),
guardando as obs cruas (com search_begin_input) das últimas decisões
NOSSAS de cada derrota + o JSON do viewer. O modo `analyze` determiniza
os ocultos com o MULTISET EXATO (offline conhecemos as duas decklists;
deck/prêmio/mão são amostrados por rollout) e compara, do MESMO root
(cg.search_begin + search_step ramificado = pareado por determinização),
a ação real vs alternativas, jogando cada ramo até o fim com os MESMOS
pilotos. Saída = taxa de vitória por ramo, nunca desfecho único (info
imperfeita + estocástico).

Honestidade: campo interno != ladder; isto GERA hipótese de fix v4 —
a confirmação vem das derrotas reais do e10 (fetch_my_episodes).

Rodar da raiz do repo (dev offline):
    python -m src.analysis.counterfactual capture --games 200
    python -m src.analysis.counterfactual analyze --samples 12
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Final

from cg import api

from ..agent_heuristics.crustle_agent import CrustleAgent
from ..agent_heuristics.heuristic_agent import HeuristicAgent
from ..deckbuilding.energy_ab import field_decks
from ..deckbuilding.legality import read_deck_ids
from ..environment_wrapper.recorder import GameRecorder
from ..environment_wrapper.selfplay import RESULT_DRAW, play_one_game
from ..ingestion.build_card_model import REPO_ROOT
from ..ingestion.build_effect_model import EffectIndex
from ..ingestion.card_index import CardIndex
from .episode_review import _classify

OUR_DECK_PATH: Final[Path] = (REPO_ROOT / "data" / "decks"
                              / "candidate_crustle_e10.csv")
LOSS_DIR: Final[Path] = REPO_ROOT / "data" / "processed" / "counterfactual"
RENDER_DIR: Final[Path] = REPO_ROOT / "viewer" / "recordings"

MAX_FULL_DECISIONS: Final[int] = 12   # obs cruas guardadas por derrota
NEAR_MISS_OPP_DECK: Final[int] = 5
STAGE_BASIC: Final[int] = 7

OPT_PLAY: Final[int] = 7
OPT_EVOLVE: Final[int] = 9
OPT_RETREAT: Final[int] = 12
OPT_ATTACK: Final[int] = 13
OPT_END: Final[int] = 14
JUMBO: Final[int] = 1147
SWITCH: Final[int] = 1123


# ------------------------------------------------------------------ #
# capture
# ------------------------------------------------------------------ #


class CaptureRecorder:
    """Recorder p/ play_one_game: obs cruas das NOSSAS decisões + viewer."""

    def __init__(self, our_seat: int, viewer: GameRecorder | None) -> None:
        self.our_seat = our_seat
        self.viewer = viewer
        self.decisions: list[dict] = []
        self.final_snapshot: dict | None = None

    def record_step(self, obs_dict: dict, answer: list[int],
                    scores: Any = None) -> None:
        current = obs_dict.get("current")
        if isinstance(current, dict):
            self.final_snapshot = _snapshot(current)
            if current.get("yourIndex") == self.our_seat:
                self.decisions.append({
                    "obs": copy.deepcopy(obs_dict),
                    "action": list(answer),
                    "scores": list(scores) if scores else None,
                })
        if self.viewer is not None:
            self.viewer.record_step(obs_dict, answer, scores)


def _snapshot(current: dict) -> dict:
    """Cópia mínima do estado p/ classificação (formato de _classify)."""
    players = []
    for ps in current.get("players") or []:
        players.append({
            "deckCount": ps.get("deckCount"),
            "prize": [None] * len(ps.get("prize") or []),
            "active": [1 for p in ps.get("active") or [] if p],
            "bench": [1 for p in ps.get("bench") or [] if p],
        })
    return {"turn": current.get("turn"), "players": players}


def run_capture(n_games: int, seed: int, index: CardIndex,
                effects: EffectIndex) -> None:
    our_deck = read_deck_ids(OUR_DECK_PATH)
    field_ = {name: read_deck_ids(path)
              for name, path in field_decks({OUR_DECK_PATH.resolve()}).items()}
    LOSS_DIR.mkdir(parents=True, exist_ok=True)
    RENDER_DIR.mkdir(parents=True, exist_ok=True)

    saved = games = exceptions = 0
    t0 = time.perf_counter()
    for opp_name, opp_deck in field_.items():
        for game_index in range(n_games):
            game_seed = seed + game_index
            our_seat = game_index % 2
            ours = CrustleAgent(seed=game_seed, index=index,
                                effects=effects, variant="v3")
            theirs = HeuristicAgent(seed=game_seed + 10_000, index=index,
                                    effects=effects)
            names = ["", ""]
            names[our_seat] = "crustle-v3(e10)"
            names[1 - our_seat] = opp_name
            recorder = CaptureRecorder(
                our_seat, GameRecorder(index, (names[0], names[1])))
            agents = ((ours, theirs) if our_seat == 0 else (theirs, ours))
            decks = ((our_deck, opp_deck) if our_seat == 0
                     else (opp_deck, our_deck))
            games += 1
            try:
                winner, turns = play_one_game(agents, list(decks[0]),
                                              list(decks[1]), recorder)
            except Exception as exc:  # noqa: BLE001 — gate de exceções
                exceptions += 1
                print(f"EXCEÇÃO {opp_name} game {game_index}: {exc}")
                continue
            if winner == RESULT_DRAW or winner == our_seat:
                continue
            result, mechanism = _classify(recorder.final_snapshot,
                                          our_seat, winner)
            snap = recorder.final_snapshot or {"players": [{}, {}]}
            opp_state = snap["players"][1 - our_seat]
            tag = f"{opp_name}_{game_seed}"
            trimmed = []
            for i, dec in enumerate(recorder.decisions):
                if i >= len(recorder.decisions) - MAX_FULL_DECISIONS:
                    trimmed.append(dec)
                else:
                    obs = dec["obs"]
                    trimmed.append({
                        "turn": (obs.get("current") or {}).get("turn"),
                        "action": dec["action"],
                    })
            loss = {
                "tag": tag, "opponent": opp_name, "game_seed": game_seed,
                "our_seat": our_seat, "mechanism": mechanism,
                "turns": turns, "opp_deck_final": opp_state.get("deckCount"),
                "opp_deck_ids": opp_deck, "decisions": trimmed,
            }
            with open(LOSS_DIR / f"loss_{tag}.json", "w",
                      encoding="utf-8") as fh:
                json.dump(loss, fh)
            recorder.viewer.save(RENDER_DIR / f"cf_{tag}.json",
                                 winner, turns)
            saved += 1
        print(f"{opp_name}: {games} jogos acumulados, {saved} derrotas "
              f"salvas ({time.perf_counter() - t0:.0f}s)")
    print(f"\ncapture: {games} jogos, {saved} derrotas, "
          f"{exceptions} exceções (deve ser 0)")


# ------------------------------------------------------------------ #
# determinização (multiset exato dos ocultos)
# ------------------------------------------------------------------ #


def _visible_ids(state: dict, player: int) -> Counter:
    seen: Counter = Counter()

    def add_card(card: dict | None) -> None:
        if card and card.get("playerIndex") == player:
            seen[card["id"]] += 1

    def add_pokemon(pokemon: dict | None, zone_owner: int) -> None:
        if not pokemon:
            return
        if zone_owner == player:
            seen[pokemon["id"]] += 1
        for key in ("energyCards", "tools", "preEvolution"):
            for card in pokemon.get(key) or []:
                add_card(card)

    for zone_owner, ps in enumerate(state.get("players") or []):
        for pokemon in ps.get("active") or []:
            add_pokemon(pokemon, zone_owner)
        for pokemon in ps.get("bench") or []:
            add_pokemon(pokemon, zone_owner)
        for card in ps.get("discard") or []:
            add_card(card)
        for card in ps.get("prize") or []:
            if card is not None:
                add_card(card)
        for card in ps.get("hand") or []:
            add_card(card)
    for card in state.get("stadium") or []:
        add_card(card)
    for card in state.get("looking") or []:
        if card is not None:
            add_card(card)
    return seen


def determinize(obs_dict: dict, our_seat: int, our_deck: list[int],
                opp_deck_ids: list[int],
                rng: random.Random) -> tuple | None:
    """Amostra (your_deck, your_prize, opp_deck, opp_prize, opp_hand).

    None se o multiset não fecha (zona não modelada) — o ponto é pulado
    e contado, nunca inventamos cartas.
    """
    state = obs_dict["current"]
    them = 1 - our_seat
    full = {our_seat: Counter(our_deck), them: Counter(opp_deck_ids)}
    pools: dict[int, list[int]] = {}
    for player in (our_seat, them):
        pool = full[player] - _visible_ids(state, player)
        pools[player] = [cid for cid, n in pool.items() for _ in range(n)]

    ps_us = state["players"][our_seat]
    ps_them = state["players"][them]
    hidden_prize_us = sum(1 for c in ps_us["prize"] if c is None)
    if len(pools[our_seat]) != ps_us["deckCount"] + hidden_prize_us:
        return None
    hidden_prize_them = sum(1 for c in ps_them["prize"] if c is None)
    expected = (ps_them["deckCount"] + hidden_prize_them
                + ps_them["handCount"])
    if len(pools[them]) != expected:
        return None

    rng.shuffle(pools[our_seat])
    rng.shuffle(pools[them])
    your_deck = pools[our_seat][:ps_us["deckCount"]]
    your_prize = ([c["id"] for c in ps_us["prize"] if c is not None]
                  + pools[our_seat][ps_us["deckCount"]:])
    opp_deck = pools[them][:ps_them["deckCount"]]
    cut = ps_them["deckCount"] + hidden_prize_them
    opp_prize = ([c["id"] for c in ps_them["prize"] if c is not None]
                 + pools[them][ps_them["deckCount"]:cut])
    opp_hand = pools[them][cut:]
    return your_deck, your_prize, opp_deck, opp_prize, opp_hand


# ------------------------------------------------------------------ #
# rollouts pareados
# ------------------------------------------------------------------ #


def _to_dict(observation: Any) -> dict:
    return json.loads(json.dumps(dataclasses.asdict(observation),
                                 default=int))


def rollout(branch: Any, our_seat: int, rollout_seed: int,
            index: CardIndex, effects: EffectIndex,
            cap: int = 600) -> int | None:
    """Joga um ramo até o terminal com os pilotos do matchup."""
    ours = CrustleAgent(seed=rollout_seed, index=index, effects=effects,
                        variant="v3")
    theirs = HeuristicAgent(seed=rollout_seed + 1, index=index,
                            effects=effects)
    state = branch
    for _ in range(cap):
        current = state.observation.current
        if current is not None and current.result != -1:
            return current.result
        obs_dict = _to_dict(state.observation)
        acting = obs_dict["current"]["yourIndex"]
        agent = ours if acting == our_seat else theirs
        state = api.search_step(state.searchId, agent(obs_dict))
    return None


def evaluate_point(obs_dict: dict, real_action: list[int],
                   alt_actions: dict[str, list[int]], our_seat: int,
                   our_deck: list[int], opp_deck_ids: list[int],
                   samples: int, index: CardIndex, effects: EffectIndex,
                   rng: random.Random) -> dict | None:
    """Taxas de vitória pareadas (mesma determinização) real vs alts."""
    wins = {label: 0 for label in ("real", *alt_actions)}
    valid = 0
    exceptions = 0
    obs_cls = api.to_observation_class(copy.deepcopy(obs_dict))
    for s in range(samples):
        det = determinize(obs_dict, our_seat, our_deck, opp_deck_ids, rng)
        if det is None:
            return None  # multiset não fecha neste ponto — pular
        try:
            root = api.search_begin(obs_cls, *det, [])
            branches = {"real": real_action, **alt_actions}
            for label, action in branches.items():
                branch = api.search_step(root.searchId, list(action))
                result = rollout(branch, our_seat, 1000 * s, index, effects)
                if result == our_seat:
                    wins[label] += 1
            valid += 1
        except Exception:  # noqa: BLE001 — contado, nunca propaga
            exceptions += 1
        finally:
            api.search_end()
    if valid == 0:
        return None
    return {"samples": valid, "exceptions": exceptions,
            "wins": wins}


# ------------------------------------------------------------------ #
# analyze
# ------------------------------------------------------------------ #


def _usable(obs_dict: dict) -> bool:
    sel = obs_dict.get("select") or {}
    state = obs_dict.get("current") or {}
    active_them = None
    players = state.get("players") or []
    your = state.get("yourIndex")
    if your is not None and len(players) == 2:
        active_them = (players[1 - your].get("active") or [None])
    return (obs_dict.get("search_begin_input") is not None
            and state.get("looking") is None
            and sel.get("deck") is None
            and sel.get("maxCount") == 1
            and not (active_them and active_them[0] is None
                     and len(active_them) > 0))


def _chosen_option(dec: dict) -> dict | None:
    options = ((dec["obs"].get("select") or {}).get("option")) or []
    action = dec.get("action") or []
    if action and 0 <= action[0] < len(options):
        return options[action[0]]
    return None


def _is_board_commit(option: dict, index: CardIndex) -> bool:
    """PLAY de Pokémon básico ou EVOLVE — peça de board saindo da mão."""
    opt_type = option.get("type")
    if opt_type == OPT_EVOLVE:
        return True
    if opt_type != OPT_PLAY:
        return False
    card = index.get_card(option.get("cardId") or -1)
    return card is not None and card.stage_code == STAGE_BASIC


def _score_alternatives(agent: CrustleAgent, obs_dict: dict,
                        allowed: list[int]) -> list[int]:
    """Índices `allowed` ordenados pelo score do piloto (desc)."""
    try:
        agent(copy.deepcopy(obs_dict))
        scores = agent.last_scores
    except Exception:  # noqa: BLE001
        scores = None
    options = ((obs_dict.get("select") or {}).get("option")) or []
    if scores and len(scores) == len(options):
        return sorted(allowed, key=lambda i: -scores[i])
    return allowed


def run_analyze(samples: int, seed: int, index: CardIndex,
                effects: EffectIndex, max_losses: int | None) -> None:
    our_deck = read_deck_ids(OUR_DECK_PATH)
    paths = sorted(LOSS_DIR.glob("loss_*.json"))
    if not paths:
        raise SystemExit(f"nenhuma derrota em {LOSS_DIR} — rode capture")
    losses = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    if max_losses:
        losses = losses[:max_losses]

    wipe = [l for l in losses if l["mechanism"].startswith("board-wipe")]
    near = [l for l in losses
            if (l.get("opp_deck_final") or 99) <= NEAR_MISS_OPP_DECK]
    print(f"derrotas: {len(losses)}  família wipe: {len(wipe)}  "
          f"família near-miss: {len(near)}  "
          f"(overlap: {sum(1 for l in near if l in wipe)})")

    # fidelidade: piloto reconstruído (mesma seed do capture) sobre as
    # obs salvas — valida a serialização/estado, não caça misplay.
    total = matched = 0
    for loss in losses:
        agent = CrustleAgent(seed=loss["game_seed"], index=index,
                             effects=effects, variant="v3")
        for dec in loss["decisions"]:
            if "obs" not in dec:
                continue
            answer = agent(copy.deepcopy(dec["obs"]))
            total += 1
            if answer == dec["action"] or sorted(answer) == sorted(dec["action"]):
                matched += 1
    print(f"fidelidade de reconstrução: {matched}/{total} = "
          f"{matched / max(total, 1):.2%}")

    scorer = CrustleAgent(seed=0, index=index, effects=effects,
                          variant="v3")
    rng = random.Random(seed)
    rows: list[dict] = []
    t0 = time.perf_counter()

    def counterfactual(loss: dict, family: str,
                       points: list[tuple[int, dict, dict[str, list[int]]]]):
        for dec_index, dec, alts in points:
            if not alts:
                continue
            result = evaluate_point(dec["obs"], dec["action"], alts,
                                    loss["our_seat"], our_deck,
                                    loss["opp_deck_ids"], samples,
                                    index, effects, rng)
            row = {"tag": loss["tag"], "family": family,
                   "opponent": loss["opponent"],
                   "mechanism": loss["mechanism"],
                   "turn": (dec["obs"].get("current") or {}).get("turn"),
                   "decision_index": dec_index,
                   "real_action": dec["action"],
                   "alts": {k: list(v) for k, v in alts.items()},
                   "result": result}
            rows.append(row)

    for loss in wipe:
        decisions = [d for d in loss["decisions"] if "obs" in d]
        points = []
        for i in range(len(decisions) - 1, -1, -1):
            dec = decisions[i]
            if not _usable(dec["obs"]):
                continue
            chosen = _chosen_option(dec)
            if chosen is None or not _is_board_commit(chosen, index):
                continue
            options = dec["obs"]["select"]["option"]
            allowed = [j for j, o in enumerate(options)
                       if j != dec["action"][0]
                       and not _is_board_commit(o, index)]
            ranked = _score_alternatives(scorer, dec["obs"], allowed)
            if ranked:
                points.append((i, dec, {"hold": [ranked[0]]}))
            if len(points) == 2:
                break
        counterfactual(loss, "wipe/over-commit", points)

    for loss in near:
        decisions = [d for d in loss["decisions"] if "obs" in d]
        points = []
        for i in range(len(decisions) - 1, max(len(decisions) - 4, -1), -1):
            dec = decisions[i]
            if not _usable(dec["obs"]):
                continue
            options = dec["obs"]["select"]["option"]
            allowed = []
            for j, option in enumerate(options):
                if j == dec["action"][0]:
                    continue
                opt_type = option.get("type")
                card_id = option.get("cardId")
                if (opt_type in (OPT_RETREAT, OPT_ATTACK, OPT_END)
                        or (opt_type == OPT_PLAY
                            and card_id in (JUMBO, SWITCH))):
                    allowed.append(j)
            ranked = _score_alternatives(scorer, dec["obs"], allowed)[:3]
            if ranked:
                points.append((i, dec, {f"alt{k}": [j]
                                        for k, j in enumerate(ranked)}))
        counterfactual(loss, "near-miss/stall", points)

    out_path = LOSS_DIR / "points.jsonl"
    with open(out_path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print(f"\n{len(rows)} pontos avaliados em "
          f"{time.perf_counter() - t0:.0f}s -> {out_path}")

    for family in ("wipe/over-commit", "near-miss/stall"):
        frows = [r for r in rows if r["family"] == family
                 and r["result"] is not None]
        skipped = sum(1 for r in rows
                      if r["family"] == family and r["result"] is None)
        print(f"\n=== {family} ===  pontos {len(frows)} "
              f"(pulados por determinização: {skipped})")
        by_tag: dict[str, list[dict]] = {}
        for row in frows:
            by_tag.setdefault(row["tag"], []).append(row)
        flips = 0
        for tag, trows in sorted(by_tag.items()):
            best = None
            for row in trows:
                res = row["result"]
                real = res["wins"]["real"]
                for label, w in res["wins"].items():
                    if label == "real":
                        continue
                    gain = w - real
                    if best is None or gain > best[0]:
                        best = (gain, w, real, res["samples"],
                                row["turn"], label)
            if best is None:
                continue
            gain, alt_w, real_w, n, turn, label = best
            has_line = gain >= max(2, n // 3) and alt_w >= n // 2
            flips += int(has_line)
            mark = "LINHA VIRA" if has_line else "sem linha"
            print(f"  {tag:28s} t{turn}: real {real_w}/{n} vs "
                  f"melhor-alt {alt_w}/{n} (ganho {gain:+d})  {mark}")
        n_losses = len(by_tag)
        print(f"  -> derrotas com linha que vira: {flips}/{n_losses}")
    exc = sum((r["result"] or {}).get("exceptions", 0) for r in rows)
    print(f"\nexceções em rollouts: {exc} (deve ser 0)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("capture", "analyze"))
    parser.add_argument("--games", type=int, default=200,
                        help="capture: jogos por pareamento")
    parser.add_argument("--samples", type=int, default=12,
                        help="analyze: determinizações por ponto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-losses", type=int, default=None)
    args = parser.parse_args()

    index = CardIndex()
    effects = EffectIndex()
    if args.command == "capture":
        run_capture(args.games, args.seed, index, effects)
    else:
        run_analyze(args.samples, args.seed, index, effects,
                    args.max_losses)


if __name__ == "__main__":
    main()
