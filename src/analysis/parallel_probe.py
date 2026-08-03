"""Can the determinizations run in parallel — and is it SAFE to?

The rollouts inside one search are independent, we have 2 vCPUs on
Kaggle, and the search is single-threaded. That looks like free speedup.
Two things have to be true before it is:

SAFETY, which is the real question. ``cg.api`` keeps ONE module-level
``agent_ptr`` and ``search_end()`` frees every search allocated against
it, with no lock anywhere. So the obvious threading (each worker doing
its own search_begin/search_end) is a correctness bug, not a slow path:
one worker's ``search_end`` pulls the ground out from under the others.
The only shape worth testing is: open ALL the roots first, run the
rollouts concurrently on distinct ``searchId``s, and call
``search_end()`` exactly once at the end. Whether the C++ tolerates
concurrent ``SearchStep`` on distinct ids is not documented, so this
probe checks it empirically — and checks RESULTS, not just absence of a
crash, because silent state corruption is the failure mode that would
actually hurt.

SPEED, which only matters if safety holds. ctypes releases the GIL
around foreign calls, so threads CAN overlap engine work even though
Python is serialised; the rollout loop also does real Python work
(agent policies, _to_dict), which does not overlap. The measurement is
the honest arbiter.

This runs the risky part in a SUBPROCESS: a C++ crash takes the process
down, and it should not take the caller with it.

Run from the repo root:
    python -m src.analysis.parallel_probe --threads 2 --rollouts 16
"""

from __future__ import annotations

import argparse
import copy
import random
import statistics
import subprocess
import sys
import textwrap
import time
from pathlib import Path

PROBE = textwrap.dedent(
    '''
    import copy, json, random, statistics, sys, time
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    from cg import api, game as cg_game
    from src.agent_heuristics.crustle_agent import CrustleAgent
    from src.agent_heuristics.heuristic_agent import HeuristicAgent
    from src.agent_heuristics.random_agent import RandomAgent
    from src.deckbuilding.legality import read_deck_ids
    from src.ingestion.build_effect_model import EffectIndex
    from src.ingestion.card_index import CardIndex
    from src.rl_models.determinize import runtime_determinize
    from src.rl_models.search_core import rollout_to_terminal

    THREADS = int(sys.argv[1]); ROLLOUTS = int(sys.argv[2]); REPS = int(sys.argv[3])

    index, effects = CardIndex(), EffectIndex()
    our = read_deck_ids(Path("deck.csv"))
    opp = read_deck_ids(Path("data/decks/meta_grimmsnarl.csv"))

    # capture a real mid-game searchable observation
    ours = CrustleAgent(seed=1, index=index, effects=effects, variant="v3")
    theirs = RandomAgent(seed=2)
    obs_dict, _ = cg_game.battle_start(list(our), list(opp))
    sample = None
    try:
        for _ in range(200):
            cur = obs_dict["current"]
            if cur["result"] != -1: break
            if cur["yourIndex"] == 0:
                a = ours(copy.deepcopy(obs_dict))
                sel = (obs_dict.get("select") or {})
                if (obs_dict.get("search_begin_input") is not None
                        and len(sel.get("option") or []) > 2
                        and sel.get("maxCount") == 1
                        and (obs_dict["current"].get("looking") is None)):
                    sample = copy.deepcopy(obs_dict)
            else:
                a = theirs(obs_dict)
            obs_dict = cg_game.battle_select(a)
    finally:
        cg_game.battle_finish()
    if sample is None:
        print(json.dumps({"error": "no searchable observation captured"})); sys.exit(0)

    seat = sample["current"]["yourIndex"]
    n_opt = len(sample["select"]["option"])

    def open_roots(n, rng):
        """search_begin n times WITHOUT ending; returns list of roots."""
        obs_cls = api.to_observation_class(copy.deepcopy(sample))
        roots = []
        for _ in range(n):
            det = runtime_determinize(sample, seat, our, opp, rng)
            if det is None: continue
            roots.append(api.search_begin(obs_cls, *det, []))
        return roots

    def do_rollout(root, cand):
        branch = api.search_step(root.searchId, [cand])
        return rollout_to_terminal(
            branch, seat,
            CrustleAgent(seed=7, index=index, effects=effects, variant="v3"),
            HeuristicAgent(seed=8, index=index, effects=effects), 600)[0]

    def run(parallel):
        rng = random.Random(99)
        vals, t = [], None
        roots = open_roots(ROLLOUTS, rng)
        if not roots:
            return None, None
        jobs = [(roots[i % len(roots)], i % n_opt) for i in range(ROLLOUTS)]
        t0 = time.perf_counter()
        try:
            if parallel:
                with ThreadPoolExecutor(max_workers=THREADS) as ex:
                    vals = list(ex.map(lambda j: do_rollout(*j), jobs))
            else:
                vals = [do_rollout(*j) for j in jobs]
            t = time.perf_counter() - t0
        finally:
            api.search_end()
        return t, vals

    out = {"threads": THREADS, "rollouts": ROLLOUTS, "n_options": n_opt}
    try:
        seq_t, seq_v = [], None
        par_t, par_v = [], None
        for _ in range(REPS):
            t, v = run(False)
            if t is None: raise RuntimeError("determinize never closed")
            seq_t.append(t); seq_v = v
        for _ in range(REPS):
            t, v = run(True)
            par_t.append(t); par_v = v
        out["sequential_s"] = statistics.median(seq_t)
        out["parallel_s"] = statistics.median(par_t)
        out["speedup"] = statistics.median(seq_t) / statistics.median(par_t)
        # results must remain in the legal value set; corruption shows up
        # as values outside {0, 0.5, 1} or as a length mismatch
        out["values_sane"] = (
            len(par_v) == ROLLOUTS
            and all(v in (0.0, 0.5, 1.0) for v in par_v))
        out["parallel_values"] = par_v[:8]
        out["sequential_values"] = seq_v[:8]
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(out))
    '''
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=2,
                        help="Kaggle has 2 vCPUs; more is dev-box fantasy")
    parser.add_argument("--rollouts", type=int, default=16,
                        help="one 4x4 search's worth")
    parser.add_argument("--reps", type=int, default=3)
    args = parser.parse_args()

    print(f"probing threads={args.threads} rollouts={args.rollouts} "
          f"(subprocess: a C++ crash must not take us with it)")
    proc = subprocess.run(
        [sys.executable, "-c", PROBE, str(args.threads), str(args.rollouts),
         str(args.reps)],
        capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        print(f"  PROBE CRASHED (rc={proc.returncode}) — threads are unsafe "
              f"against this engine")
        print(proc.stderr.strip()[-1500:])
        return
    print(proc.stdout.strip())
    print(proc.stderr.strip()[-500:] if proc.stderr.strip() else "")


if __name__ == "__main__":
    main()
