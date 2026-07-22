"""Co-evolutionary league: parameterized pilots + the loop that evolves them.

The package is SPLIT by what may be packaged into a submission.

RUNTIME (shippable — pure python over cg.api + ingestion.card_index; see
the `grimmsnarl_heur` target in src/build_submission.py):

- `theta.py`         the GENOME: a typed, bounded parameter vector.
- `board.py`         None-safe board reads shared by every deck module.
- `module.py`        the DeckModule contract (theta + obs -> score deltas).
- `parametric_agent.py`  ParametricHeuristicAgent: HeuristicAgent + module.
- `modules/`         one rules module per deck, plus deck->theta defaults.

DEV ONLY (never bundle — these import deckbuilding.gauntlet and
environment_wrapper.arena, which are not part of the submission):

- `fitness.py`       league fitness = spread vs the cohort, Wilson CIs.
- `hall_of_fame.py`  persisted champions, resumable across runs.
- `evolve.py`        the co-evolutionary mutation loop.
- `gate.py`          significance gate + successive halving.
- `portfolio.py`     the deck x pilot payoff matrix.
- `null_control.py`  the "is this signal or noise" control.
"""
