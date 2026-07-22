"""Co-evolutionary league (Phase 1 — foundation, no mutation loop).

OFFLINE/dev only: nothing here is imported by main.py or the submission
package. The pieces:

- `theta.py`         the GENOME: a typed, bounded parameter vector.
- `board.py`         None-safe board reads shared by every deck module.
- `module.py`        the DeckModule contract (theta + obs -> score deltas).
- `parametric_agent.py`  ParametricHeuristicAgent: HeuristicAgent + module.
- `modules/`         one rules module per deck, plus deck->theta defaults.
- `fitness.py`       league fitness = spread vs the cohort, Wilson CIs.
- `hall_of_fame.py`  persisted champions, ready for the Phase 2 loop.
"""
