"""Experiment harness for the conference-paper results (paper Section 11).

Each experiment script exposes ``compute(...)`` (runs the simulation, returns a
results dict), ``plot(results, path)`` (renders the figure), and ``main()``
(does both, saving JSON + PNG). Figures regenerate from the saved JSON, so a
re-render never re-simulates -- the reproducibility rule in Section 11.
"""
