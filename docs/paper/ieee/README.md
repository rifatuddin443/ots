# IEEE conference paper

`ots_conference_paper.tex` — a 6-page, two-column IEEE conference paper
(`IEEEtran`, `conference` option) built from the novelty report
(`docs/novelty_report` / the audit) and the methodology and results write-up
(`docs/methodology_and_results.md`). Every number is from a re-run of the
experiment code in this repository; no value is carried over stale.

## Build

```bash
cd docs/paper/ieee
pdflatex ots_conference_paper.tex   # run 2-3x to settle refs
```

Needs `IEEEtran.cls` (included here), plus the usual `amsmath`, `algorithm`,
`algorithmic`, `booktabs`, `multirow`, `stfloats`, `graphicx`.

## Figures and data

Figures live in `figures/`. The two schematic figures and the three result
figures regenerate from the repository:

```bash
# from the repo root, with numpy + matplotlib installed
python -m experiments.paper_diagrams     # fig_architecture, fig_methodology
python -m scripts.run_all                # refresh results/*.json (all experiments)
python -m experiments.paper_extras \
    confusion mismatch predict conformal switch   # larger paper-only runs -> data/
python -m experiments.paper_figures      # fig_confusion, fig_results, fig_mismatch
```

`data/extra_*.json` holds the larger paper-only runs (per-sensor confusion over
100 seeds, the mis-specified-sensor sweep, prediction and conformal coverage at
the typical/contested sensors, and 24-plan switch tracking) so the result
figures re-render without re-simulating. The headline identification evaluation
alone aggregates 5,200 episodes; the coverage and mismatch runs add
~18,000 more.

| Figure | Source |
| --- | --- |
| `fig_architecture.png` | `experiments/paper_diagrams.py` |
| `fig_methodology.png`  | `experiments/paper_diagrams.py` |
| `fig_confusion.png`    | `data/extra_confusion.json` |
| `fig_results.png`      | `results/exp13,exp14,exp20` + `data/extra_predict.json` |
| `fig_mismatch.png`     | `data/extra_mismatch.json` |
