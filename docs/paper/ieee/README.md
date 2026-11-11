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
| `fig_denial.png`       | `data/exp24_denial.json` (`python -m scripts.paper_additions`) |

## Baselines, response and compute-time results

Tables III and V, Fig. 5 and the compute-time paragraph come from four
standalone experiments (not in `scripts.run_all`; exp23 trains an LSTM per
sensor and needs `torch`, about 30 minutes on 4 cores):

```bash
python -m experiments.exp22_id_baselines
python -m experiments.exp23_prediction_baselines
python -m experiments.exp24_denial
python -m experiments.exp25_runtime
python -m scripts.paper_additions    # copies JSON to data/, renders fig_denial.png
```

`fig_methodology.png` is no longer included by the paper (Algorithm 1 carries
the per-slot loop); the file is kept so it can be restored.
