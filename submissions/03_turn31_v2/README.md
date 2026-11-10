# Submission 3 — turn-31 forecast, V2

## Approach

The original V2 pipeline uses a turn-31 forecast, a LightGBM/XGBoost/CatBoost stack and its
original prior-calibrated threshold.

## Files

`run.py`, `requirements.txt`, `source.json`, `reference.json`, `code/`, `original/` and `outputs/`
follow the same layout as the other two submissions. `original/notebook.ipynb` preserves the
saved notebook. `code/` contains cells 1, 2, 3, 5, 6, 7 and 10 (zero-based indices).
`cell_06.py` supplies the turn-level dataset builder, Stage-1 switch/next-type models, feature
definitions and test loader from
[Lawrence's implementation](https://github.com/lawrencee7/fds-pokemon-challenge/commit/ee68bb6b3e885bed66961a6c0f564bab02a26c22).
`source.json` records the source provenance, checksums and execution order.

## Run

```bash
python submissions/03_turn31_v2/run.py --data /path/to/competition --check
python submissions/03_turn31_v2/run.py --data /path/to/competition --out /path/to/output
```

## Submission reference

Original Kaggle submission: **57675780**, public **0.69466**, private **0.67485**.
`reference.json` records the submitted prediction fingerprint. Each execution validates the
generated CSV and records the comparison in `run_manifest.json`.
