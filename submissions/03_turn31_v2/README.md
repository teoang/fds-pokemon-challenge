# Submission 3 — turn-31 forecast, V2

## Approach

The original V2 pipeline uses a turn-31 forecast, a LightGBM/XGBoost/CatBoost stack and its
original prior-calibrated threshold.

## Files

`run.py`, `requirements.txt`, `source.json`, `reference.json`, `code/`, `original/` and `outputs/`
follow the same layout as the other two submissions. `original/notebook.ipynb` is the complete,
unchanged saved file. `code/` contains its nonempty model cells 1, 2, 3, 5, 7 and 10
(zero-based indices), preserved verbatim and listed in `source.json`.

## Run

```bash
python submissions/03_turn31_v2/run.py --data /path/to/competition --check
python submissions/03_turn31_v2/run.py --data /path/to/competition --out /path/to/output
```

## Reproduction status

Original Kaggle submission: **57675780**, public **0.69466**, private **0.67485**.
**The saved source is incomplete.** It has an empty cell and lacks the turn-level dataset
builder, `TURN_FEATURE_COLS`, stage-1 switch/next-type model training, `extract_multioutput_probs`
and `load_test_data`. It depended on definitions left in the original session.

The runner stops before training and lists those missing definitions. Recover the original
missing cells, preserve their exact code and update `source.json` before running this approach.
Code from V5 is not silently substituted for the missing V2 algorithm.
