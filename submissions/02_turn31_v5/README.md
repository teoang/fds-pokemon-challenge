# Submission 2 — turn-31 forecast, V5

## Approach

The original pipeline predicts the next switches and active types, then combines LightGBM,
XGBoost and CatBoost predictions using the original 80/12/8 CatBoost-dominant blend.

## Files

`run.py`, `requirements.txt`, `source.json`, `reference.json`, `code/`, `original/` and `outputs/`
follow the same layout as the other two submissions. `original/notebook.ipynb` is the complete,
unchanged notebook. `code/` contains verbatim original cells 2, 3, 4, 6, 8, 18 and 22
(zero-based indices), with their hashes and execution order recorded in `source.json`.

The old V3/V4 and exploratory optimizer cells remain archived but are not executed for the
final V5 submission. Only filesystem variables are rebound after the original configuration cell.

## Run

```bash
python submissions/02_turn31_v5/run.py --data /path/to/competition --check
python submissions/02_turn31_v5/run.py --data /path/to/competition --out /path/to/output
```

## Reproduction status

Original Kaggle submission: **57696039**, public **0.69666**, private **0.68314**.
A fresh execution of these final cells produced a valid 5,000-row CSV, with **43 predictions
different** from the downloaded original. The original runtime and any effect of the old
session state remain unresolved. The pinned dependencies describe the local check environment,
not a recovered original environment. Do not claim exact reproduction until the fingerprint matches.
