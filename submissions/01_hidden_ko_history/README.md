# Submission 1 — hidden knockouts and P1 history

## Approach

The original LightGBM pipeline corrects Player 2's hidden knockouts and adds Player 1's team
history to the state at turn 30, species, items and move counts. The original implementation
was moved from the repository root into `code/` without changing its contents.

## Files

`run.py`, `requirements.txt`, `source.json`, `reference.json`, `code/`, `original/` and `outputs/`
follow the same layout as the other two submissions. `original/` preserves version 4 of
`matteoangeloro/fds-pokemon-idea1`, which was a Python script.

## Run

```bash
python submissions/01_hidden_ko_history/run.py --data /path/to/competition --check
python submissions/01_hidden_ko_history/run.py --data /path/to/competition --out /path/to/output
```

The shared runner only adapts input/output paths and omits extra ablation fits. Model settings,
seeds, feature code and the final fit are unchanged. The legacy root entry points remain available.

## Submission reference

Original Kaggle submission: **57708941**, public **0.70000**, private **0.68314**.
`reference.json` records the original prediction fingerprint; each run checks it automatically.
