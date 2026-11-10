# Three final submissions

[`final_submissions.ipynb`](../final_submissions.ipynb) is the single entry notebook.
It clones this repository, runs each approach in a separate Python process and checks the
generated predictions against fingerprints of the CSVs actually submitted to Kaggle.

**The notebook is a working draft.** V2 needs missing original code, and the fresh V5 result
differs from the submitted CSV in 43 of 5,000 predictions. These issues must be resolved
before a final course hand-in. The runner reports them; it does not substitute another model.

The first approach was executed through this notebook from a fresh local Git clone and matched
all 5,000 original predictions. The default all-three preflight correctly stopped on the missing
V2 code. The notebook format, code cells, matching folder layout and CSV/export checks were
validated, and its HTML presentation was visually inspected. A successful full three-approach
Kaggle run remains pending the two issues above.

## Matching folder layout

```text
submissions/
├── 01_hidden_ko_history/
├── 02_turn31_v5/
└── 03_turn31_v2/
```

Each folder contains the same entries:

```text
README.md          approach, command and reproduction status
run.py             the same command-line interface for every approach
requirements.txt   recorded local runtime dependencies
source.json        original source hashes, execution order and missing definitions
reference.json     original Kaggle submission ID, scores and prediction fingerprint
code/              unchanged model implementation or verbatim notebook cells
original/          original Kaggle script or original notebook, preserved in full
outputs/           generated files and local reference CSVs; excluded from Git
```

The first approach was a Kaggle Python script; the other two were notebooks. Their original
formats remain in `original/`. All three expose the same `run.py` interface.
The legacy root `run.py` and `from src import pipeline` still reach the first approach.

## Run an approach

From the repository root:

```bash
python -m pip install -r requirements-notebook.txt
python submissions/01_hidden_ko_history/run.py --data /path/to/competition --check
python submissions/01_hidden_ko_history/run.py --data /path/to/competition
```

Use `02_turn31_v5` or `03_turn31_v2` with the same arguments. `--out /path/to/folder` overrides
the default `outputs/run/` inside that submission's folder. `--check` does not train models.
V2 currently exits with an explicit incomplete-source error.

Each successful run saves `submission.csv`, `run.log`, `run_manifest.json` and the original
algorithm's additional files under `artifacts/`. The manifest records code commit, input hashes,
runtime versions and whether the predictions match the original Kaggle submission.
The runtime pins record the local check environment; the original V5/V2 environment is not known.

## Open the final notebook on Kaggle

1. Make this repository public when it is ready to share. Download `final_submissions.ipynb`
   from GitHub and import the file into a new Kaggle notebook. Kaggle also supports loading
   notebooks from a URL; see [Kaggle's notebook import announcement](https://www.kaggle.com/discussions/product-feedback/152480).
2. Attach the `fds-pokemon-battles-prediction-2026` competition data. The data folder must contain
   `train.jsonl`, `test.jsonl`, `type_chart_gen2.csv` and `sample_submission.csv`.
3. Enable Internet for the Git clone and package installation. See the
   [Kaggle notebook documentation](https://www.kaggle.com/docs/notebooks).
4. Keep the recorded commit in `REPO_REF`. The first cells clone the repository and check out
   that exact code. There is no model code to paste into Kaggle.
5. For development, select an available approach with the notebook controls. For the final
   hand-in, recover V2's original missing cells, resolve V5's prediction differences and leave
   all three approaches selected. Restart the session and run all cells.
6. Check that all three runs complete and match their original prediction fingerprints. The
   last section then exports the three regenerated CSVs and their manifest in one ZIP.
7. Save a successfully executed Kaggle version and make the notebook public. In the
   [course submission form](https://docs.google.com/forms/d/e/1FAIpQLSd8OEhyowMLGBJF_lXk_u8tReDVFKx0FrVKCwRryD-6OFmS-Q/viewform?usp=header),
   provide the public Kaggle notebook link and the requested report. The GitHub notebook
   alone does not replace the runnable public Kaggle notebook required by the slides.

The slides specify a short Markdown explanation before each of the three submissions; those
sections are already in the notebook. Their material deadline is 12 November 2026.

## Original reference files

The CSVs are downloaded locally into each `outputs/reference/submission.csv` and are excluded
from Git. `reference.json` contains both the file hash and an order-independent prediction hash.
Verification works from a public clone without downloading or embedding the reference CSVs.
To obtain a local original again, use its `submission_ref` with:

```bash
kaggle competitions submission-download SUBMISSION_REF -p /local/reference/folder
```

The old exploration spreadsheet is unused by these runners and is excluded from the current
tracked files. Copies still exist in older Git commits. Removing it from the current tree does
not remove those historical copies; publication without the spreadsheet needs a separate
history-cleanup decision. Keep the repository private until that is resolved.
