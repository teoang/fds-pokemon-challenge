# fds-pokemon-challenge

Our solution to the FDS 2026/27 Kaggle challenge at Sapienza University of Rome: predict whether
Player 1 wins a Gen 2 OU Pokémon battle (competitive singles, second-generation games) from its
first 30 turns. There are 10,000 training battles and 5,000 test battles, and the metric is
accuracy.

**Main result:** CV accuracy **0.699** (5-fold × 3 repeats) and **0.700 on the public
leaderboard**, with approach `01_hidden_ko_history`. The same LightGBM model with both
knockout correction and team history disabled scores 0.685. Almost all of the gain comes from the
knockout correction.

## The idea

The battle log hides who has really lost Pokémon:

- **Player 2's knockouts are hidden.** A fainted P2 Pokémon is almost never marked as fainted:
  the log often shows its replacement without a separate fainting state for the departing
  Pokémon. It therefore looks like an ordinary switch.
- **Player 1's knockouts are always visible.** So P1's switches show what a real switch looks
  like, and the difference between P2's and P1's switches gives the hidden knockouts.

Once the hidden knockouts are added back, P2's mean knockout count is close to P1's:

![Mean knockouts per battle during recorded turns 1–10, 11–20 and 21–30. Player 1: 0.34, 0.58, 0.76. Player 2 visible alone: 0.11, 0.18, 0.22; with the estimated hidden knockouts Player 2 reaches 0.33, 0.55, 0.73.](docs/figures/ko_symmetry.svg)

The three groups cover recorded turns **1–10, 11–20 and 21–30**. Each bar shows the mean number
of knockouts per battle within that window. Depending on the window, between about 0.2 and 0.5
P2 knockouts per battle are hidden.

A secondary signal is **Player 1's team history**: the same P1 team appears in many battles, and its win
rate in the other battles predicts the outcome. In the final model, however, it adds little on
top of P1's species and move counts (see [Results](#results)).

## How it works

The code is in `submissions/01_hidden_ko_history/code/src/`.

1. **Hidden knockouts** (`hidden_ko.py`). For departures with no observed fainting state where
   the Pokémon does not reappear within the first 30 turns, a classifier distinguishes
   P2's departures from P1's voluntary switches. It uses HP and status before leaving, recent HP
   changes, both players' move powers and whether the replacement entered unhurt after accounting
   for Spikes. The excess of P2 events estimates the probability of a hidden knockout. No
   battle-outcome labels are used for this estimate.
2. **State at turn 30** (`features.py`). Pokémon alive, HP, statuses, the active Pokémon and
   Spikes, with the estimated knockouts removed. On top: P1's species and items, and P1-minus-P2
   move counts.
3. **Player 1's history** (`history.py`). Smoothed win rates for two team keys: the six species,
   and the six species together with their held items. Both keys ignore team order. These are
   out-of-fold target encodings computed inside each CV fold. The history uses the six-member
   P1 roster supplied with each battle.
4. **Model** (`models.py`). LightGBM.

Every configuration of the main approach is evaluated with the same 5-fold stratified CV, repeated 3 times.
Differences are paired on the same folds, and ± is the standard error with the
Nadeau–Bengio correction for overlapping training sets, implemented in
[`validation.py`](submissions/01_hidden_ko_history/code/src/validation.py).

## Results

![CV accuracy of the ten configurations, sorted: E final model 0.6986 first, R baseline without correction or history 0.6849, A state at face value 0.6753 last.](docs/figures/ablation.svg)

Key configurations: **E** is the final model, **R** the same LightGBM without knockout correction
or history, and **A** the turn-30 state taken at face value. All ten are defined in
[`pipeline.py`](submissions/01_hidden_ko_history/code/src/pipeline.py).

![Gain in CV accuracy from each block, in percentage points: knockout correction +1.15 (±0.29), LightGBM instead of logistic regression +0.54 (±0.50), Player 1's history +0.17 (±0.35), Player 2's bench −0.04 (±0.35).](docs/figures/contributions.svg)

- **Knockout correction:** +1.15 pp, the only gain clearly above noise.
- **LightGBM instead of logistic regression:** +0.54 pp, at the edge of noise.
- **Player 1's history:** +0.17 pp, within noise once species and move counts are in.
- **Player 2's estimated bench:** no gain, so the final model leaves it out.
- **Also tried:** more state features, XGBoost, CatBoost and blends. All stayed within noise.

## Other submitted approaches

The repository also contains two other approaches that were submitted to the competition:

| approach               | what it does                 | CV    | public LB |
| ---------------------- | ---------------------------- | ----- | --------- |
| `01_hidden_ko_history` | the approach described above | 0.699 | 0.700     |
| `02_turn31_v5`         | turn-31 forecast with an 80/12/8 CatBoost/XGBoost/LightGBM blend | 0.689 | 0.697 |
| `03_turn31_v2`         | turn-31 forecast with a logistic-regression stack of boosted trees | 0.681 | 0.695 |

The forecast pipelines predict the next switches and active Pokémon types after the observed
30 turns, then use these turn-31 estimates as features for predicting the final winner. They do
not observe the actual 31st turn. Their original CV results use ten stratified folds.

## Data use and assumptions

- The test set's features, never its labels, are used to calibrate the knockout estimate and to
  count teams.
- The estimate assumes that both players switch Pokémon the same way.

## How to run

**On Kaggle:** use [`final_submissions.ipynb`](final_submissions.ipynb). It runs the three
approaches independently, checks the competition inputs and the generated CSVs, and records the
source version and runtime of each run.

**Locally** (Python 3.12 or 3.13), for the main approach only:

```bash
pip install -r requirements.txt
kaggle competitions download -c fds-pokemon-battles-prediction-2026 -p data
unzip data/fds-pokemon-battles-prediction-2026.zip -d data
python run.py
```

`run.py` writes `submission.csv` and the CV report to `outputs/local/` (or `/kaggle/working` on
Kaggle). The default run includes the full ablation. `--no-ablation` runs only the final model
and `--quick` a single CV repeat. `python run.py --help` lists every
option. The `data/` folder is git-ignored.

From a notebook, after cloning the repository:

```python
import sys
sys.path.append('fds-pokemon-challenge')
from src import pipeline
pipeline.run(ablation=False)
```

`run.py` and `src/` are thin wrappers around `submissions/01_hidden_ko_history/`, kept so these
older commands still work.

## Repository

| path                                | content                                            |
| ----------------------------------- | -------------------------------------------------- |
| `final_submissions.ipynb`           | Kaggle entry notebook for the three approaches     |
| `submissions/01_hidden_ko_history/` | main approach: source and reference metadata       |
| `submissions/02_turn31_v5/`         | turn-31 V5 approach: source and reference metadata |
| `submissions/03_turn31_v2/`         | turn-31 V2 approach: source and reference metadata |
| `submission_tools/`                 | shared path handling, execution and CSV checks     |
| `run.py`, `src/`                    | local entry points for the main approach           |
| `docs/figures/`                     | the figures, and `make_figures.py` to redraw them  |
| `docs/uml/`                         | UML diagram of the battle JSON                     |
