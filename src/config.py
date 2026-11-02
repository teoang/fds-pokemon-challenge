"""Constants shared by the whole pipeline."""

import os

COMPETITION = 'fds-pokemon-battles-prediction-2026'

# Validation
SEED = 2026
N_FOLDS = 5
N_REPEATS = 3
N_JOBS = os.cpu_count() or 4

# Battle records
STATUSES = ('slp', 'par', 'frz', 'tox', 'psn', 'brn')
BOOSTS = ('atk', 'def', 'spa', 'spd', 'spe')
EFFECTS = ('substitute', 'leechseed', 'confusion', 'perish', 'trapped', 'encore',
           'attract', 'nightmare', 'clamp', 'whirlpool', 'wrap', 'firespin', 'bind',
           'curse', 'destinybond', 'protect', 'endure', 'focusenergy')
SIDES = (('p1', 'p2'), ('p2', 'p1'))
STAT_NAMES = ('hp', 'atk', 'def', 'spa', 'spd', 'spe')

# Vocabulary thresholds (counts over train + test, labels never used)
MIN_SPECIES = 30
MIN_ITEM = 20
MIN_MOVE = 100

# Player 1's history
TE_SMOOTHING = 20          # pseudo-battles of prior in the team target encoding

# Hidden-knockout classifier
KO_PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=200,
                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
KO_SPLITS = 3              # the vanishing classifier is averaged over this many battle splits

# Outcome models
LGB_PARAMS = dict(n_estimators=700, learning_rate=0.02, num_leaves=12,
                  min_child_samples=50, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.4, reg_lambda=2.0, verbose=-1)
LOGREG_C = 0.005
