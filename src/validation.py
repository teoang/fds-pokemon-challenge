"""Repeated stratified cross-validation and paired comparisons between configurations."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .config import N_FOLDS
from .models import fit_predict
from .utils import log


def make_folds(y, n_folds, n_repeats, seed):
    """(repeat, train indices, validation indices) for every fold of every repeat."""
    folds = []
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed + r)
        for a, v in skf.split(np.zeros(len(y)), y):
            folds.append((r, a, v))
    return folds


def select_columns(columns, groups):
    prefixes = tuple(g + '_' for g in groups)
    return [c for c in columns if c.startswith(prefixes)]


def run_cv(name, X, groups, use_history, model, folds, y, hist_cache, freq, seed, n_jobs):
    """Accuracy of one configuration on every fold, plus its out-of-fold predictions.

    `hist_cache[k]` holds the P1-history encodings of fold k, computed from that
    fold's training labels only.
    """
    n_repeats = folds[-1][0] + 1
    cols = select_columns(X.columns, groups)
    accs, oof = [], np.zeros((n_repeats, len(y)))
    for k, (r, a, v) in enumerate(folds):
        Xa = X.iloc[a][cols].reset_index(drop=True)
        Xv = X.iloc[v][cols].reset_index(drop=True)
        if use_history:
            ha, hv = hist_cache[k]
            Xa = pd.concat([Xa, ha, freq.iloc[a].reset_index(drop=True)], axis=1)
            Xv = pd.concat([Xv, hv, freq.iloc[v].reset_index(drop=True)], axis=1)
        p, _ = fit_predict(model, Xa, y[a], Xv, seed + k, n_jobs)
        oof[r, v] = p
        accs.append(float(((p > 0.5) == y[v]).mean()))
    per_repeat = [float(((oof[r] > 0.5) == y).mean()) for r in range(n_repeats)]
    res = {'name': name, 'groups': groups, 'history': use_history, 'model': model,
           'n_features': len(cols) + (4 if use_history else 0),
           'fold_acc': accs, 'repeat_acc': per_repeat,
           'acc': float(np.mean(per_repeat)), 'acc_std_repeats': float(np.std(per_repeat))}
    log(f'{name:<34s} acc {res["acc"]:.4f} (repeats {", ".join(f"{a:.4f}" for a in per_repeat)}) '
        f'features {res["n_features"]}')
    return res, oof.mean(axis=0)


def paired(res_a, res_b, test_frac=1 / (N_FOLDS - 1)):
    """Mean fold-level difference b - a with the Nadeau-Bengio corrected SE."""
    d = np.array(res_b['fold_acc']) - np.array(res_a['fold_acc'])
    j = len(d)
    se = float(np.sqrt((1 / j + test_frac) * d.var(ddof=1))) if j > 1 else float('nan')
    return float(d.mean()), se
