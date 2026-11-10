"""Player 1's history: out-of-fold target encoding of Player 1's exact team."""

from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .config import TE_SMOOTHING


def history_keys(battles):
    """P1's team as a key: species only, and species plus items."""
    team = np.array(['|'.join(sorted(p['name'] for p in b['p1_team_details'])) for b in battles], dtype=object)
    team_items = np.array(['|'.join(sorted(p['name'] + ':' + p['item'] for p in b['p1_team_details']))
                           for b in battles], dtype=object)
    return {'hist_team': team, 'hist_team_items': team_items}


def target_encode(keys, y, fit_idx, apply_idx, prior, m=TE_SMOOTHING):
    """Smoothed win rate of each key, from the labels of `fit_idx` only."""
    s, n = defaultdict(float), defaultdict(int)
    for k, t in zip(keys[fit_idx], y[fit_idx]):
        s[k] += t
        n[k] += 1
    return np.array([(s[k] + prior * m) / (n[k] + m) for k in keys[apply_idx]])


def history_features(keys_by_name, y, fit_idx, apply_idx, seed):
    """Encodings for the fitting rows (out-of-fold inside them) and the apply rows.

    Only labels of `fit_idx` are read, so validation and test labels never leak.
    """
    prior = float(y[fit_idx].mean())
    fit_cols, app_cols = {}, {}
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for name, keys in keys_by_name.items():
        col = np.zeros(len(fit_idx))
        for ia, ib in inner.split(np.zeros(len(fit_idx)), y[fit_idx]):
            col[ib] = target_encode(keys, y, fit_idx[ia], fit_idx[ib], prior)
        fit_cols[name] = col
        app_cols[name] = target_encode(keys, y, fit_idx, apply_idx, prior)
    return (pd.DataFrame(fit_cols).astype(np.float32),
            pd.DataFrame(app_cols).astype(np.float32))


def frequency_features(keys_by_name):
    """How often each P1 team appears in train + test (labels not used)."""
    out = {}
    for name, keys in keys_by_name.items():
        c = Counter(keys)
        out[name.replace('hist_', 'freq_')] = np.log1p([c[k] for k in keys])
    return pd.DataFrame(out).astype(np.float32)
