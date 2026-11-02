"""Outcome models: LightGBM (the final model) and a logistic-regression reference."""

import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .config import LGB_PARAMS, LOGREG_C


def fit_predict(model, Xa, ya, Xv, seed, n_jobs):
    """Fit on (Xa, ya); return P(P1 wins) for the rows of Xv and the fitted model."""
    if model == 'lgb':
        m = lgb.LGBMClassifier(**LGB_PARAMS, random_state=seed, n_jobs=n_jobs)
        m.fit(Xa, ya)
        return m.predict_proba(Xv)[:, 1], m
    if model == 'logreg':
        scaler = StandardScaler().fit(Xa.values)
        m = LogisticRegression(C=LOGREG_C, max_iter=5000)
        m.fit(scaler.transform(Xa.values), ya)
        return m.predict_proba(scaler.transform(Xv.values))[:, 1], m
    raise ValueError(model)
