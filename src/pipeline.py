"""The whole pipeline: features, repeated CV with the ablation, final fit and outputs.

From a notebook, after cloning the repository and adding it to sys.path:

    from src import pipeline
    report = pipeline.run(ablation=False)

It never submits anything to Kaggle.
"""

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn

from .config import N_FOLDS, N_JOBS, N_REPEATS, SEED
from .data import Reference, find_data_dir, find_out_dir, load_jsonl
from .features import build_static, build_vocab
from .hidden_ko import calibrate_hidden_kos
from .history import frequency_features, history_features, history_keys
from .models import fit_predict
from .team_pool import TeamPool
from .utils import log
from .validation import make_folds, paired, run_cv, select_columns

# The final model: the corrected state, P1 team species and items, move counts
# and P1's history. P2's bench is measured in the ablation (D, H) but left out.
FINAL_GROUPS = ['core', 'ko', 'p1sp', 'it', 'mv']
FINAL = ('E final', FINAL_GROUPS, True, 'lgb')

ABLATION = [
    # name, feature groups, P1 history, model
    ('A state, naive', ['core', 'naive'], False, 'lgb'),
    ('B state, hidden KOs', ['core', 'ko'], False, 'lgb'),
    ('C B + P1 history', ['core', 'ko'], True, 'lgb'),
    ('D C + P2 bench', ['core', 'ko', 'bench', 'p2sp'], True, 'lgb'),
    ('R loops-style reference', ['core', 'naive', 'p1sp', 'it', 'mv'], False, 'lgb'),
    ('F final - history', FINAL_GROUPS, False, 'lgb'),
    ('G final - hidden KOs', ['core', 'naive', 'p1sp', 'it', 'mv'], True, 'lgb'),
    ('H final + P2 bench', FINAL_GROUPS + ['bench', 'p2sp'], True, 'lgb'),
    ('I final, logistic', FINAL_GROUPS, True, 'logreg'),
]

# (label, config a, config b): paired difference b - a
COMPARISONS = [
    ('final vs loops-style reference', 'R loops-style reference', FINAL[0]),
    ('P1 history (final minus F)', 'F final - history', FINAL[0]),
    ('hidden KOs (final minus G)', 'G final - hidden KOs', FINAL[0]),
    ('P2 bench (H minus final)', FINAL[0], 'H final + P2 bench'),
    ('LightGBM vs logistic (final minus I)', 'I final, logistic', FINAL[0]),
]


def run(data_dir=None, out_dir=None, repeats=N_REPEATS, ablation=True, configs=None,
        seed=SEED, n_jobs=N_JOBS):
    """Run the pipeline end to end and return the CV report as a dict.

    data_dir: folder with train.jsonl and test.jsonl (default: find_data_dir()).
    out_dir: where the outputs go (default: /kaggle/working on Kaggle,
        outputs/local/ in the repository otherwise).
    repeats: repeats of the 5-fold stratified CV.
    ablation: also run the ablation configurations, not only the final model.
    configs: comma-separated ablation letters to run, e.g. 'A,B' (None: all).
    seed: seed of the folds, of the hidden-KO splits and of the models.
    n_jobs: LightGBM threads.

    Writes submission.csv, test_probabilities.csv, oof_predictions.csv,
    feature_importance.csv, cv_report.json and cv_report.md.
    """
    data = Path(data_dir) if data_dir else find_data_dir()
    out = Path(out_dir) if out_dir else find_out_dir()
    out.mkdir(parents=True, exist_ok=True)
    log(f'data {data}  output {out}  cpus {n_jobs}')
    log(f'python {sys.version.split()[0]}  lightgbm {lgb.__version__}  scikit-learn {sklearn.__version__}  '
        f'pandas {pd.__version__}  numpy {np.__version__}')
    train = load_jsonl(data / 'train.jsonl')
    test = load_jsonl(data / 'test.jsonl')
    battles = train + test
    y = np.array([b['player_won'] for b in train], dtype=int)
    n_train = len(train)
    log(f'{n_train} train battles, {len(test)} test battles, P1 win rate {y.mean():.3f}')

    ref = Reference(battles)
    ko_probs, ko_info = calibrate_hidden_kos(battles, ref, seed, n_jobs)
    pool = TeamPool(battles)
    vocab = build_vocab(battles)
    log(f'species {len(ref.species)}, pool teams {len(pool.teams)}, vocab: '
        f'{len(vocab["species"])} species, {len(vocab["items"])} items, {len(vocab["moves"])} moves')
    log(f'hidden-KO calibration: {ko_info}')

    X_all = build_static(battles, ref, pool, ko_probs, vocab)
    keys = history_keys(battles)
    freq_all = frequency_features(keys)
    X, X_test = X_all.iloc[:n_train].reset_index(drop=True), X_all.iloc[n_train:].reset_index(drop=True)
    freq, freq_test = freq_all.iloc[:n_train].reset_index(drop=True), freq_all.iloc[n_train:].reset_index(drop=True)
    keys_train = {k: v[:n_train] for k, v in keys.items()}
    log(f'static features: {X.shape[1]} columns; expected hidden P2 KOs per battle '
        f'{X["ko_p2_hidden_sum"].mean():.2f}, visible P2 KOs {X["core_p2_fnt"].mean():.2f}, '
        f'visible P1 KOs {X["core_p1_fnt"].mean():.2f}')

    folds = make_folds(y, N_FOLDS, repeats, seed)
    hist_cache = [history_features(keys_train, y, a, v, seed + 100 + k) for k, (r, a, v) in enumerate(folds)]
    log(f'{len(folds)} folds ({N_FOLDS}-fold x {repeats} repeats); history encodings ready')

    chosen = ABLATION if ablation else []
    if configs:
        wanted = {c.strip() for c in configs.split(',')}
        chosen = [c for c in chosen if c[0].split()[0] in wanted]
    results, oofs = {}, {}
    for name, groups, use_hist, model in chosen + [FINAL]:
        res, oof = run_cv(name, X, groups, use_hist, model, folds, y, hist_cache, freq, seed, n_jobs)
        results[name], oofs[name] = res, oof

    # Paired comparisons on identical folds
    final_name, groups, use_hist, model = FINAL
    base_name = 'A state, naive' if 'A state, naive' in results else final_name
    lines = ['| config | features | CV accuracy | vs ' + base_name.split()[0] + ' (paired, ±SE) |',
             '|---|---:|---:|---:|']
    for name, res in results.items():
        d, se = paired(results[base_name], res)
        lines.append(f'| {name} | {res["n_features"]} | {res["acc"]:.4f} ± {res["acc_std_repeats"]:.4f} | '
                     f'{d:+.4f} ± {se:.4f} |')
        res['diff_vs_base'], res['se_vs_base'] = d, se
    table = '\n'.join(lines)
    print('\n' + table + '\n', flush=True)
    comparisons = {}
    for label, a, b in COMPARISONS:
        if a in results and b in results:
            d, se = paired(results[a], results[b])
            comparisons[label] = {'diff': d, 'se': se}
            log(f'{label}: {d:+.4f} ± {se:.4f} (corrected SE)')

    # Final model, fitted on all training battles
    log(f'final model: {final_name} (CV {results[final_name]["acc"]:.4f})')
    cols = select_columns(X.columns, groups)
    Xa, Xt = X[cols], X_test[cols]
    if use_hist:
        all_idx = np.arange(n_train)
        y_full = np.concatenate([y, np.zeros(len(test), dtype=int)])   # only train labels are read
        ha, ht = history_features(keys, y_full, all_idx, np.arange(n_train, n_train + len(test)), seed + 999)
        Xa = pd.concat([Xa.reset_index(drop=True), ha, freq], axis=1)
        Xt = pd.concat([Xt.reset_index(drop=True), ht, freq_test], axis=1)
    p_test, fitted = fit_predict(model, Xa, y, Xt, seed, n_jobs)
    sub = pd.DataFrame({'battle_id': [b['battle_id'] for b in test],
                        'player_won': (p_test > 0.5).astype(int)})
    sub.to_csv(out / 'submission.csv', index=False)
    pd.DataFrame({'battle_id': [b['battle_id'] for b in test], 'p_win': p_test}).to_csv(
        out / 'test_probabilities.csv', index=False)
    pd.DataFrame({'battle_id': [b['battle_id'] for b in train], 'player_won': y,
                  'oof_p_win': oofs[final_name]}).to_csv(out / 'oof_predictions.csv', index=False)
    if hasattr(fitted, 'booster_'):
        imp = pd.DataFrame({'feature': fitted.booster_.feature_name(),
                            'gain': fitted.booster_.feature_importance('gain')})
        imp.sort_values('gain', ascending=False).to_csv(out / 'feature_importance.csv', index=False)
    log(f'submission.csv: {len(sub)} rows, predicted P1 win rate {sub.player_won.mean():.3f}')

    report = {'final': final_name, 'n_folds': N_FOLDS, 'n_repeats': repeats, 'seed': seed,
              'hidden_ko_calibration': ko_info, 'comparisons': comparisons,
              'results': results, 'table': table}
    (out / 'cv_report.json').write_text(json.dumps(report, indent=2))
    (out / 'cv_report.md').write_text(table + f'\n\nFinal model: {final_name}\n')
    log('done; nothing was submitted')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description='Repeated CV, ablation, final fit and submission.csv.')
    parser.add_argument('--quick', action='store_true', help='one CV repeat instead of three')
    parser.add_argument('--repeats', type=int, default=None, help='number of CV repeats')
    parser.add_argument('--no-ablation', action='store_true', help='only the final model')
    parser.add_argument('--configs', default=None,
                        help='comma-separated ablation letters to run, e.g. "A,B" (the final model always runs)')
    parser.add_argument('--jobs', type=int, default=None, help='LightGBM threads')
    parser.add_argument('--out', default=None, help='output folder (default: outputs/local, or /kaggle/working)')
    parser.add_argument('--seed', type=int, default=None, help=f'seed for folds and models (default {SEED})')
    args, _ = parser.parse_known_args(argv)
    run(out_dir=args.out,
        repeats=args.repeats or (1 if args.quick else N_REPEATS),
        ablation=not args.no_ablation,
        configs=args.configs,
        seed=SEED if args.seed is None else args.seed,
        n_jobs=args.jobs or N_JOBS)
