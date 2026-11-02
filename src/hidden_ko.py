"""Hidden knockouts of Player 2.

Player 2's `fnt` state shows only on a double KO or when Player 2 has no
Pokémon left, so most P2 knockouts look like ordinary switches. Player 1's
knockouts are always shown, so P1's "vanishings" (a Pokémon leaves the field
and never comes back) are all real switches and act as a control group.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import KO_PARAMS, KO_SPLITS
from .data import spikes_after

STATUS_CODE = {'nostatus': 0, 'slp': 1, 'par': 2, 'frz': 3, 'tox': 4, 'psn': 5, 'brn': 6}


def vanish_events(tl, side, spikes):
    """Switch-outs after which the Pokémon never shows up again in the 30 records.

    Each event gets a description `x` that reads the same for either player:
    the leaver's HP, status and last HP drop, the base power of both players'
    moves in the leaving record and the one before, the opponent's HP drop
    (Explosion users), and whether the replacement came in unhurt (Spikes
    aside): after a knockout the replacement enters at the end of the turn,
    while after a voluntary switch it usually takes the opponent's attack.
    """
    opp = 'p2' if side == 'p1' else 'p1'
    names = [t[side + '_pokemon_state']['name'] for t in tl]
    last_hp = {}
    events = []
    for i, t in enumerate(tl):
        cur = t[side + '_pokemon_state']
        if i > 0:
            prev = tl[i - 1][side + '_pokemon_state']
            if (prev['name'] != cur['name'] and prev['status'] != 'fnt'
                    and prev['name'] not in names[i:]):
                om = t[opp + '_move_details']
                hit = om is not None and om['base_power'] > 0
                allowance = 0.02 + (0.125 if spikes[side][i - 1] else 0.0)
                unhurt = last_hp.get(cur['name'], 1.0) - cur['hp_pct'] <= allowance
                before = tl[i - 2][side + '_pokemon_state'] if i >= 2 else None
                x = {'i': i, 'hp': prev['hp_pct'], 'status': STATUS_CODE.get(prev['status'], 7),
                     'drop': (before['hp_pct'] - prev['hp_pct']
                              if before is not None and before['name'] == prev['name'] else np.nan),
                     'signature': int(hit and unhurt), 'unhurt': int(unhurt),
                     'spikes': int(spikes[side][i - 1]), 'rep_hp': cur['hp_pct'],
                     'opp_switch': int(tl[i - 1][opp + '_pokemon_state']['name'] != t[opp + '_pokemon_state']['name'])}
                for j, tag in ((i - 1, 'prev'), (i, 'cur')):
                    mv, att = tl[j][opp + '_move_details'], tl[j][opp + '_pokemon_state']
                    own = tl[j][side + '_move_details']
                    x[f'opp_bp_{tag}'] = mv['base_power'] if mv else -1
                    x[f'own_bp_{tag}'] = own['base_power'] if own else -1
                    att_before = tl[j - 1][opp + '_pokemon_state'] if j > 0 else None
                    x[f'opp_drop_{tag}'] = (att_before['hp_pct'] - att['hp_pct']
                                            if att_before is not None and att_before['name'] == att['name'] else np.nan)
                events.append({'name': prev['name'], 'x': x})
        last_hp[cur['name']] = cur['hp_pct']
    return events


def calibrate_hidden_kos(battles, ref, seed, n_jobs):
    """P(knockout) for every P2 vanishing, using P1's vanishings as the control.

    Both players switch alike, so where P2 events outnumber P1 events with the
    same description, the excess is hidden P2 knockouts. A classifier telling
    P2 events from P1 events estimates the ratio n2(x) / n1(x) and
    P(KO | x) = 1 - n1(x) / n2(x). Predictions are out-of-fold by battle and
    averaged over KO_SPLITS splits; one scale factor makes the expected total
    equal the overall excess.
    Battle labels are not used. Returns (per_battle, info): one list of
    probabilities per battle, in vanish_events order, and calibration statistics.
    """
    rows, side, owner = [], [], []
    for k, b in enumerate(battles):
        tl = b['battle_timeline']
        sp = spikes_after(tl, ref)
        for s in ('p1', 'p2'):
            for e in vanish_events(tl, s, sp):
                rows.append(e['x'])
                side.append(int(s == 'p2'))
                owner.append(k)
    X, side, owner = pd.DataFrame(rows), np.array(side), np.array(owner)
    q = np.zeros(len(side))
    for r in range(KO_SPLITS):
        # Battles go to folds through a seeded permutation, which is the same on
        # every platform (GroupKFold breaks ties with an unstable sort).
        fold = np.random.RandomState(seed + r).permutation(len(battles))[owner] % 5
        for k in range(5):
            a, v = fold != k, fold == k
            m = lgb.LGBMClassifier(**KO_PARAMS, random_state=seed + r, n_jobs=n_jobs)
            m.fit(X[a], side[a])
            q[v] += m.predict_proba(X[v])[:, 1] / KO_SPLITS
    is_p2 = side == 1
    odds = (1 - q[is_p2]) / q[is_p2]
    target = is_p2.sum() - (~is_p2).sum()
    lo, hi = 0.5, 3.0   # bisection on the scale factor
    for _ in range(50):
        c = (lo + hi) / 2
        lo, hi = (c, hi) if np.clip(1 - c * odds, 0, 1).sum() > target else (lo, c)
    p = np.clip(1 - c * odds, 0, 1)
    per_battle = [[] for _ in battles]
    for k, pk in zip(owner[is_p2], p):
        per_battle[k].append(float(pk))
    return per_battle, {'scale': c, 'p2_events': int(is_p2.sum()), 'p1_events': int((~is_p2).sum()),
                        'mean': float(p.mean()), 'sd': float(p.std())}
