"""Static features of each battle. Column prefixes define the feature groups.

- core: the state that does not depend on the knockout correction
- naive: the state taken at face value from the timeline
- ko: the same state with P2's hidden knockouts removed in expectation
- p1sp, it: P1's species and items; mv: P1-minus-P2 move counts
- p2sp, bench: P2's revealed Pokémon and its estimated bench (ablation only)
"""

from collections import Counter

import numpy as np
import pandas as pd

from .config import BOOSTS, EFFECTS, MIN_ITEM, MIN_MOVE, MIN_SPECIES, SIDES, STAT_NAMES, STATUSES
from .data import p2_seen, spikes_after
from .hidden_ko import vanish_events
from .utils import clean


def battle_features(b, ref, pool, ko_probs, vocab):
    """Static features of one battle, as a dict of column name to value."""
    tl = b['battle_timeline']
    sp = spikes_after(tl, ref)
    f = {}

    # Hidden knockouts of P2: probability per vanished Pokémon.
    ko_prob = {'p1': {}, 'p2': {}}
    p2_events = vanish_events(tl, 'p2', sp)
    for e, pk in zip(p2_events, ko_probs):
        ko_prob['p2'][e['name']] = pk
    f['ko_p2_vanished'] = len(p2_events)
    f['ko_p2_hidden_sum'] = sum(ko_prob['p2'].values())

    alive_w = {}
    for side, opp in SIDES:
        last = {}
        for t in tl:
            s = t[side + '_pokemon_state']
            last[s['name']] = s
        fnt = {n for n, s in last.items() if s['status'] == 'fnt'}
        n_unseen = 6 - len(last)   # P1: never sent out; P2: not revealed yet
        w = {n: (0.0 if n in fnt else 1.0 - ko_prob[side].get(n, 0.0)) for n in last}
        alive_w[side] = w

        # core: unaffected by the knockout correction
        f[f'core_{side}_seen'] = len(last)
        f[f'core_{side}_fnt'] = len(fnt)
        act = tl[-1][side + '_pokemon_state']
        f[f'core_{side}_act_hp'] = act['hp_pct']
        f[f'core_{side}_act_fnt'] = int(act['status'] == 'fnt')
        for st in STATUSES:
            f[f'core_{side}_act_{st}'] = int(act['status'] == st)
        for k in BOOSTS:
            f[f'core_{side}_act_boost_{k}'] = act['boosts'][k]
        for e in EFFECTS:
            f[f'core_{side}_act_eff_{e}'] = int(e in act['effects'])
        f[f'core_{side}_spikes'] = int(sp[side][-1])

        # naive: what the timeline shows at face value
        alive_naive = [s for n, s in last.items() if n not in fnt]
        f[f'naive_{side}_alive'] = 6 - len(fnt)
        f[f'naive_{side}_hp'] = sum(s['hp_pct'] for s in alive_naive) + n_unseen
        f[f'naive_{side}_low'] = sum(1 for s in alive_naive if s['hp_pct'] < 0.25)
        f[f'naive_{side}_healthy'] = sum(1 for s in alive_naive if s['hp_pct'] >= 0.75) + n_unseen
        for st in STATUSES:
            f[f'naive_{side}_st_{st}'] = sum(1 for s in alive_naive if s['status'] == st)

        # ko: same quantities with P2's hidden knockouts removed in expectation
        f[f'ko_{side}_alive'] = sum(w.values()) + n_unseen
        f[f'ko_{side}_hp'] = sum(s['hp_pct'] * w[n] for n, s in last.items()) + n_unseen
        f[f'ko_{side}_low'] = sum(w[n] for n, s in last.items() if 0 < s['hp_pct'] < 0.25)
        f[f'ko_{side}_healthy'] = sum(w[n] for n, s in last.items() if s['hp_pct'] >= 0.75) + n_unseen
        for st in STATUSES:
            f[f'ko_{side}_st_{st}'] = sum(w[n] for n, s in last.items() if s['status'] == st)

    for prefix in ('core', 'naive', 'ko'):
        for key in [k for k in f if k.startswith(prefix + '_p1_')]:
            other = key.replace('_p1_', '_p2_', 1)
            if other in f:
                f[key.replace('_p1_', '_d_', 1)] = f[key] - f[other]

    # P1 team composition and items
    for p in b['p1_team_details']:
        if p['name'] in vocab['species']:
            f['p1sp_' + clean(p['name'])] = 1
        if p['item'] in vocab['items']:
            f['it_' + clean(p['item'])] = f.get('it_' + clean(p['item']), 0) + 1
    levels = [p['level'] for p in b['p1_team_details']]
    f['p1sp_min_level'] = min(levels)

    # Moves used: P1 count minus P2 count, per move name
    for t in tl:
        for side, sign in (('p1', 1), ('p2', -1)):
            m = t[side + '_move_details']
            if m is not None and m['name'] in vocab['moves']:
                key = 'mv_' + clean(m['name'])
                f[key] = f.get(key, 0) + sign

    # P2 composition: revealed Pokémon weighted by survival, plus the bench
    seen = p2_seen(b)
    post, n_cand, exact = pool.bench(seen)
    for n, wt in alive_w['p2'].items():
        if n in vocab['species']:
            f['p2sp_' + clean(n)] = wt
    lead = b['p2_lead_details']['name']
    if lead not in alive_w['p2'] and lead in vocab['species']:
        f['p2sp_' + clean(lead)] = 1.0
    for n, pr in post.items():
        if n in vocab['species']:
            f['bench_sp_' + clean(n)] = pr
    need = 6 - len(seen)
    f['bench_need'] = need
    f['bench_cands'] = np.log1p(n_cand)
    f['bench_exact'] = exact
    if post:
        tot = sum(post.values())
        exp_stats = sum(ref.stats(n) * pr for n, pr in sorted(post.items())) / tot
    else:
        exp_stats = ref.mean_stats
    for k, s in enumerate(STAT_NAMES):
        f[f'bench_{s}'] = exp_stats[k] if need > 0 else 0.0
    f['bench_bst'] = float(exp_stats.sum()) if need > 0 else 0.0
    return f


def build_vocab(battles):
    """Species, items and moves frequent enough to get their own column."""
    sp, it, mv = Counter(), Counter(), Counter()
    for b in battles:
        for p in b['p1_team_details']:
            sp[p['name']] += 1
            it[p['item']] += 1
        for t in b['battle_timeline']:
            for side in ('p1', 'p2'):
                m = t[side + '_move_details']
                if m is not None:
                    mv[m['name']] += 1
    return {'species': {k for k, v in sp.items() if v >= MIN_SPECIES},
            'items': {k for k, v in it.items() if v >= MIN_ITEM},
            'moves': {k for k, v in mv.items() if v >= MIN_MOVE}}


def build_static(battles, ref, pool, ko_probs, vocab):
    """Feature table of all battles, with columns in a fixed order."""
    rows = [battle_features(b, ref, pool, kp, vocab) for b, kp in zip(battles, ko_probs)]
    X = pd.DataFrame(rows).fillna(0.0).astype(np.float32)
    return X[sorted(X.columns)]
