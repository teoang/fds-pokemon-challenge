"""Paths, loading, and what can be read directly from the battle records."""

import json
import os
from pathlib import Path

import numpy as np

from .config import COMPETITION, SIDES, STAT_NAMES

REPO_ROOT = Path(__file__).resolve().parents[1]


def find_data_dir():
    """Folder with train.jsonl: FDS_DATA, the Kaggle mount, or data/ in the repository."""
    candidates = []
    if os.environ.get('FDS_DATA'):
        candidates.append(Path(os.environ['FDS_DATA']))
    candidates += [Path('/kaggle/input') / COMPETITION,
                   Path('/kaggle/input/competitions') / COMPETITION,
                   REPO_ROOT / 'data']
    for c in candidates:
        if (c / 'train.jsonl').exists():
            return c
    kaggle_input = Path('/kaggle/input')
    if kaggle_input.exists():
        for p in sorted(kaggle_input.rglob('train.jsonl')):
            return p.parent
    raise FileNotFoundError('train.jsonl not found: set the FDS_DATA environment variable')


def find_out_dir():
    """/kaggle/working on Kaggle, outputs/local/ in the repository otherwise."""
    if Path('/kaggle/working').exists():
        return Path('/kaggle/working')
    out = REPO_ROOT / 'outputs' / 'local'
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class Reference:
    """Species types and base stats, read from the teams and leads in the data."""

    def __init__(self, battles):
        self.species = {}
        for b in battles:
            for p in b['p1_team_details'] + [b['p2_lead_details']]:
                if p['name'] not in self.species:
                    self.species[p['name']] = {
                        'types': tuple(t for t in p['types'] if t != 'notype'),
                        'stats': np.array([p['base_' + s] for s in STAT_NAMES], dtype=float)}
        self.mean_stats = np.mean([v['stats'] for v in self.species.values()], axis=0)

    def types(self, name):
        sp = self.species.get(name)
        return sp['types'] if sp else ()

    def stats(self, name):
        sp = self.species.get(name)
        return sp['stats'] if sp else self.mean_stats


def p2_seen(b):
    """Every Player 2 Pokémon revealed in the battle: the lead plus the timeline."""
    return frozenset([b['p2_lead_details']['name']] +
                     [t['p2_pokemon_state']['name'] for t in b['battle_timeline']])


def p1_species(b):
    return frozenset(p['name'] for p in b['p1_team_details'])


def spikes_after(tl, ref):
    """For every record, whether Spikes lie on each side's field after it.

    Spikes land on a side when the opponent uses the move, and leave it when the
    side uses Rapid Spin against a non-Ghost opponent (Ghosts block the spin).
    """
    on = {'p1': False, 'p2': False}
    out = {'p1': [], 'p2': []}
    for t in tl:
        for side, opp in SIDES:
            mo = t[opp + '_move_details']
            if mo is not None and mo['name'] == 'spikes':
                on[side] = True
        for side, opp in SIDES:
            ms = t[side + '_move_details']
            if (ms is not None and ms['name'] == 'rapidspin'
                    and 'ghost' not in ref.types(t[opp + '_pokemon_state']['name'])):
                on[side] = False
        for side in out:
            out[side].append(on[side])
    return out
