"""Player 2's bench, estimated from the pool of teams shared by both players.

The ablation measures it (configs D and H); the final model leaves it out
because it adds nothing once P1's species and the move counts are in.
"""

from collections import Counter, defaultdict

import numpy as np

from .data import p1_species, p2_seen


class TeamPool:
    """Every team seen in the data: P1 teams and fully revealed P2 teams."""

    def __init__(self, battles):
        count = Counter()
        for b in battles:
            count[p1_species(b)] += 1
            seen = p2_seen(b)
            if len(seen) == 6:
                count[seen] += 1
        self.teams = list(count)
        self.weights = np.array([count[t] for t in self.teams], dtype=float)
        self.index = defaultdict(set)
        for k, team in enumerate(self.teams):
            for s in team:
                self.index[s].add(k)

    def bench(self, seen):
        """Expected count of each unseen species on P2's team.

        Uses the teams that contain every revealed Pokémon; if there are none,
        the teams with the largest overlap. Returns (posterior, n_candidates,
        exact_match).
        """
        need = 6 - len(seen)
        if need <= 0:
            return {}, 0, 1
        sets = [self.index.get(s, set()) for s in seen]
        cands = set.intersection(*sets) if sets else set()
        exact = 1
        if not cands:
            exact = 0
            overlap = Counter()
            for s in sets:
                for k in s:
                    overlap[k] += 1
            if not overlap:
                return {}, 0, 0
            best = max(overlap.values())
            cands = {k for k, v in overlap.items() if v == best}
        post, total = Counter(), 0.0
        for k in sorted(cands):
            rest = self.teams[k] - seen
            if not rest:
                continue
            w = self.weights[k]
            scale = need / len(rest)   # loose matches leave more than `need` members
            for s in sorted(rest):   # sorted: set order changes with the hash seed
                post[s] += w * scale
            total += w
        if total == 0:
            return {}, len(cands), exact
        return {s: v / total for s, v in post.items()}, len(cands), exact
