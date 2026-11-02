#!/usr/bin/env python3
"""Rebuild the README figures from the saved outputs. No model is re-run.

Usage, from the repository root:
    python docs/figures/make_figures.py

The CV numbers come from outputs/kaggle/cv_report.json. The knockout counts
and the public leaderboard scores are copied below, with their source. Each
figure is an SVG that follows the reader's light or dark theme. Only the
Python standard library is needed.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

# Knockouts per battle by record range, over train and test (15,000 battles).
# P1 and P2 visible: `fnt` states in the data. P2 total: visible plus the hidden
# knockouts estimated by src/hidden_ko.py (final calibrator, seed 2026).
KO_RANGES = ['Records 1–10', 'Records 11–20', 'Records 21–30']
KO_P1 = [0.342, 0.580, 0.764]
KO_P2_VISIBLE = [0.109, 0.175, 0.223]
KO_P2_TOTAL = [0.331, 0.547, 0.729]

# The team's public leaderboard submissions, from `kaggle competitions submissions`.
LEADERBOARD = [('v1', '26 Oct', 0.68666), ('v2', '26 Oct', 0.69466), ('v3', '26 Oct', 0.69400),
               ('v5 CatBoost', '27 Oct', 0.69666), ('idea1', '27 Oct', 0.70000)]
PUBLIC_SE = 0.012          # about 1,500 public battles: sqrt(0.7 * 0.3 / 1500)
IDEA1_CV = 0.6986

# Colors from the data-viz reference palette (light, then dark).
STYLE = """
svg { font-family: system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }
.surface { fill: #fcfcfb; }
.ink { fill: #0b0b0b; } .ink2 { fill: #52514e; } .muted { fill: #898781; }
.grid { stroke: #e1e0d9; } .axis { stroke: #c3c2b7; } .err { stroke: #52514e; }
.ring { stroke: #fcfcfb; }
.accent { fill: #2a78d6; } .accent-line { stroke: #2a78d6; }
.slot2 { fill: #eb6834; } .slot3 { fill: #1baf7a; }
.rest { fill: #898781; } .rest-line { stroke: #898781; }
.num { font-variant-numeric: tabular-nums; }
@media (prefers-color-scheme: dark) {
  .surface { fill: #1a1a19; }
  .ink { fill: #f0efec; } .ink2 { fill: #c3c2b7; }
  .grid { stroke: #2c2c2a; } .axis { stroke: #383835; } .err { stroke: #c3c2b7; }
  .ring { stroke: #1a1a19; }
  .accent { fill: #3987e5; } .accent-line { stroke: #3987e5; }
  .slot2 { fill: #d95926; } .slot3 { fill: #199e70; }
}
"""


def esc(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


class Figure:
    """A small SVG canvas with the chart surface, a title and a subtitle."""

    def __init__(self, name, width, height, title, subtitle, desc):
        self.name, self.w, self.h = name, width, height
        self.parts = [f'<rect class="surface" x="0" y="0" width="{width}" height="{height}" rx="8"/>']
        self.text(24, 34, title, 'ink', 16, weight=600)
        self.text(24, 56, subtitle, 'ink2', 12.5)
        self.title, self.desc = title, desc

    def text(self, x, y, s, cls='ink', size=13, anchor='start', weight=None):
        w = f' font-weight="{weight}"' if weight else ''
        self.parts.append(f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" font-size="{size}" '
                          f'text-anchor="{anchor}"{w}>{esc(s)}</text>')

    def line(self, x1, y1, x2, y2, cls, width=1.0, cap='butt'):
        self.parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="{cls}" '
                          f'stroke-width="{width}" stroke-linecap="{cap}"/>')

    def path(self, d, cls):
        self.parts.append(f'<path d="{d}" class="{cls}"/>')

    def dot(self, cx, cy, cls, r=5):
        self.parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" class="{cls} ring" stroke-width="2"/>')

    def swatch(self, x, y, cls, label):
        self.parts.append(f'<rect x="{x:.1f}" y="{y - 10:.1f}" width="12" height="12" rx="3" class="{cls}"/>')
        self.text(x + 18, y, label, 'ink2', 12.5)

    def save(self):
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
               f'viewBox="0 0 {self.w} {self.h}" role="img" aria-labelledby="t d">\n'
               f'<title id="t">{esc(self.title)}</title>\n<desc id="d">{esc(self.desc)}</desc>\n'
               f'<style>{STYLE}</style>\n' + '\n'.join(self.parts) + '\n</svg>\n')
        (OUT / f'{self.name}.svg').write_text(svg, encoding='utf-8')
        print('wrote', OUT / f'{self.name}.svg')


def hbar(x0, x1, yc, h, r=4):
    """Horizontal bar from baseline x0 to x1: square at the baseline, rounded data end."""
    top, bot = yc - h / 2, yc + h / 2
    r = min(r, abs(x1 - x0) / 2, h / 2)
    if x1 >= x0:
        return (f'M{x0:.1f},{top:.1f} H{x1 - r:.1f} A{r:.1f},{r:.1f} 0 0 1 {x1:.1f},{top + r:.1f} '
                f'V{bot - r:.1f} A{r:.1f},{r:.1f} 0 0 1 {x1 - r:.1f},{bot:.1f} H{x0:.1f} Z')
    return (f'M{x0:.1f},{top:.1f} H{x1 + r:.1f} A{r:.1f},{r:.1f} 0 0 0 {x1:.1f},{top + r:.1f} '
            f'V{bot - r:.1f} A{r:.1f},{r:.1f} 0 0 0 {x1 + r:.1f},{bot:.1f} H{x0:.1f} Z')


def column(xc, y0, y1, w, rounded=True, r=4):
    """Vertical column from baseline y0 up to y1; rounded top unless it is a lower stack segment."""
    left, right = xc - w / 2, xc + w / 2
    r = min(r, (y0 - y1) / 2, w / 2) if rounded else 0
    if r <= 0:
        return f'M{left:.1f},{y0:.1f} V{y1:.1f} H{right:.1f} V{y0:.1f} Z'
    return (f'M{left:.1f},{y0:.1f} V{y1 + r:.1f} A{r:.1f},{r:.1f} 0 0 1 {left + r:.1f},{y1:.1f} '
            f'H{right - r:.1f} A{r:.1f},{r:.1f} 0 0 1 {right:.1f},{y1 + r:.1f} V{y0:.1f} Z')


def signed(v, digits):
    """Signed number with a true minus sign."""
    return f'{v:+.{digits}f}'.replace('-', '\u2212')


def scale(d0, d1, r0, r1):
    return lambda v: r0 + (v - d0) / (d1 - d0) * (r1 - r0)


def contributions(report):
    """Paired gain of each block on the final model, the hidden-knockout one emphasised."""
    c = report['comparisons']
    rows = [('Hidden-knockout correction', c['hidden KOs (final minus G)'], 'accent'),
            ('LightGBM instead of logistic regression', c['LightGBM vs logistic (final minus I)'], 'rest'),
            ("Player 1's history", c['P1 history (final minus F)'], 'rest'),
            ("Player 2's bench", c['P2 bench (H minus final)'], 'rest')]
    f = Figure('contributions', 720, 324, 'Where the gain comes from',
               'Change in CV accuracy (percentage points) from each block, paired on the same 15 folds; '
               'whiskers ±1 corrected SE',
               'Horizontal bars: hidden-knockout correction +1.15 points (±0.29), LightGBM instead of logistic '
               'regression +0.54 (±0.50), Player 1 history +0.17 (±0.35), Player 2 bench -0.04 (±0.35).')
    x = scale(-0.5, 1.75, 300, 680)
    top, row_h = 96, 44
    bottom = top + row_h * len(rows)
    for t in (-0.5, 0, 0.5, 1.0, 1.5):
        f.line(x(t), top - 12, x(t), bottom - 8, 'grid')
        f.text(x(t), bottom + 10, signed(t, 1) if t else '0', 'muted num', 12, 'middle')
    f.line(x(0), top - 12, x(0), bottom - 8, 'axis')
    for k, (label, cmp, cls) in enumerate(rows):
        yc = top + k * row_h + row_h / 2 - 8
        v, se = cmp['diff'] * 100, cmp['se'] * 100
        f.text(24, yc + 4.5, label, 'ink', 13)
        f.path(hbar(x(0), x(v), yc, 20), cls)
        f.line(x(v - se), yc, x(v + se), yc, 'err', 1.5)
        for e in (v - se, v + se):
            f.line(x(e), yc - 5, x(e), yc + 5, 'err', 1.5)
        f.text(x(v + se) + 8, yc + 4.5, signed(v, 2), 'ink num', 12.5, weight=600 if cls == 'accent' else None)
    f.text(x(0.625), bottom + 30, 'percentage points of CV accuracy', 'muted', 12, 'middle')
    f.save()


# Labels by configuration letter, so the script reads reports of any run.
ABLATION_LABELS = {
    'E': 'E  final model',
    'H': 'H  final + P2 bench',
    'F': 'F  final without P1 history',
    'C': 'C  B + P1 history',
    'D': 'D  C + P2 bench',
    'I': 'I  final, logistic regression',
    'B': 'B  state with hidden knockouts',
    'G': 'G  final without hidden knockouts',
    'R': 'R  baseline: no correction, no history',
    'A': 'A  state at face value',
}


def ablation(report):
    """Dot plot of every configuration's CV accuracy, the final model emphasised."""
    res = sorted(report['results'].values(), key=lambda r: -r['acc'])
    f = Figure('ablation', 720, 112 + 30 * len(res) + 40, 'CV accuracy of each configuration',
               '5-fold × 3 repeats, seed 2026. Dot: mean accuracy; line: range of the three repeats',
               'Dot plot sorted by accuracy: ' + ', '.join(
                   f'{ABLATION_LABELS[r["name"].split()[0]].replace("  ", " ")} {r["acc"]:.4f}' for r in res) + '.')
    x = scale(0.670, 0.705, 300, 680)
    top, row_h = 88, 30
    bottom = top + row_h * len(res)
    for t in (0.670, 0.680, 0.690, 0.700):
        f.line(x(t), top - 6, x(t), bottom, 'grid')
        f.text(x(t), bottom + 18, f'{t:.3f}', 'muted num', 12, 'middle')
    for k, r in enumerate(res):
        yc = top + k * row_h + row_h / 2
        letter = r['name'].split()[0]
        final = letter == 'E'
        cls = 'accent' if final else 'rest'
        f.text(24, yc + 4.5, ABLATION_LABELS[letter], 'ink', 13, weight=600 if final else None)
        lo, hi = min(r['repeat_acc']), max(r['repeat_acc'])
        f.line(x(lo), yc, x(hi), yc, f'{cls}-line', 2, 'round')
        f.dot(x(r['acc']), yc, cls)
        if letter in ('E', 'R', 'A'):
            f.text(x(hi) + 10, yc + 4.5, f'{r["acc"]:.4f}', 'ink num', 12.5, weight=600 if final else None)
    f.text(x(0.6875), bottom + 38, 'CV accuracy', 'muted', 12, 'middle')
    f.save()


def symmetry():
    """Player 1 knockouts against Player 2 visible plus estimated hidden knockouts."""
    f = Figure('ko_symmetry', 720, 380, "Estimated Player 2 knockouts match Player 1's",
               'Knockouts per battle by record range, over train and test (15,000 battles)',
               'Grouped columns per record range. P1 visible knockouts: 0.342, 0.580, 0.764. P2 visible: '
               '0.109, 0.175, 0.223; with the estimated hidden ones P2 reaches 0.331, 0.547, 0.729.')
    f.swatch(24, 86, 'accent', 'P1 knockouts (always visible)')
    f.swatch(250, 86, 'slot2', 'P2 knockouts, visible')
    f.swatch(430, 86, 'slot3', 'P2 knockouts, estimated hidden')
    y = scale(0, 0.8, 330, 120)
    left, right = 70, 690
    for t in (0, 0.2, 0.4, 0.6, 0.8):
        f.line(left, y(t), right, y(t), 'axis' if t == 0 else 'grid')
        f.text(left - 10, y(t) + 4, f'{t:.1f}', 'muted num', 12, 'end')
    group_w = (right - left) / 3
    for g in range(3):
        gc = left + group_w * (g + 0.5)
        x1, x2 = gc - 16, gc + 16
        f.path(column(x1, y(0), y(KO_P1[g]), 24), 'accent')
        f.text(x1, y(KO_P1[g]) - 8, f'{KO_P1[g]:.2f}', 'ink num', 12.5, 'middle')
        f.path(column(x2, y(0), y(KO_P2_VISIBLE[g]), 24, rounded=False), 'slot2')
        f.path(column(x2, y(KO_P2_VISIBLE[g]) - 2, y(KO_P2_TOTAL[g]), 24), 'slot3')
        f.text(x2, y(KO_P2_TOTAL[g]) - 8, f'{KO_P2_TOTAL[g]:.2f}', 'ink num', 12.5, 'middle')
        f.text(gc, y(0) + 22, KO_RANGES[g], 'ink2', 12.5, 'middle')
    f.save()


def leaderboard():
    """The team's public scores in submission order, with the public-split noise."""
    f = Figure('leaderboard', 720, 340, "Public leaderboard over the team's submissions",
               'Accuracy on the public part of the test set (about 1,500 battles); whiskers ±1 SE (±0.012)',
               'Five submissions in order: ' + ', '.join(f'{n} {s:.5f}' for n, _, s in LEADERBOARD) +
               f'. Reference line: idea1 CV accuracy {IDEA1_CV}.')
    y = scale(0.670, 0.715, 290, 90)
    left, right = 70, 640
    for t in (0.670, 0.680, 0.690, 0.700, 0.710):
        f.line(left, y(t), right, y(t), 'grid')
        f.text(left - 10, y(t) + 4, f'{t:.3f}', 'muted num', 12, 'end')
    f.line(left, y(IDEA1_CV), right, y(IDEA1_CV), 'accent-line', 1)
    f.text(right + 8, y(IDEA1_CV) + 4, 'idea1 CV', 'ink2', 12)
    step = (right - left) / len(LEADERBOARD)
    xs = [left + step * (k + 0.5) for k in range(len(LEADERBOARD))]
    pts = ' '.join(f'{px:.1f},{y(s):.1f}' for px, (_, _, s) in zip(xs, LEADERBOARD))
    f.parts.append(f'<polyline points="{pts}" class="rest-line" fill="none" stroke-width="2" '
                   f'stroke-linejoin="round" stroke-linecap="round"/>')
    for k, (px, (name, date, s)) in enumerate(zip(xs, LEADERBOARD)):
        last = k == len(LEADERBOARD) - 1
        f.line(px, y(s - PUBLIC_SE), px, y(s + PUBLIC_SE), 'err', 1.5)
        for e in (s - PUBLIC_SE, s + PUBLIC_SE):
            f.line(px - 5, y(e), px + 5, y(e), 'err', 1.5)
        f.dot(px, y(s), 'accent' if last else 'rest')
        f.text(px, y(0.670) + 20, name, 'ink', 12.5, 'middle', weight=600 if last else None)
        f.text(px, y(0.670) + 36, date, 'muted', 12, 'middle')
        if last:
            f.text(px + 12, y(s) - 8, f'{s:.3f}', 'ink num', 12.5, weight=600)
    f.save()


def main():
    report = json.loads((ROOT / 'outputs' / 'kaggle' / 'cv_report.json').read_text())
    contributions(report)
    ablation(report)
    symmetry()
    leaderboard()


if __name__ == '__main__':
    main()
