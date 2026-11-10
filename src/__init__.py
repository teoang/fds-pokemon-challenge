"""Compatibility imports for the original hidden-KO pipeline.

The unchanged implementation now lives in submissions/01_hidden_ko_history/code.
Existing ``from src import pipeline`` calls continue to use the repository's
original data and output locations.
"""

from pathlib import Path

_root = Path(__file__).resolve().parents[1]
__path__ = [str(_root / 'submissions/01_hidden_ko_history/code/src')]

from . import data as _data

_data.REPO_ROOT = _root
