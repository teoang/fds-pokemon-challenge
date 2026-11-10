"""Small helpers: timestamped logging and safe column names."""

import re
import time

_START = time.time()


def log(msg):
    print(f'[{time.time() - _START:7.1f}s] {msg}', flush=True)


def clean(name):
    """Species, item or move name as a safe column-name fragment."""
    return re.sub(r'[^0-9a-zA-Z_]', '_', name)
