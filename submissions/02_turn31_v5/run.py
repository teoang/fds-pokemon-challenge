#!/usr/bin/env python3
"""Run this submission through the shared I/O adapter."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from submission_tools.runner import main

if __name__ == "__main__":
    main(Path(__file__).resolve().parent)
