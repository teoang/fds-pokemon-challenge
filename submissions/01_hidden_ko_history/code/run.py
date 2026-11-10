#!/usr/bin/env python3
"""Run the whole pipeline: repeated CV, ablation, final fit and submission.csv.

Usage, from the repository root:
    python run.py [--quick] [--no-ablation] [--configs A,B] [--seed 7] [--jobs 4] [--out folder]

It never submits anything to Kaggle.
"""

from src.pipeline import main

if __name__ == '__main__':
    main()
