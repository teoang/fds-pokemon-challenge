"""FDS Pokémon Battles 2026: predict the winner from the first 30 recorded turns.

The approach rebuilds what the timeline hides and gives the model Player 1's
history:

- `hidden_ko`: Player 2's knockouts are hidden in the data. Player 1's
  vanishings act as a control group for a classifier that estimates which
  Player 2 vanishings were knockouts.
- `features`: the state at the last record, corrected for those knockouts,
  plus Player 1's species and items and the move counts of both players.
- `history`: an out-of-fold target encoding of Player 1's exact team.
- `pipeline`: repeated cross-validation, the ablation and the final fit.

Run `python run.py` from the repository root, or call `src.pipeline.run()`
from a notebook.
"""
