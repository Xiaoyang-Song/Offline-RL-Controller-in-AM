"""
baselines/constant
---------------------
Baseline 3: constant laser power. No state feedback at all — every layer
gets the same fixed power. Swept across the full training grid
(100, 110, ..., 400 W); the reported "Constant Policy" performance is the
MEAN return across that whole sweep (per the task spec), with the
per-constant breakdown also reported/plotted as a diagnostic.
"""
