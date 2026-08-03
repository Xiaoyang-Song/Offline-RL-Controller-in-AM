"""
baselines/kalman_particle
----------------------------
Baseline 4: Kalman-filter and particle-filter controllers. Both filter a
synthetically-noised measurement of the scalar ROI-mean pre-heat
temperature (a random-walk process model + Gaussian sensor noise — the
process model a controller would actually have, NOT the full nonlinear PDE)
and apply certainty-equivalent control: invert a linear regression fit
(alpha, beta, gamma_c) from the offline dataset to pick the laser power that
aims the filtered temperature estimate at the target window's centre.

Expected (and intended) to underperform: a scalar linear/random-walk model
cannot capture the layer-dependent, spatially-varying nonlinear heat
transfer the real process (and the neural surrogate) actually has — which is
exactly the point of including it as a baseline.
"""
