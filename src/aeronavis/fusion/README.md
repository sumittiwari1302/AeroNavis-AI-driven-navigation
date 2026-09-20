# aeronavis.fusion — InEKF + NHC

Right-invariant error-state Kalman filter (Barrau & Bonnabel). Fuses IMU with
velocity/pseudo-odo/visual-odo pseudo-measurements; NHC (no-velocity in the
lateral/vertical body frame) constrains road-straight segments. A GNSS fix
greater than `fusion.reseed_jump_max_m` from the filter state is treated as a
reseeding event (reset), smaller ones as ordinary updates. Implemented in
Parts 2–3.