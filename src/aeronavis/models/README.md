# aeronavis.models — learned dead-reckoning networks

- **velocity**: IMU-window → forward speed (speed estimator).
- **pseudo_odo**: odometry-style residual speed, acts as fake wheel odometer.
- **visual_odo**: keypoint-track displacement gated by texture thresholds.
- **adaptation**: RLS speed-scale re-calibration after reacquire.
- **gates**: when to trust/drop each source.

All share `model.*` window/stride/budget from config. Implemented from Part 5.