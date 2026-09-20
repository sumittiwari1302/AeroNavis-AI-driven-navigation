# aeronavis.predict — PENDING

- Survival-model family (Kaplan-Meier vs Cox vs learned classifier on window
  features) — choose in Part 4.
- Label definition: horizon "success" = predicted drift under
  `fusion.reseed_jump_max_m`? Needs a Part 10 metric to anchor on.
- Prediction cadence: every IMU tick vs every `model.stride` window.