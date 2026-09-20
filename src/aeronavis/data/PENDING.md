# aeronavis.data — PENDING

- Synchronize IMU/GNSS/cam timestamps: nearest-neighbour vs linear interp
  (assumption: linear interp to IMU grid).
- Hand-held vs fixed-rig IMU noise parameters (Allan variance) — source in
  Part 1, hardcode in config.
- Window features: raw strap samples vs hand-designed (Δheading, accel mag)
  — decided by model Part 5 experiments.