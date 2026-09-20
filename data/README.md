# data/ — artifact homes for the ingestion pipeline

- `raw/` — as-downloaded public datasets (Kalman-style IMU+GNSS logs, prior route
  traces). Written by `src/aeronavis/data` downloaders; gitignored.
- `processed/` — derived tensors/parquets after preprocess; gitignored.
- `splits/` — train/val/test index files (k-fold CSV/JSON); gitignored.

You never edit these by hand. **How to run** (Part 1+): from repo root,
`python -m aeronavis.data.download` and `python -m aeronavis.data.preprocess`. Until
Part 1 lands, these hold only `.gitkeep`.

Paths are resolved from `config.yaml` under `paths.*`.