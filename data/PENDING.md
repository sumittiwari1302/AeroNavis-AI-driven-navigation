# data — PENDING

- Dataset finalization deferred to Part 1; unknown whether we may use
  downloaded public IMU/GNSS logs vs only ISRO-supplied flight data
  (open decision, SIH26168 constraints).
- Whether `processed` stores numpy `*.npz`, pandas `*.parquet`, or torch
  `*.pt` — pick in Part 1 after profiling read times.
- k-fold count for `splits/` (default assumption: 5) — decide in Part 1.
- `paths.*` are relative to repo root; if the pipeline ever runs from a
  different CWD, absolute resolution will be needed (follow-up).
- IO-VNBD: phone (S) CSVs use **two different gyroscope naming schemes**:
  some files have `GYROSCOPE X/Y/Z (rad/s)` (body-frame rates), others
  `GYROSCOPE Yaw/Pitch/Roll (rad/s)` (Euler-rate style). The parser in
  `src/aeronavis/data/preprocess.py:_find_gyro_columns` handles both; pitch→x,
  roll→y, yaw→z mapping is a convention and should be verified against
  the dataset paper. Recorded by LayoutError at preprocess.py:348.
- IO-VNBD: the "Unsynchronised" zip was not processed due to disk space
  constraints; only "Synchronised" subset (564 CSVs) is currently used.