# aeronavis.data — downloaders, preprocess, splits

- **Downloaders**: fetch public IMU/GNSS + prior-route logs into `data/raw`.
- **Preprocess**: resample to `sensor.*` rates, stack windows
  (`model.window`/`stride`), reject corrupt straps.
- **Splits**: write k-fold index files to `data/splits`.

**How to run** (Part 1+): `python -m aeronavis.data.download`. Empty in Part 0.