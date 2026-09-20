# tests — PENDING

- Add slow-marked integration tests later (train smoke, full pipeline) and
  gate them behind `-m integration`; not needed for Part 0.
- Whether every part must ship its own test module or tests mirror the
  `src/aeronavis` tree 1:1 — assumption for now: mirror the tree.
- Keep `tests/__init__.py`? pytest rootdir handles `src/` layout via the
  editable install, so the package marker may become removable.