# tests/ — pytest suite

Proves behavior; grows with each part. Uses the real `config.yaml` at repo root
plus temporary configs written per-test. **How to run**: `make test`
(alias: `.venv/bin/pytest -q`) from the repo root. `tests/__init__.py` keeps the
package importable under the `src/` layout.