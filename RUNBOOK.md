# AeroNavis 3.0 — Runbook

Exact steps to go from a fresh clone to a working repo on macOS (darwin, arm64 or
x86_64). No manual steps beyond these.

## 0. Prerequisites

- macOS with Command Line Tools: `xcode-select --install`
- A Python 3.10+ interpreter. Python 3.11 is a safe default:
  `brew install python@3.11`
- Optionally `uv` (`brew install uv`) to fetch interpreters faster, but it is not
  required — the Makefile works with plain venv.

## 1. Setup

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

Or, one command from the repo root:

```sh
make setup && source .venv/bin/activate
```

### torch wheel note (macOS)

On macOS do **not** install a separate CPU wheel. `torch==2.3.1` ships universal2
wheels for Darwin that run on both arm64 and x86_64 with the CPU/MPS backends. If
you are on Linux and want CPU-only, instead run:

```sh
python -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
```

## 2. The three commands that must pass

```sh
make test
make lint
make format
```

`make format` is idempotent; after a correct checkout `make format` introduces no
diffs. The full set of green gates is:

```sh
make setup && make test && make lint
```

## 3. Smoke check

```sh
source .venv/bin/activate
python -c "from aeronavis.config import get_config; print(get_config().sensor.imu_hz)"
# expected: 100
```

## 4. What is implemented so far

Part 0 only: packaging, strict config loader, tooling, data-dir hygiene. Model
code, benchmarks, and the full pipeline land in Parts 1–13.

## 5. Troubleshooting

- `command not found: python3.11` → install Python 3.11, or point the Makefile at
  another interpreter: `make setup PYTHON=python3.12`.
- `ModuleNotFoundError: aeronavis` under a different interpreter → you are not in the
  `.venv`; re-run the `source .venv/bin/activate` line above.
- Install is slow on py3.12 → torch 2.3.1 needs py3.11/3.10 wheels; prefer 3.11.