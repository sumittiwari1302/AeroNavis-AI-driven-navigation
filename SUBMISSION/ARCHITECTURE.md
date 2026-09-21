# NAV-X 3.0 — Architecture Reference
**Version: 3.0 | SIH 2026 | ISRO SIH26168**

---

## 1. System Block Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            NAV-X 3.0 PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  SENSORS │───►│  PREPROCESS  │───►│  LEARNED     │───►│    FUSION    │  │
│  │  (IMU,   │    │  (Part 1)    │    │  MODELS      │    │  (InEKF)     │  │
│  │   GNSS,  │    │  IO-VNBD,    │    │  (Parts 2-4) │    │  + Seeder    │  │
│  │   Wheel, │    │  RONIN,      │    │              │    │  + Ranchor   │  │
│  │   Cam,   │    │  IDOL, Self) │    │  • Velocity  │    │  + Map Match │  │
│  │   Baro)  │    │              │    │  • Pseudo-   │    │              │  │
│  └──────────┘    └──────────────┘    │    Wheel     │    └──────┬───────┘  │
│                                       │    • Slip    │           │          │
│                                       │    • Texture │           ▼          │
│                                       │    • Visual  │    ┌──────────────┐  │
│                                       │    ODO       │    │   EXPORT     │  │
│                                       └──────────────┘    │  (Part 9)    │  │
│                                                              │  • TFLite    │  │
│                                                       ┌─────►│  • Parity    │  │
│                                                       │      │  • Android   │  │
│                                       ┌──────────────┐  │      └──────────────┘  │
│                                       │  ADAPTATION  │  │                       │
│                                       │  (Part 5)    │  │                       │
│                                       │  • LoRA      │  │                       │
│                                       │  • 3 phones  │  │                       │
│                                       │  • 2-Wheeler │  │                       │
│                                       └──────────────┘  │                       │
│                                                       │                       │
│                                       ┌──────────────┐  │                       │
│                                       │  PREDICT     │──┘                       │
│                                       │  (Part 6)    │                          │
│                                       │  • Forecaster│                          │
│                                       │  • 3-15s     │                          │
│                                       └──────────────┘                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Flow (Per-Timestep)

```
100 Hz IMU ───► Predict (velocity) ──► v_body
                 │
                 ├─► Pseudo-Wheel-Odo ──► v_wheel + reliability
                 │
                 ├─► Slip Classifier ──► gate ∈ {full, low, zupt}
                 │
                 └─► Forecaster ──► P(outage@5/10/15s) ──► Seeder ──► Cov inflation

1 Hz GNSS  ───► ENU projection ──► z_gnss = [x, y, ψ] ──► InEKF.correct_gnss()

10 Hz Wheel ───► Slip-gated ──► z_wheel = vx_body ──► InEKF.correct_wheel() + NHC

5 Hz VO  ───► Texture gate ──► z_vo = [dx, dy, dψ] ──► InEKF.correct_vo()

Map Match ──► Viterbi snap ──► z_map = [x, y, ψ] ──► InEKF.correct_gnss() (tight R)

Re-Anchor SM: DROPPED → COASTING → REACQUIRING → REANCHORED
   │              │               │                 │
   ▼              ▼               ▼                 ▼
Cov×2×       Cov×8×         Blend α(t)        Cov×0.1, trust GNSS
```

---

## 3. Module → Research Paper Mapping

| Module | File | Research Basis | Key Parameters |
|--------|------|----------------|----------------|
| Velocity LSTM/TCN | `src/aeronavis/models/velocity.py` | RoNIN (Chen et al., 2018), EqNIO (Huai et al., 2022) | 480K params, window=200, TCN dilations 1-16 |
| Pseudo Wheel ODO | `src/aeronavis/models/pseudo_odo.py` | DeepOdo (Li et al., 2020) | 280K params, spectrogram 8-bin (0-5Hz) |
| Slip Classifier | `src/aeronavis/models/slip.py` | Shared trunk with pseudo-odo | 180K params, 3-class (grip/slip/stationary) |
| Texture Gate | `src/aeronavis/models/texture_gate.py` | UL-VIO (Li et al., 2021) keypoint quality | 90K params, 96×96 grayscale → 3-class |
| Visual ODO | `src/aeronavis/models/visual_odo.py` | UL-VIO, TartanVO | 980K params, 224×224 stereo pair → 6-DoF |
| InEKF (SE(2)) | `src/aeronavis/fusion/inekf.py` | Right-Invariant EKF (Bonnabel, 2007) | 10-state, process noise diag, residual gate 6σ |
| Outage Seeder | `src/aeronavis/predict/seeder.py` | Predictive covariance control | Horizons 5/10/15s, threshold 0.85 |
| Re-Anchor SM | `src/aeronavis/app/ranchor.py` | Zero-jump handover (Groves, 2013) | Blend 5s, max jump 0.5m, 3 consecutive good |
| Map Match | `src/aeronavis/app/mapmatch.py` | HMM Viterbi (Newson & Krumm, 2009) | Candidate radius 25m, window 20 |
| LoRA Adapt | `src/aeronavis/models/adaptation_engine.py` | LoRA (Hu et al., 2021), test-time adaptation | Rank 8, α=16, 180 KB/device |
| Forecaster | `src/aeronavis/predict/model.py` | Transformer (Vaswani et al., 2017) | 630K params, d_model=128, 3 layers |

---

## 4. Repository Layout

```
nav-x/
├── config.yaml                 # Single source of truth (flat namespaced)
├── Makefile                    # setup, format, lint, test, bench, full-pipeline
├── pyproject.toml              # Package metadata, deps, ruff/pytest/mypy config
├── scripts/
│   ├── full_pipeline.sh        # One-command reproduction
│   ├── bench_summary.py        # SUMMARY.md generator
│   └── collect_submission.py   # SUBMISSION/ packager
├── src/
│   ├── aeronavis/              # Core library (pip install -e .)
│   │   ├── config.py           # Strict YAML→dataclass loader
│   │   ├── data/               # Part 1: preprocessing, splits, manifest
│   │   ├── models/             # Parts 2-4, 6: all learned models + trainers
│   │   ├── fusion/             # Part 7: InEKF, NHC, learned noise
│   │   ├── predict/            # Part 6: forecaster, features, seeder
│   │   ├── app/                # Parts 7-8: ranchor, mapmatch, maps
│   │   └── eval/               # Part 10: metrics, forced_blackout, adapt_eval
│   └── navx/                   # Thin CLI namespace
│       └── eval/               # Re-exports for python -m navx.eval.*
├── models/                     # Trained artifacts (gitignored, rebuilt by pipeline)
│   ├── velocity/best.pt
│   ├── pseudo_odo/best.pt
│   ├── slip/best.pt
│   ├── texture_gate/texture_gate.pt
│   ├── visual_odo/visual_odo.pt
│   ├── predict/forecaster.pt
│   ├── adaptation/stable.pt
│   ├── calibration/calibration_report.json
│   └── export/navx_bundle.tflite
├── data/
│   ├── raw/                    # Downloaded corpora (gitignored)
│   ├── processed/              # NavSequence cache + manifest.json
│   ├── splits/                 # Train/val/test manifests per source
│   └── maps/                   # OSM PBF + extracted graph
├── benchmarks/
│   ├── blackout_comparison.json
│   ├── SUMMARY.md
│   └── calibration_report.json
├── tests/                      # Unit tests (pytest)
└── SUBMISSION/                 # Judge-facing deliverables
    ├── DECK_10_SLIDES.md
    ├── VIDEO_STORYBOARD.md
    ├── ARCHITECTURE.md
    ├── ROADMAP.md
    ├── QA.md
    └── CHECKMETA.md
```

---

## 5. Quickstart for Reviewers

```bash
# 1. Clone
git clone https://github.com/sumittiwari1302/AeroNavis-AI-driven-navigation.git
cd AeroNavis-AI-driven-navigation

# 2. One-command full reproduction (1-2 hrs on CPU, ~30 min with GPU)
make full-pipeline

# 3. Or step-by-step:
make setup              # Creates .venv, installs deps
make test               # Runs 41 unit tests
make bench              # Runs forced-blackout + calibration → benchmarks/SUMMARY.md

# 4. Inspect results
cat benchmarks/SUMMARY.md
cat models/calibration/calibration_report.json
cat benchmarks/blackout_comparison.json

# 5. Run a single blackout test
python -m navx.eval.forced_blackout \
    --sequence data/processed/io_vnbd/io_vnbd_S-Vta12_1 \
    --distances 100,200,300 \
    --durations 10,10,10 \
    --output benchmarks/test_blackout

# 6. Run calibration benchmark
python -m navx.eval.adapt_eval \
    --profiles phone_high_end,phone_mid_range,two_wheeler \
    --epochs 5 \
    --output models/calibration
```

---

## 6. Configuration Schema (config.yaml)

All keys are flat namespaced: `namespace.key`. Every key has a matching frozen dataclass field in `src/aeronavis/config.py`.

Key sections:
- `sensor.*` — Native hardware sampling rates
- `model.*` — Window/stride, hidden sizes, param budgets
- `adaptation.*` — LoRA rank/alpha, loss weights, update schedule
- `fusion.*` — Process/measurement noise, numerical safety
- `predict.*` — Forecaster horizons, threshold
- `data.*` — Target rates, quality gates, gap handling
- `paths.*` — Artifact locations (relative to repo root)

---

## 7. Key Invariants (Enforced by Code)

1. **Config is frozen** — `get_config()` returns immutable `Config` dataclass
2. **No magic numbers** — All thresholds in config.yaml
3. **Idempotent stages** — `best.pt` exists → skip retrain (unless `--force`)
4. **Single manifest** — `data/processed/manifest.json` tracks all sequences
5. **Deterministic** — Seed 42, `torch.use_deterministic_algorithms(True)`
5. **No legacy names** — IO-VNBD everywhere (not 10-VNBD)

---

## 8. Metrics Contract (Part 10)

All metrics write JSON in consistent schema:
```json
{
  "ate": {"rmse": 1.23, "mean": 0.98, "median": 0.87, "p95": 2.1, "max": 3.5},
  "rte": {"window_lengths": [10,30,60,120,300], "rmse_per_window": [...], "mean": 1.1, "median": 0.9, "p95": 2.3},
  "drift": {"drift_pct_per_km": 4.2, "endpoint_error_m": 12.3, "traveled_distance_m": 2900},
  "heading_error": {"mean_deg": 3.2, "median_deg": 2.8, "p95_deg": 6.1, "max_deg": 12.4},
  "latency": {"p50_ms": 8.4, "p95_ms": 12.1, "mean_ms": 9.2, "std_ms": 1.8}
}
```

---

## 9. License & Attribution

- Code: Apache-2.0 (or as specified in repo)
- Datasets: IO-VNBD (CC-BY-4.0), RONIN (custom), IDOL (CC-BY-4.0)
- Research citations: See module→paper mapping table above