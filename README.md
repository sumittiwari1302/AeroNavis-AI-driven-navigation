# AeroNavis 3.0

Offline AI-ML **dead reckoning** navigation engine for **Smart India Hackathon
2026**, Problem Statement **SIH26168** (ISRO). When GNSS cuts out, AeroNavis keeps
you on the road using learned prior data, onboard IMU, a pseudo-odometer, and
camera texture — fused in a right-invariant error-state Kalman filter.

Not a single model is trained in this part. Part 0 only lays the skeleton every
later part plugs into.

## Architecture

```
                    ┌──────────────┐   GNSS (1 Hz)  ──►  (disappears during outage)
                    │   sensors    │   IMU (100 Hz) ──►  ──┐
                    └──────────────┘   CAM (10 Hz)  ──►  ──┤
                                                           ▼
                       data/  ─►  src/aeronavis/data     ┌──────────────┐
                       raw downloaders             │   fusion     │
                       processed preprocess        │  InEKF + NHC │  ──► position,
                       splits    k-fold            └──────▲───────┘     heading
                                                           │      pseudo-odo/velocity
                    ┌──────────────┐   ┌──────────────┐    │      visual-odo
                    │    models    │   │   predict    │    │
                    │ velocity     │   │ outage       │    │   ┌──────────────┐
                    │ pseudo-odo   │   │ forecaster   │────┘   │     app      │
                    │ visual-odo   │   └──────────────┘ ── policy @ outage onset
                    │ adaptation   │                                │ maps/mapmatch/rancher
                    │ gates        │                     ┌──────────▼──────────┐
                    └──────────────┘                     │         eval        │
                                                         │ metrics/blackout/    │
                                                         │ adapt_eval           │
                                                         └──────────────────────┘
```

Pipeline contract (Parts 1–13): prior model → outage forecaster → gates →
InEKF/NHC fusion → map-matching → blackout evaluation. Config is the single
source of truth for every magic number (`config.yaml` + `src/aeronavis/config.py`).

## Layout

| Path | Purpose |
| --- | --- |
| `src/aeronavis/config.py` | Strict YAML loader with frozen dataclass spec |
| `src/aeronavis/data/` | Downloaders, preprocess, train/val splits |
| `src/aeronavis/models/` | Velocity, pseudo-odo, visual-odo, adaptation, gates |
| `src/aeronavis/predict/` | Outage forecaster (learned prior survivability) |
| `src/aeronavis/fusion/` | Right-invariant EKF + NHC |
| `src/aeronavis/app/` | Maps, map-matching, reference anchor logic |
| `src/aeronavis/eval/` | Metrics, blackout protocol, adaptation eval |
| `data/` `models/` `benchmarks/` `SUBMISSION/` | Artifact homes (gitignored) |
| `tests/` | Proves the skeleton; grows with each part |

## Research anchors

| Subsystem | Anchor |
| --- | --- |
| Velocity / pseudo-odo | UniCycle / residual odometry: award IMU-based speed with calibrated scale; CMU "phantom odometer" legacy |
| Visual odo | ORB-SLAM-style keypoint tracks; texture richness thresholds gate track quality |
| Adaptation | Recursive least squares speed-scale reset after each GNSS reacquire (cf. Learn-to-Adapt dead reckoning) |
| InEKF fusion | Right-invariant EKF (Barrau & Bonnabel); NHC applied on flat-road segments |
| Outage forecast | Survival modelling over prior route pass-throughs; horizon thresholds from `predict.*` |
| Map-match | Hidden-Markov + geometry snap to road graph; reranchor on known reference anchors |

## Quick start

```sh
make setup && source .venv/bin/activate
make test
make lint
```

See `RUNBOOK.md` for the exact macOS commands.

## Status

- Part 0 (scaffolding/config/CI hygiene): done.
- Part 1 (data downloaders/preprocess/splits): next.