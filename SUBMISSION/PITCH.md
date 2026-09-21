# NAV-X 3.0 — Pitch Rehearsal Pack
**SIH 2026 | PS: ISRO SIH26168 | One-honest-line answers for every tough question**

---

## Core FAQ

### Q: "Why phone-only instead of RTK?"
**A:** RTK needs correction networks (CORS/NTRIP) — fails in tunnels, urban canyons, rural India. NAV-X is **offline-first**; works where RTK literally cannot reach. *Evidence: Part 7 InEKF runs 100% offline; Part 10 forced-blackout at 300s.*

### Q: "Why not ORB-SLAM / VINS-Mono / OpenVINS?"
**A:** Scene-fragile in tunnels/low-texture/night; heavy (50–200 MB, GPU). We **gate VO by texture** (Part 4) and only enable when reliable — otherwise fall back to learned wheel+IMU. *Evidence: Part 4 texture_gate.pt (90 KB) gates visual_odo.pt (980 KB).*

### Q: "Is sub-meter a real claim?"
**A:** **Only within the ≤ 60 s GNSS-blackout envelope**, per Part 10 forced-blackout protocol (121 sequences, 66.7 vehicle-hours). Beyond 60 s we quote 1–2.5 m. We do NOT claim long-term drift-free. *Evidence: benchmarks/SUMMARY.md, HONESTY.md.*

### Q: "Dataset is UK/NG/FR — Indian roads?"
**A:** Self-collected demo drives + **LoRA fine-tune in 30 s per device** (Part 5). Transfer from Part 2: frozen UK model → Indian fine-tune drops ATE 35% on local data. *Evidence: models/calibration/calibration_report.json, scripts/collect_self_data.py.*

### Q: "How is this different from NAVIQ / DRISHTI / VeTorch / TERN / Pendulum / Point One / IDCITI?"
**A:** Three unique pillars: **Predict** (outage forecaster), **Adapt** (LoRA per-device), **Correct** (learned slip + VO gate). Zero hardware, offline, <15 ms CPU. See competitor table below.

### Q: "Two-wheeler dynamics?"
**A:** Bike-mode adds lean compensation (gyr_x bias estimation) + slip classifier trained on motorcycle data (Part 3). Vibration profile in calibration (two_wheeler profile). *Evidence: models/calibration/calibration_report.json two_wheeler profile.*

### Q: "Battery?"
**A:** 10 Hz GNSS + 100 Hz IMU sampling; VO throttled to 2 Hz when texture POOR; early-exit MEIO-Net-style velocity head. Target **< 2 mAh/hour** on Snapdragon 778G. *Evidence: scripts/ci_budget.sh latency budget, Part 9 export quantization.*

### Q: "What if GNSS is spoofed?"
**A:** Forecaster + seeder detects >5 m jump vs predicted cov; map match consistency check; Ranchor requires 3 consecutive good fixes (config: `ranchor.consecutive_good=3`). *Evidence: Part 7 ranchor.py, Part 6 seeder.py.*

---

## Competitor Rebuttals (2 lines each + evidence)

| Competitor | Rebuttal | Evidence |
|------------|----------|----------|
| **NAVIQ** | Cloud-dependent (needs 4G/5G for corrections); fails offline. NAV-X is 100% offline, edge inference. | `src/aeronavis/fusion/inekf.py` (no network calls), `benchmarks/SUMMARY.md` (offline blackout) |
| **DRISHTI** | Visual-inertial only; collapses in tunnels/night. NAV-X has learned wheel + slip + forecaster as backups. | `src/aeronavis/models/pseudo_odo.py`, `src/aeronavis/predict/seeder.py` |
| **VeTorch** | Research code, no mobile export, no adaptation. NAV-X has TFLite bundle + LoRA per-device. | `src/navx/export/`, `src/aeronavis/models/adaptation_engine.py` |
| **TERN** | Tightly-coupled VI-SLAM; heavy GPU, no wheel odometry. NAV-X fuses wheel+slip for 2-wheeler. | `src/aeronavis/fusion/inekf.py` (wheel+NHC), `src/aeronavis/models/slip.py` |
| **Pendulum** | Factor graph SLAM; needs loop closure, heavy. NAV-X is filtering (InEKF) + map match — lighter, real-time. | `src/aeronavis/fusion/inekf.py` (10-state EKF), `src/aeronavis/app/mapmatch.py` |
| **Point One** | Commercial RTK + IMU; needs base station / NTRIP. NAV-X needs zero infrastructure. | `HONESTY.md` (RTK non-claim), Part 7 (no GNSS corrections needed) |
| **IDCITI** | Carrier-phase + IMU; survey-grade hardware. NAV-X runs on phone IMU (code-phase GNSS). | `config.yaml` (sensor.imu_hz=100), `src/aeronavis/models/velocity.py` (phone IMU only) |
| **Carrier-based (Jio/BSNL RTK)** | Requires SIM + correction stream; dead in tunnels/remote. NAV-X works air-gapped. | Part 6 forecaster predicts outage, Part 7 seeder inflates cov proactively |

---

## Design Decision Defense Map

| Decision | Why | Evidence |
|----------|-----|----------|
| **SE(2) InEKF not SE(3)** | Ground vehicle → 2D dominant; 10-state vs 15-state = 40% faster | `src/aeronavis/fusion/inekf.py` state_dim=10 |
| **Right-invariant EKF** | Better consistency on manifold; standard in robotics (Bonnabel 2007) | `src/aeronavis/fusion/inekf.py` class InEKF |
| **TCN + LSTM fusion** | TCN: long context; LSTM: temporal dynamics; RoNIN/EqNIO proven | `src/aeronavis/models/velocity.py` TCNEncoder + LSTMEncoder |
| **LoRA not full fine-tune** | 180 KB vs 500 KB; privacy (base frozen); 30 s adaptation | `src/aeronavis/models/adaptation_engine.py` rank=8 |
| **Texture gate VO** | Prevents VO hallucination in tunnels/rain; 90 KB overhead | `src/aeronavis/models/texture_gate.py`, `src/aeronavis/models/visual_odo.py` |
| **Forecaster 3-horizon** | 5/10/15 s actionable for seeder inflation; longer = unreliable | `src/aeronavis/predict/model.py` horizons=[5,10,15] |
| **Ranchor state machine** | Zero-jump re-anchor required for UX; DROPPED→COASTING→REACQUIRING→REANCHORED | `src/aeronavis/app/ranchor.py` RanchorStateMachine |
| **Slip classifier 3-class** | Grip/Slip/Stationary → gate wheel correction; not binary | `src/aeronavis/models/slip.py` label ∈ {0,1,2} |
| **Pseudo-odo spectrogram** | Vibration fingerprint (0-5 Hz) distinguishes grip vs slip | `src/aeronavis/models/pseudo_odo.py` SpectrogramExtractor 8-bin |

---

## Numbers to Memorize

| Metric | Value | Source |
|--------|-------|--------|
| Forced-blackout protocol sequences | **121** | `data/processed/manifest.json` |
| Vehicle-hours in benchmark | **66.7** | Same |
| Velocity model params | **480K** | `models/velocity/best.pt` |
| Pseudo-odo + slip params | **280K + 180K** | `models/pseudo_odo/best.pt`, `models/slip/best.pt` |
| Texture gate params | **90K** | `models/texture_gate/texture_gate.pt` |
| Visual ODO params | **980K** | `models/visual_odo/visual_odo.pt` |
| Forecaster params | **630K** | `models/predict/forecaster.pt` |
| LoRA adapter per device | **180 KB** | `models/adaptation/stable.pt` |
| Inference+fusion p95 (CPU) | **≤ 15 ms** | `scripts/ci_budget.sh` |
| Bundle target | **≤ 15 MB** | Part 9 export |
| 60s blackout ATE (VO on) | **≤ 1.0 m** | `benchmarks/SUMMARY.md` |
| 300s blackout ATE | **≤ 2.5 m** | Same |
| Drift rate | **1–2 %/km** | `benchmarks/calibration_report.json` |

---

## "We Will Not Demo" List

1. **Pedestrian inside mall without map** — no Wi-Fi/BLE, no step detector
2. **Multi-floor parking garage (z-accuracy)** — barometer drift > 5 m
3. **Off-road dirt trail** — slip classifier trained on paved roads
4. **Aggressive spoofing (crypto-level)** — we detect gross jumps only
5. **> 5 min tunnel without map** — forecaster horizon 15 s max
6. **Underwater / subway** — no GNSS, no map, IMU-only > 10 min not supported

---

## Closing Line

> "NAV-X 3.0 doesn't pretend to beat physics. It **predicts** when physics will hurt you, **adapts** to your specific phone, and **corrects** with every sensor available — all offline, all on-device, all auditable. That's the difference between a demo and a product."

---

## Appendix: Quick Links for Live Demo

| Demo | Command | Duration |
|------|---------|----------|
| Forced blackout (60s) | `python -m navx.eval.forced_blackout --sequence data/processed/io_vnbd/io_vnbd_S-Vta12_1 --distances 100,200 --durations 10,10` | 30 s |
| Adaptation (phone profile) | `python -m navx.eval.adapt_eval --profiles phone_mid_range --epochs 1` | 15 s |
| Full pipeline | `make full-pipeline --skip-train` | 2 min |
| Budget check | `scripts/ci_budget.sh` | 10 s |
| Edge cases | `python -m navx.eval.edge_cases` | 1 min |