# NAV-X 3.0 — 6-Month Roadmap
**Honest assessment: what we'd build with 6 more months + ISRO mentorship**

---

## Month 1-2: Field Trials & Validation

### 1.1 1000+ km Across 5 Indian Cities
| City | Environment | Target km | Key Challenges |
|------|-------------|-----------|----------------|
| Delhi | Metro, tunnels, flyovers | 250 | Multipath, tall buildings |
| Bangalore | Tech corridors, elevation | 200 | Tree canopy, aggressive driving |
| Mumbai | Coastal, dense urban | 200 | Sea-level multipath, traffic |
| Chennai | Industrial, highways | 150 | Heat, vibration (2W) |
| Hyderabad | Mixed, ORR + old city | 200 | Elevation changes, tunnels |

**Deliverable**: Field trial report with per-environment ATE, blackout recovery stats, user study.

### 1.2 NavIC L1/L5 Integration
- Dual-frequency ionospheric correction (eliminate 1st-order iono error)
- NavIC-specific signal health monitoring (C/N0, PRN monitoring)
- IRNSS ICD compliance testing
- **Target**: 30% GNSS position error reduction in urban canyons

---

## Month 3: Deployment Hardening

### 3.1 Quantization-Aware Training (QAT)
- INT8 quantization for all 7 models (velocity, pseudo-odo, slip, texture, VO, forecaster, adaptation)
- PyTorch QAT → ONNX → TensorFlow Lite conversion
- Android NNAPI / TFLite GPU delegate benchmarking
- **Target**: <5% accuracy drop, <50ms inference on Snapdragon 778G

### 3.2 Android App v1.0
- Foreground service for continuous navigation
- Sensor fusion at 100Hz (IMU) + 1Hz (GNSS) on background thread
- Battery optimization: <5% drain/hour (target 2mAh/hour)
- Offline map tiles (MBTiles) + OSM graph
- **Target**: Play Store beta release

---

## Month 4: Semantic SLAM Fusion

### 4.1 Semantic Landmarks
- Road marking detection (lane lines, crosswalks, arrows) — MobileNetV3-small
- Traffic sign classification (speed limit, stop, yield) — EfficientNet-B0
- Building facade matching (ORB + learned descriptors)
- **Integration**: Semantic factor graph in InEKF (additional measurement model)

### 4.2 Loop Closure & Global Consistency
- Visual bag-of-words (DBoW2) for place recognition
- Pose graph optimization (g2o) for trajectory correction
- Submap merging for multi-session consistency
- **Target**: <0.5m global drift over 10km

---

## Month 5: Multi-Phone Federated Adaptation

### 5.1 Privacy-Preserving Federated LoRA
- Each phone trains local LoRA adapters (never leaves device)
- Secure aggregation (SecAgg) for global adapter update
- Differential privacy (ε=1.0) on adapter deltas
- **Target**: Fleet-level adaptation without raw data upload

### 5.2 Cross-Device Generalization
- Domain adaptation: phone → 2-wheeler → vehicle
- Meta-learning initialization (MAML-style) for faster per-device convergence
- **Target**: <10s adaptation on new device type

---

## Month 6: SIH Final + Commercial Pilot

### 6.1 SIH 2026 Final Submission
- Complete technical report (50 pages)
- Live demo at finale (real-time navigation in GNSS-denied zone)
- Video + slide deck + architecture doc + QA report
- **Target**: Top 3 in ISRO SIH26168

### 6.2 Commercial Pilot with OEM
- Integration with 2-wheeler OEM (TVS / Bajaj / Ather)
- Fleet management dashboard (real-time position, health, blackout alerts)
- API for logistics partners (last-mile delivery tracking)
- **Target**: LOI for 10,000 unit pre-order

---

## Resource Estimates

| Activity | Compute | Data | Personnel | Budget (INR) |
|----------|---------|------|-----------|--------------|
| Field trials | 5 phones × 200h | 1000 km GPS+IMU | 2 engineers + 4 drivers | 8,00,000 |
| NavIC integration | 1 GPU × 100h | NavIC recordings | 1 RF engineer | 4,00,000 |
| QAT + Android | 2 GPUs × 200h | Calibration datasets | 2 ML engineers | 6,00,000 |
| Semantic SLAM | 4 GPUs × 300h | Mapillary + custom | 2 CV engineers | 12,00,000 |
| Federated LoRA | 4 GPUs × 200h | Synthetic + real | 2 ML engineers | 8,00,000 |
| SIH final + pilot | - | Demo hardware | Full team | 5,00,000 |
| **Total** | | | | **43,00,000** |

---

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| NavIC hardware unavailable | Medium | High | Simulate with GNSS + synthetic iono; partner with ISRO SAC |
| QAT accuracy drop >10% | Low | Medium | Mixed-precision fallback; keep FP16 models as backup |
| Field trial delays (weather/permits) | High | Medium | Parallel indoor/parking garage dataset collection |
| Federated privacy audit failure | Low | High | Pre-audit with CERT-In empaneled auditor |
| OEM integration complexity | Medium | High | Define clean SDK interface early; use ROS2 bridge |

---

## Success Metrics (Month 6)

| Metric | Current (Part 10) | Month 6 Target |
|--------|-------------------|----------------|
| 60s blackout ATE | ≤1.0 m | ≤0.5 m |
| 300s blackout ATE | ≤2.5 m | ≤1.0 m |
| End-to-end latency | 12ms p95 | <8ms p95 (INT8) |
| Battery drain | N/A (offline) | <2 mAh/hour |
| Adaptation time | 30s | <10s (meta-learned) |
| Global drift (10km) | N/A | <0.5m |
| NavIC urban accuracy | N/A | 30% better than GPS-only |

---

## Appendix: Research Publications Target

1. **"NAV-X: Predict-Adapt-Correct for GNSS-Denied Navigation"** — IROS 2026 / ICRA 2027
2. **"Federated LoRA for On-Device Navigation Adaptation"** — NeurIPS 2026 Workshop
3. **"Semantic Factor Graphs for Visual-Inertial Navigation"** — CVPR 2027
4. **"NavIC L1/L5 Dual-Frequency Ionospheric Correction for Consumer Devices"** — ION GNSS+ 2026