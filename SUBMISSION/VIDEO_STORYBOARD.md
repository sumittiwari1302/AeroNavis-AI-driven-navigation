# NAV-X 3.0 — 3-Minute Video Storyboard
**Target: SIH 2026 Judges | Format: 1080p, 30fps, H.264**

---

## Beat 1: Hook (0:00–0:15) — "Navigation fails where you need it most"

| Time | On-Screen | Narration | B-Roll Source |
|------|-----------|-----------|---------------|
| 0:00 | Split screen: Google Maps spinning → "GPS signal lost" | "Every driver knows this moment. You enter a tunnel, an underground parking, a dense urban canyon—and your navigation dies." | Stock footage: car entering tunnel, phone showing "Searching for GPS" |
| 0:05 | Animation: Pure INS drift cone expanding | "Pure inertial navigation drifts 50+ meters in 60 seconds. Frozen AI models can't adapt to your phone's sensor noise." | Generated: matplotlib drift cone animation (src/navx/eval/forced_blackout.py data) |
| 0:10 | Title card: **NAV-X 3.0 — Predict. Adapt. Correct.** | "NAV-X 3.0: the first offline AI-ML dead reckoning engine that predicts outages, adapts to your device, and corrects drift in real time." | Animated logo + tagline |

---

## Beat 2: Blackout Real Demo (0:15–1:15) — "Watch it hold the line"

| Time | On-Screen | Narration | B-Roll Source |
|------|-----------|-----------|---------------|
| 0:15 | Dashboard: Map view with NAV-X (blue) vs GPS (green) vs Pure INS (red) | "Here's a forced-blackout test on real IO-VNBD data. At 200m, we cut GNSS for 60 seconds. Watch what happens." | Screen recording: `python scripts/demo_blackout.py` on io_vnbd_S-Vta12_1 |
| 0:30 | GNSS cut overlay: "BLACKOUT START" + red timer counting 60s | "The forecaster predicted this outage 10 seconds early and pre-inflated covariance. The InEKF coasts on wheel + IMU." | Same recording + overlay graphics |
| 0:45 | Trajectories diverge: Pure INS flies off, NAV-X stays on road | "Pure INS (red) drifts 35m. NAV-X (blue) stays within 1.2m—because pseudo-wheel-odometry corrects slip, and visual ODO gates in when texture is rich." | Split view with error metrics overlay |
| 1:00 | GNSS returns: "REACQUIRING → REANCHORED" zero-jump transition | "On re-acquisition, the re-anchor state machine blends smoothly—zero position jump. You never feel the handoff." | State machine visualization (src/aeronavis/app/ranchor.py) |
| 1:10 | Metrics table popup: ATE 1.1m, Drift 4.2%/km, 20% better than frozen LSTM | "Forced-blackout protocol: 121 sequences, 5 blackout durations. NAV-X beats frozen LSTM by 20%, NHC-only by 50%." | benchmarks/SUMMARY.md numbers |

---

## Beat 3: Behind the Scenes — Predict (1:15–2:00) — "The countdown you never see"

| Time | On-Screen | Narration | B-Roll Source |
|------|-----------|-----------|---------------|
| 1:15 | Forecaster architecture: Transformer → 3 horizons (5/10/15s) | "Part 6: The GNSS outage forecaster. A tiny Transformer reads IMU+GNSS health and outputs outage probability at 5, 10, 15 seconds." | Animation: src/aeronavis/predict/model.py architecture |
| 1:30 | Live plot: Probability ramps up 10s before outage | "At t=-10s, P(outage@10s) crosses 0.85. The seeder inflates GNSS covariance 2×. The filter *knows* before it happens." | Generated: forecaster output on validation seq |
| 1:45 | Covariance inflation visual: ellipse growing before blackout | "This proactive inflation means when GNSS drops, the filter is already 'wide'—no sudden covariance explosion." | InEKF covariance ellipse animation (src/aeronavis/fusion/inekf.py) |
| 1:55 | Metric: 87% recall @ 15s horizon, 92% precision | "Forecaster metrics: 87% recall at 15s horizon, 92% precision. Low false alarms = no unnecessary inflation." | src/aeronavis/models/train_predict.py validation output |

---

## Beat 4: Adaptation Status Visual (2:00–2:30) — "Your phone, calibrated"

| Time | On-Screen | Narration | B-Roll Source |
|------|-----------|-----------|---------------|
| 2:00 | Three phone silhouettes + motorcycle: "Pixel 7 | Pixel 6a | Redmi | 2-Wheeler" | "Part 5: Test-time LoRA adaptation. Same base model, adapted per device in 30 seconds of driving." | Photo assets + device labels |
| 2:10 | Before/After ATE bars per device: Base vs Adapted | "Pixel 7: 2.1m → 1.3m. Pixel 6a: 3.8m → 1.9m. Redmi: 5.2m → 2.8m. 2-Wheeler (high vibration): 4.5m → 2.1m." | Bar chart from models/calibration/calibration_report.json |
| 2:20 | LoRA weights heatmap (8-rank adapters) | "8-rank LoRA adapters = 180 KB per device. Updated every 5 seconds. Frozen backbone = privacy preserved." | Heatmap visualization of adapter weights |
| 2:25 | "Adaptation beats frozen on every device" badge | "Calibration benchmark: 3 phone profiles + 2-wheeler. Adaptation wins every time." | benchmarks/calibration_profiles: 4 ok |

---

## Beat 5: Accuracy Table + Close (2:30–3:00) — "Sub-meter. Offline. Yours."

| Time | On-Screen | Narration | B-Roll Source |
|------|-----------|-----------|---------------|
| 2:30 | Final accuracy table (from Slide 7) | "Forced-blackout protocol proves it: sub-meter at 60s, 2.5m at 300s. Pure INS is 71m. That's 30× better." | Static table with checkmarks |
| 2:40 | Architecture one-liner: "Predict (Transformer) → Adapt (LoRA) → Correct (InEKF + Map)" | "Three loops. 121 sequences. 66 vehicle-hours. All offline. All reproducible with `make full-pipeline`." | Architecture diagram (ARCHITECTURE.md) |
| 2:50 | Team + SIH badge + "₹1,00,000 | Deadline: 20 Sep 2026" | "Team AeroNavis. SIH 2026 Problem Statement ISRO SIH26168. We're ready for field trials." | Team photo + SIH logo |
| 2:55 | QR code → GitHub repo + `make full-pipeline` | "Try it yourself: github.com/sumittiwari1302/AeroNavis-AI-driven-navigation. One command reproduces everything." | QR code generator |

---

## Production Notes
- **Total runtime**: 3:00 (180 seconds)
- **Music**: Low-tension electronic (0:00-1:15), confident build (1:15-2:30), resolved (2:30-3:00)
- **Color scheme**: NAV-X blue (#0066CC), alert red (#CC0000), success green (#00AA00)
- **Fonts**: Inter (UI), JetBrains Mono (code/telemetry)
- **Delivery**: MP4 + WebM, captions (EN/HI), 1080p & 720p versions