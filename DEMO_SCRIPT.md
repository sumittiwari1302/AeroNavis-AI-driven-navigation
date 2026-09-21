# NAV-X 3.0 Demo Script

## Overview
5-minute live demonstration of the NAV-X 3.0 offline AI-ML dead reckoning navigation engine for SIH 2026 (ISRO Problem Statement SIH26168).

**Hardware**: Android phone (mid-range, e.g., Snapdragon 7xx) mounted on vehicle dashboard.
**Software**: NAV-X 3.0 app with all TFLite models bundled offline.

---

## Demo Flow (5 minutes)

### 0:00 - 0:30 | Setup & Calibration
1. **Launch app** - Shows map with current position (blue dot), dashboard overlay
2. **Verify sensors** - Green checkmarks for IMU, Wheel, VO, GNSS
3. **Calibration** - Drive straight 10s for IMU alignment
4. **Start recording** - Press "Start Recording" (logs to CSV)

### 0:30 - 1:30 | Normal Driving (GNSS Available)
- Drive on open road with good GNSS
- Dashboard shows: Velocity ~10 m/s, Heading ~90°, σx/σy < 1m
- All sensors green: IMU ✓ Wheel ✓ VO ✓ GNSS ✓
- Outage forecast: 5s/10s/15s < 10% (green)
- Map shows snapped trajectory (blue line)

### 1:30 - 2:30 | Tunnel Entry (Simulated Blackout)
**Action**: Press "Simulate Blackout" button OR drive into tunnel
- **0:00** Blackout begins: GNSS lost
- **Dashboard changes**:
  - GNSS status → RED (OFFLINE)
  - Outage forecast: 5s/10s/15s → 95%/98%/99% (RED)
  - "OUTAGE MODE" banner appears
  - Covariance inflation starts (ellipse grows)
- **InEKF behavior**:
  - Switches to IMU + Wheel + VO dead-reckoning
  - Covariance grows smoothly
  - VO gated by texture (if tunnel dark, VO drops)
  - Wheel odometry carries forward

### 2:30 - 3:30 | Tunnel Transit (60s blackout)
- **Countdown timer**: 60s → 0s (red)
- **Covariance ellipse** visible on map (growing)
- **Adaptation active**:
  - Wheel speed adapts to tire pressure changes
  - Slip detector: "SLIP" → "STATIONARY" → "GRIP"
  - Velocity bias adapts to tire pressure
- **Texture gate**: In tunnel → POOR (VO disabled automatically)

### 3:30 - 4:00 | Tunnel Exit (Re-acquisition)
- **GNSS re-acquired**: Signal acquired
- **Re-anchor sequence**:
  1. REACQUIRING (yellow): Blending INS→GNSS over 5s
  2. REANCHORED: Green checkmark, covariance collapse
  3. Position jump < 0.5m (verified on map)
- **Handover complete**: Covariance collapses, ellipse shrinks

### 4:00 - 4:30 | Post-Recovery Verification
- Drive 30s with GNSS
- Verify ATE < 5m (displayed on dashboard)
- Covariance ellipse < 2m
- All sensors GREEN

### 4:30 - 5:00 | Recording Review & Handoff
- Stop recording
- Export CSV via adb
- Show dashboard summary:
  - Total distance, max outage duration
  - Max ATE during outage
  - Adaptation stats (velocity bias, slip events)

---

## Simulated Blackout Controls (Hidden Dev Menu)
Long-press dashboard title 3x → Dev Menu:
- [ ] Simulate GNSS Blackout (60s)
- [ ] Simulate VO Drop (texture=poor)
- [ ] Simulate Wheel Slip (gate=slip)
- [ ] Simulate Wheel Lock (gate=zupt)
- [ ] Inject GNSS Multipath (noise + bias)
- [ ] Inject IMU Bias (accel/gyro drift)

---

## Recording & Data Export
```bash
# Pull recorded data
adb pull /sdcard/navx/recordings/ .

# Files generated:
# - session_YYYYMMDD_HHMMSS_imu.csv
# - session_YYYYMMDD_HHMMSS_gnss.csv
# - session_YYYYMMDD_HHMMSS_wheel.csv
# - session_YYYYMMDD_HHMMSS_vo.csv
# - session_YYYYMMDD_HHMMSS_events.json
# - session_YYYYMMDD_HHMMSS_meta.json
```

---

## Smoke Test (Automated)
```bash
# Run instrumentation tests
./gradlew connectedAndroidTest

# Test cases:
# 1. testSensorHubStartup - sensors initialize within 2s
# 2. testPipelineLatency - end-to-end < 15ms
# 3. testInEKFAccuracy - ATE < 1m (straight), 1.5m (turn)
# 3. testAdaptationConvergence - LoRA converges in <50 steps
# 4. testReanchorJump - jump < 0.5m on re-acquisition
# 5. testTextureGate - rich/medium/poor classification
# 6. testMapMatching - snap within 5m on highway
# 7. testReanchorJump - handover jump < 0.5m
# 8. testNoNetworkAccess - no network syscalls
```

---

## Performance Targets (Device: Snapdragon 7 Gen 1)

| Component | Target | Budget |
|-----------|--------|--------|
| Velocity inference | 4ms | 5ms |
| Pseudo-odo | 3ms | 5ms |
| Slip classify | 1ms | 2ms |
| Texture gate | 2ms | 3ms |
| Visual odometry | 8ms | 10ms |
| Forecaster | 1ms | 2ms |
| InEKF step | 0.5ms | 1ms |
| Adaptation step | 2ms | 3ms |
| **Total per frame** | **~20ms** | **< 15ms** |

---

## Battery & Thermal
- **Target**: < 500mW additional draw
- **VO throttling**: 5Hz when texture=poor
- **No GC in hot path**: Buffer pooling, object reuse
- **Thermal throttle**: Drop VO to 2Hz at 42°C

---

## Offline Verification Checklist
- [ ] No network permissions in manifest
- [ ] No `HttpURLConnection`/`OkHttp` in runtime path
- [ ] All models in `assets/`
- [ ] Map tiles in `assets/maplibre/tiles/`
- [ ] ProGuard rules block `java.net.*`, `okhttp3.*`
- [ ] `adb shell dumpsys package com.navx | grep permission` → no INTERNET

---

## Emergency Procedures
| Issue | Action |
|-------|--------|
| App crash | Check logcat, restart app |
| GNSS stuck | Toggle airplane mode |
| VO frozen | Toggle camera permission |
| Adaptation diverged | Auto-reset (kill switch) |
| Battery critical | Auto-disable VO/camera |

---

**End of Demo Script**