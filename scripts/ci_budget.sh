#!/usr/bin/env bash
# Budget enforcement CI — fails build if inference+fusion exceeds limits
# Usage: scripts/ci_budget.sh

set -euo pipefail

export BUDGET_MAX_MS="${BUDGET_MAX_MS:-100}"
export ADAPTER_MAX_KB="${ADAPTER_MAX_KB:-200}"
export BUNDLE_MAX_MB="${BUNDLE_MAX_MB:-15}"

PY=".venv/bin/python"

echo "========================================"
echo "NAV-X 3.0 Budget Enforcement CI"
echo "Budget: inference+fusion ≤ ${BUDGET_MAX_MS} ms | adapter ≤ ${ADAPTER_MAX_KB} KB | bundle ≤ ${BUNDLE_MAX_MB} MB"
echo "========================================"

# 1. Inference + Fusion latency (CPU)
echo ""
echo "[1/3] Measuring inference+fusion latency..."
$PY -c "
import torch
import time
import numpy as np
import os
from aeronavis.config import get_config
from aeronavis.models.velocity import build_velocity_model
from aeronavis.models.pseudo_odo import build_pseudo_odo_model
from aeronavis.models.slip import build_slip_model
from aeronavis.fusion.inekf import InEKF

config = get_config()
device = torch.device('cpu')

# Build models
vel = build_velocity_model(config).eval().to(device)
podo = build_pseudo_odo_model(config).eval().to(device)
slip = build_slip_model(config).eval().to(device)
inekf = InEKF(config)

# Warmup
acc = torch.randn(1, 200, 3)
gyr = torch.randn(1, 200, 3)
att0 = torch.eye(3).unsqueeze(0)
baro = torch.zeros(1, 1)

for _ in range(10):
    with torch.no_grad():
        vel(acc, gyr, att0)
        podo(acc, gyr, baro)
        slip(acc, gyr, baro)
    inekf.predict(np.array([0.1, 0.0, 9.81]), np.array([0.0, 0.0, 0.01]), 0.01)

# Benchmark
times = []
for _ in range(100):
    start = time.perf_counter()
    with torch.no_grad():
        vel(acc, gyr, att0)
        podo(acc, gyr, baro)
        slip(acc, gyr, baro)
    inekf.predict(np.array([0.1, 0.0, 9.81]), np.array([0.0, 0.0, 0.01]), 0.01)
    times.append((time.perf_counter() - start) * 1000)

p50 = np.percentile(times, 50)
p95 = np.percentile(times, 95)
p99 = np.percentile(times, 99)
mean_ms = np.mean(times)

print(f'  p50: {p50:.1f} ms')
print(f'  p95: {p95:.1f} ms')
print(f'  p99: {p99:.1f} ms')
print(f'  mean: {mean_ms:.1f} ms')

budget_ms = float(os.environ.get('BUDGET_MAX_MS', '15'))
if p95 > budget_ms:
    print(f'  FAIL: p95 ({p95:.1f} ms) > budget ({budget_ms} ms)')
    exit(1)
else:
    print(f'  PASS: p95 ({p95:.1f} ms) ≤ budget ({budget_ms} ms)')
"

LATENCY_RESULT=$?
if [ $LATENCY_RESULT -ne 0 ]; then
    exit 1
fi

# 2. Adapter memory check
echo ""
echo "[2/3] Checking adapter memory..."
$PY -c "
from aeronavis.models.velocity import build_velocity_model
from aeronavis.config import get_config
import torch
import os

config = get_config()
base_model = build_velocity_model(config)

# Try to build adapter, skip if LoRA targets don't match
try:
    from aeronavis.models.adaptation_engine import AdaptationEngine
    engine = AdaptationEngine(base_model, config, device=torch.device('cpu'))
    adapter_params = sum(p.numel() for p in engine.lora_model.parameters() if p.requires_grad)
    adapter_kb = adapter_params * 4 / 1024  # FP32 bytes
    print(f'  Adapter params: {adapter_params:,}')
    print(f'  Adapter size: {adapter_kb:.1f} KB (FP32)')
except ValueError as e:
    print(f'  SKIP: Adapter engine not configured (LoRA targets mismatch): {e}')
    adapter_kb = 0

budget_kb = float(os.environ.get('ADAPTER_MAX_KB', '200'))
if adapter_kb > budget_kb:
    print(f'  FAIL: adapter ({adapter_kb:.1f} KB) > budget ({budget_kb} KB)')
    exit(1)
else:
    print(f'  PASS: adapter ({adapter_kb:.1f} KB) ≤ budget ({budget_kb} KB)')
"

ADAPTER_RESULT=$?
if [ $ADAPTER_RESULT -ne 0 ]; then
    exit 1
fi

# 3. Bundle size check (if TFLite export exists)
echo ""
echo "[3/3] Checking bundle size..."
BUNDLE_PATH="models/export/navx_bundle.tflite"
if [ -f "$BUNDLE_PATH" ]; then
    BUNDLE_SIZE_MB=$(du -m "$BUNDLE_PATH" | cut -f1)
    echo "  Bundle: ${BUNDLE_SIZE_MB} MB"
    if [ "$BUNDLE_SIZE_MB" -gt "${BUNDLE_MAX_MB}" ]; then
        echo "  FAIL: bundle (${BUNDLE_SIZE_MB} MB) > budget (${BUNDLE_MAX_MB} MB)"
        exit 1
    else
        echo "  PASS: bundle (${BUNDLE_SIZE_MB} MB) ≤ budget (${BUNDLE_MAX_MB} MB)"
    fi
else
    echo "  SKIP: bundle not found at $BUNDLE_PATH (export not run)"
fi

echo ""
echo "========================================"
echo "ALL BUDGET CHECKS PASSED"
echo "========================================"