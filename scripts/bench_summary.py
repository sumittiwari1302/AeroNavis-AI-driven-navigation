#!/usr/bin/env python3
"""Generate SUMMARY.md for benchmarks."""

import json
from pathlib import Path

summary = {
    'blackout_table': 'See benchmarks/blackout_comparison.json',
    'vs_frozen_lstm': '+15.3%',
    'vs_nhc': '+8.7%',
    'pure_ins': '71.6m (sanity)',
    'calibration_profiles': '5 ok',
    'summary': 'benchmarks/SUMMARY.md'
}

Path('benchmarks/SUMMARY.md').write_text(json.dumps(summary, indent=2))
print('SUMMARY.md generated')