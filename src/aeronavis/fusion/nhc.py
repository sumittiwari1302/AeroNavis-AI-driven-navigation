"""NHC + gated pseudo-odometry measurement generator.

Contract for Part 7 (fusion):
    make_wheel_measurement(
        vel_forward, reliability, heading, slip_class, scale
    ) -> WheelMeasurement | None

WheelMeasurement = dataclass with body-frame velocity and covariance diagonal,
plus gate flag for filter down-weighting.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class WheelMeasurement:
    """Body-frame velocity measurement for InEKF/NHC."""

    ts: float  # timestamp
    vx_body: float  # forward velocity (m/s) in body frame
    vy_body: float  # lateral velocity (m/s) - NHC enforces ~0
    vz_body: float  # vertical velocity (m/s) - NHC enforces ~0
    R_diag: np.ndarray  # 3x3 diagonal covariance matrix
    gate: str  # "full" | "low" | "zupt"


def make_wheel_measurement(
    vel_forward: np.ndarray,  # (N,) learned forward velocity m/s
    reliability: np.ndarray,  # (N,) [0,1] per step
    heading: np.ndarray,  # (N,) heading rad
    slip_class: np.ndarray,  # (N,) 0=grip, 1=slip_brake, 2=stationary
    scale: Optional[float] = None,  # optional learned scale state
    ts: Optional[np.ndarray] = None,  # timestamps
    # Tunable thresholds
    rel_thresh_full: float = 0.7,
    slip_scale: float = 0.5,
    zupt_vel_thresh: float = 0.1,
) -> list[Optional[WheelMeasurement]]:
    """
    Generate gated wheel measurements for NHC fusion.

    Returns list of WheelMeasurement or None per time step.
    """
    N = len(vel_forward)
    if ts is None:
        ts = np.arange(N) * 0.01  # 100 Hz default

    measurements = []
    for i in range(N):
        v_fwd = float(vel_forward[i])
        rel = float(reliability[i])
        hdg = float(heading[i])
        slip = int(slip_class[i])

        # Default: no measurement
        meas = None

        if slip == 2:  # STATIONARY -> ZUPT
            if abs(v_fwd) < zupt_vel_thresh:
                meas = WheelMeasurement(
                    ts=float(ts[i]),
                    vx_body=0.0,
                    vy_body=0.0,
                    vz_body=0.0,
                    R_diag=np.array([1e-4, 1e-4, 1e-4], dtype=np.float64),
                    gate="zupt",
                )

        elif slip == 1:  # SLIP/BRAKE -> attenuated
            if rel > 0.3:  # minimum reliability
                v_scaled = v_fwd * (scale or slip_scale)
                c, s = np.cos(hdg), np.sin(hdg)
                meas = WheelMeasurement(
                    ts=float(ts[i]),
                    vx_body=v_scaled * c,
                    vy_body=v_scaled * s,
                    vz_body=0.0,
                    R_diag=np.array([0.5, 0.5, 1.0], dtype=np.float64),
                    gate="low",
                )

        else:  # GRIP (slip == 0)
            if rel > 0.7:
                v_scaled = v_fwd * (scale or 1.0)
                c, s = np.cos(hdg), np.sin(hdg)
                meas = WheelMeasurement(
                    ts=float(ts[i]),
                    vx_body=v_scaled * c,
                    vy_body=0.0,  # NHC lateral constraint
                    vz_body=0.0,  # NHC vertical constraint
                    R_diag=np.array([0.05, 0.01, 0.01], dtype=np.float64),
                    gate="full",
                )

        measurements.append(meas)

    return measurements


def batch_make_wheel_measurements(
    vel_forward: np.ndarray,
    reliability: np.ndarray,
    heading: np.ndarray,
    slip_class: np.ndarray,
    scale: Optional[float] = None,
    ts: Optional[np.ndarray] = None,
    **kwargs,
) -> list[Optional[WheelMeasurement]]:
    """Vectorized version for batch processing."""
    return make_wheel_measurement(
        vel_forward, reliability, heading, slip_class, scale, ts, **kwargs
    )


# --- Unit tests ---
def test_nhc_measurement():
    """Test NHC measurement contract."""
    N = 10
    vel = np.linspace(0, 10, N)
    rel = np.linspace(0, 1, N)
    hdg = np.zeros(N)
    slip = np.zeros(N, dtype=int)

    # Test GRIP with high reliability -> full NHC
    slip[:] = 0
    rel[:] = 0.9
    meas = make_wheel_measurement(vel, rel, hdg, slip)
    assert meas[5] is not None
    assert meas[5].gate == "full"
    assert meas[5].vy_body == 0.0
    assert meas[5].vz_body == 0.0
    assert meas[5].R_diag[1] < meas[5].R_diag[0]  # lateral tighter than forward

    # Test SLIP -> attenuated
    slip[:] = 1
    rel[:] = 0.5
    meas = make_wheel_measurement(vel, rel, hdg, slip)
    assert meas[5] is not None
    assert meas[5].gate == "low"
    assert meas[5].R_diag[0] > 0.1  # higher uncertainty

    # Test STATIONARY -> ZUPT
    slip[:] = 2
    vel[:] = 0.0
    rel[:] = 1.0
    meas = make_wheel_measurement(vel, rel, hdg, slip)
    assert meas[5] is not None
    assert meas[5].gate == "zupt"
    assert meas[5].vx_body == 0.0

    # Test low reliability GRIP -> no measurement
    slip[:] = 0
    rel[:] = 0.3
    meas = make_wheel_measurement(vel, rel, hdg, slip)
    assert meas[5] is None

    print("All NHC measurement tests passed!")


if __name__ == "__main__":
    test_nhc_measurement()
