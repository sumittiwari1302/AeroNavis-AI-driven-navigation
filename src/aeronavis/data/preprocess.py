"""Normalize every research corpus into a single internal NavSequence format.

Magical numbers live in config under ``data.*``; column layouts are asserted
per source and raise `LayoutError` (recording reality to PENDING.md) instead of
guessing when a corpus deviates from the documented layout.
"""

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import signal

from aeronavis.config import Config
from aeronavis.data.downloader import LayoutError

logger = logging.getLogger(__name__)

ACC_COLS = ("acc_x", "acc_y", "acc_z")
GYR_COLS = ("gyr_x", "gyr_y", "gyr_z")
MAG_COLS = ("mag_x", "mag_y", "mag_z")
TRUTH_COLS = ("x_m", "y_m", "z_m", "heading")


def _substr(full: str, *needles: str) -> str | None:
    text = full.strip().lower().replace(" ", "").replace("²", "2")
    for needle in needles:
        if needle in text:
            return full
    return None


@dataclass
class NavSequence:
    # Ordered per the Part 1 contract.
    seq_id: str
    source: str
    imu: pd.DataFrame
    mag: pd.DataFrame | None = None
    baro: pd.DataFrame | None = None
    gnss: pd.DataFrame | None = None
    wheel: pd.DataFrame | None = None
    truth: pd.DataFrame | None = None
    outages: list[tuple[float, float]] | None = None
    # Extension fields (kept after the contract fields).
    orientation_only: bool = False
    drive_id: str = ""
    region: str = ""
    vehicle: str = ""
    metadata: dict = field(default_factory=dict)

    # -- computed profile ----------------------------------------------------

    @property
    def start_ts(self) -> float:
        return float(self.imu["ts"].iloc[0])

    @property
    def end_ts(self) -> float:
        return float(self.imu["ts"].iloc[-1])

    @property
    def hours(self) -> float:
        return (self.end_ts - self.start_ts) / 3600.0

    @property
    def km(self) -> float:
        if self.truth is None:
            return 0.0
        d = np.diff(self.truth[["x_m", "y_m"]].to_numpy(), axis=0)
        return float(np.cumsum(np.hypot(d[:, 0], d[:, 1]))[-1]) / 1000.0 if len(d) else 0.0

    @property
    def outage_minutes(self) -> float:
        if not self.outages:
            return 0.0
        return sum(e - s for s, e in self.outages) / 60.0

    @property
    def gnss_coverage(self) -> float:
        return gnss_coverage(self.imu["ts"].to_numpy(), self.gnss)

    def passed_quality_gates(self, config: Config) -> bool:
        d = config.data
        if self.hours * 3600 < d.min_seq_s:
            return False
        if self.gnss is not None and not self.gnss.empty:
            return self.gnss_coverage >= d.min_gnss_coverage
        return True

    # -- cache io ------------------------------------------------------------

    def save(self, root: Path) -> Path:
        seq_dir = root / self.seq_id
        seq_dir.mkdir(parents=True, exist_ok=True)
        self.imu.to_parquet(seq_dir / "imu.parquet", index=False)
        for key, frame in (
            ("mag", self.mag),
            ("baro", self.baro),
            ("gnss", self.gnss),
            ("wheel", self.wheel),
            ("truth", self.truth),
        ):
            if frame is not None and not frame.empty:
                frame.to_parquet(seq_dir / f"{key}.parquet", index=False)
        meta = {
            "seq_id": self.seq_id,
            "source": self.source,
            "orientation_only": self.orientation_only,
            "drive_id": self.drive_id,
            "region": self.region,
            "vehicle": self.vehicle,
            "outages": [[start, end] for start, end in (self.outages or [])],
            "metadata": self.metadata,
        }
        (seq_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
        )
        return seq_dir

    @classmethod
    def load(cls, seq_dir: Path) -> "NavSequence":
        meta = json.loads((seq_dir / "meta.json").read_text(encoding="utf-8"))
        frames = {
            key: pd.read_parquet(seq_dir / f"{key}.parquet")
            for key in ("mag", "baro", "gnss", "wheel", "truth")
            if (seq_dir / f"{key}.parquet").exists()
        }
        return cls(
            seq_id=meta["seq_id"],
            source=meta["source"],
            imu=pd.read_parquet(seq_dir / "imu.parquet"),
            mag=frames.get("mag"),
            baro=frames.get("baro"),
            gnss=frames.get("gnss"),
            wheel=frames.get("wheel"),
            truth=frames.get("truth"),
            outages=[tuple(pair) for pair in meta.get("outages", [])] or None,
            orientation_only=meta.get("orientation_only", False),
            drive_id=meta.get("drive_id", ""),
            region=meta.get("region", ""),
            vehicle=meta.get("vehicle", ""),
            metadata=meta.get("metadata", {}),
        )

    @classmethod
    def load_all(cls, root: Path) -> list["NavSequence"]:
        seqs = []
        for seq_dir in sorted(root.iterdir()):
            if seq_dir.is_dir() and (seq_dir / "meta.json").exists():
                try:
                    seqs.append(cls.load(seq_dir))
                except Exception as exc:  # corrupted cache entry
                    logger.warning(f"skipping unreadable cache {seq_dir}: {exc}")
        return seqs


# ---------------------------------------------------------------------------
# shared signal plumbing
# ---------------------------------------------------------------------------


def _longest_gap_free_span(
    df: pd.DataFrame, ts_col: str, max_gap_s: float
) -> tuple[pd.DataFrame, int]:
    ts = df[ts_col].to_numpy(float)
    if len(ts) < 2:
        return df, 0
    gaps = np.diff(ts) > max_gap_s
    if not gaps.any():
        return df, 0
    boundaries = np.flatnonzero(gaps) + 1
    cuts = np.split(np.arange(len(df)), boundaries)
    longest = max(cuts, key=len)
    kept = df.iloc[longest]
    dropped = len(df) - len(longest)
    logger.info(
        f"dropped {dropped} samples across {len(cuts) - 1} gap(s) > {max_gap_s * 1000:.0f} ms"
    )
    return kept.reset_index(drop=True), dropped


def _resample_poly_df(
    df: pd.DataFrame, ts_col: str, target_hz: float
) -> tuple[pd.DataFrame, float]:
    ts = df[ts_col].to_numpy(float)
    if len(ts) < 2:
        raise ValueError("need at least two samples to resample")
    t0, t1 = float(ts.min()), float(ts.max())
    n = int(round((t1 - t0) * target_hz)) + 1
    n = max(2, n)
    value_cols = [c for c in df.columns if c != ts_col]
    out: dict[str, np.ndarray] = {}
    for col in value_cols:
        x = df[col].to_numpy(float)
        factor = math.gcd(n, len(x))
        y = signal.resample_poly(x, up=n // factor, down=len(x) // factor)
        y = np.resize(y, n)
        out[col] = y
    out["ts"] = t0 + np.arange(n) / target_hz
    native = 1.0 / float(np.median(np.diff(ts))) if len(ts) > 1 else float("nan")
    return pd.DataFrame(out), native


def normalize_imu(
    raw: pd.DataFrame, ts_col: str, imu_hz: float, max_gap_s: float
) -> tuple[pd.DataFrame, float, int]:
    """Drop long gaps (keep longest span), then exactly resample to ``imu_hz``.

    Returns (imu, native_hz, dropped_samples). Skipped NaN payload rows are
    removed before gap splitting, so kept segments never contain NaN.
    """
    clean = raw.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)
    value_cols = [c for c in clean.columns if c != ts_col]
    if value_cols:
        block = clean[value_cols].to_numpy(dtype=float)
        finite = np.isfinite(block).all(axis=1)
        clean = clean.loc[finite].reset_index(drop=True)
    span, dropped = _longest_gap_free_span(clean, ts_col, max_gap_s)
    out, native = _resample_poly_df(span, ts_col, imu_hz)
    return out, native, dropped


def gnss_coverage(imu_ts: np.ndarray, gnss: pd.DataFrame | None) -> float:
    if gnss is None or gnss.empty:
        return 0.0
    t0, t1 = float(imu_ts.min()), float(imu_ts.max())
    if t1 <= t0:
        return 0.0
    bin_s = 1.0
    gnss_bin = np.floor(gnss["ts"].to_numpy(float) / bin_s).astype(int)
    occupied = np.unique(gnss_bin[(gnss["ts"] >= t0) & (gnss["ts"] <= t1)])
    total = int(np.floor((t1 - t0) / bin_s)) + 1
    return float(len(occupied)) / float(total) if total else 0.0


def derive_outages(
    gnss_ts: np.ndarray, t0: float, t1: float, min_gap_s: float
) -> list[tuple[float, float]]:
    if len(gnss_ts) < 2:
        return []
    ts = np.sort(gnss_ts[(gnss_ts >= t0) & (gnss_ts <= t1)])
    if len(ts) < 2:
        return []
    pairs = np.diff(ts) > min_gap_s
    starts = np.flatnonzero(pairs)
    intervals = []
    for i in starts:
        s, e = float(ts[i]), float(ts[i + 1])
        intervals.append((max(s, t0), min(e, t1)))
    return intervals


def enu_truth_from_gnss(gnss: pd.DataFrame, truth_hz: float, earth_radius_m: float) -> pd.DataFrame:
    """GNSS fixes -> local ENU trajectory, resampled to ``truth_hz``."""
    fix = gnss.dropna(subset=["lat", "lon", "alt_m"]).sort_values("ts")
    if len(fix) < 2:
        return pd.DataFrame(columns=["ts", *TRUTH_COLS])
    lat0, lon0 = float(fix["lat"].iloc[0]), float(fix["lon"].iloc[0])
    lat, lon, alt = (
        fix["lat"].to_numpy(float),
        fix["lon"].to_numpy(float),
        fix["alt_m"].to_numpy(float),
    )
    cos0 = math.cos(math.radians(lat0))
    x = (lon - lon0) * earth_radius_m * cos0 * math.pi / 180.0
    y = (lat - lat0) * earth_radius_m * math.pi / 180.0
    z = alt - float(fix["alt_m"].iloc[0])
    pos = pd.DataFrame({"ts": fix["ts"].to_numpy(float), "x_m": x, "y_m": y, "z_m": z})
    resampled, _ = _resample_poly_df(pos, "ts", truth_hz)
    heading = np.unwrap(np.arctan2(np.diff(resampled["y_m"]), np.diff(resampled["x_m"])))
    heading = np.concatenate([heading[:1], heading])
    resampled["heading"] = heading
    return resampled


def enu_truth_from_relative_pose(pose: pd.DataFrame, hz: float) -> pd.DataFrame:
    resampled, _ = _resample_poly_df(pose, "ts", hz)
    resampled["heading"] = 0.0
    return resampled


# ---------------------------------------------------------------------------
# column layout helpers (asserted per source)
# ---------------------------------------------------------------------------


def find_columns(candidates: list[str], needles: str | tuple[str, ...]) -> str | None:
    for col in candidates:
        if isinstance(needles, tuple):
            if _substr(col, *needles):
                return col
        elif _substr(col, needles):
            return col
    return None


def required_column(candidates: list[str], needles: tuple[str, ...], file: Path) -> str:
    col = find_columns(candidates, needles)
    if col is None:
        raise LayoutError(
            f"{file}: expected a column matching {needles} in {candidates!r}; "
            "record the real layout in PENDING.md, do not guess."
        )
    return col


# ---------------------------------------------------------------------------
# io_vnbd
# ---------------------------------------------------------------------------

S_NEEDLES = "accelerometerx"
V_NEEDLES = "wheelspeed"


def _classify_io_vnbd(file: Path) -> str:
    df = pd.read_csv(file, nrows=0, encoding="latin-1")
    cols = list(df.columns)
    if find_columns(cols, "accelerometerx"):
        return "S"
    if find_columns(cols, "wheelspeed"):
        return "V"
    raise LayoutError(
        f"{file}: header {cols!r} matches neither phone (S) nor vehicle (V) "
        "layout; record it in PENDING.md."
    )


def _route_key(stem: str) -> str:
    key = stem.split("-", 1)[1] if "-" in stem else stem
    if len(key) > 1 and key[-1] in "abcdefgh":
        return key[:-1]
    return key


def _driver_of(route: str) -> str:
    for prefix in ("vta", "vtb", "vw", "vf", "st"):
        if route.lower().startswith(prefix):
            return prefix.upper()
    return route[0].upper()


def _region_of(lat: float, lon: float) -> str:
    if 4.0 <= lat <= 14.0 and 3.0 <= lon <= 15.0:
        return "ng"
    if 42.0 <= lat <= 51.0:
        return "fr" if -4.0 <= lon <= 8.0 else "uk"
    return "uk"


def _find_gyro_columns(cols: list[str], file: Path) -> tuple[str, str, str]:
    # Try X/Y/Z first (body-frame rates)
    gx = find_columns(cols, ("gyroscopex",))
    gy = find_columns(cols, ("gyroscopey",))
    gz = find_columns(cols, ("gyroscopez",))
    if gx and gy and gz:
        return gx, gy, gz
    # Fallback: Euler-rate style Yaw/Pitch/Roll
    gyaw = find_columns(cols, ("gyroscopeyaw",))
    gpitch = find_columns(cols, ("gyroscopepitch",))
    groll = find_columns(cols, ("gyroscoperoll",))
    if gyaw and gpitch and groll:
        # Map pitch→x, roll→y, yaw→z (aerospace convention)
        return gpitch, groll, gyaw
    raise LayoutError(
        f"{file}: no gyroscope columns found in {cols!r}; expected X/Y/Z or Yaw/Pitch/Roll"
    )


def parse_io_vnbd_phone(file: Path) -> pd.DataFrame:
    df = pd.read_csv(file, encoding="latin-1")
    cols = list(df.columns)
    t_ms = required_column(cols, ("timesincestart",), file)
    lat = required_column(cols, ("gpslatitude",), file)
    lon = required_column(cols, ("gpslongitude",), file)
    alt = required_column(cols, ("gpsaltitude",), file)
    speed = required_column(cols, ("gpsspeed",), file)
    heading = required_column(cols, ("gpsorientation",), file)
    acc_x = required_column(cols, ("accelerometerx",), file)
    acc_y = required_column(cols, ("accelerometery",), file)
    acc_z = required_column(cols, ("accelerometerz",), file)
    gyr_x, gyr_y, gyr_z = _find_gyro_columns(cols, file)
    mag_x = find_columns(cols, ("magneticfieldx", "magnetometerx"))
    mag_y = find_columns(cols, ("magneticfieldy", "magnetometery"))
    mag_z = find_columns(cols, ("magneticfieldz", "magnetometerz"))

    ts = pd.to_numeric(df[t_ms], errors="coerce").to_numpy(float) / 1000.0
    ts = ts - float(np.nanmin(ts))
    imu = pd.DataFrame(
        {
            "ts": ts,
            "acc_x": pd.to_numeric(df[acc_x], errors="coerce"),
            "acc_y": pd.to_numeric(df[acc_y], errors="coerce"),
            "acc_z": pd.to_numeric(df[acc_z], errors="coerce"),
            "gyr_x": pd.to_numeric(df[gyr_x], errors="coerce"),
            "gyr_y": pd.to_numeric(df[gyr_y], errors="coerce"),
            "gyr_z": pd.to_numeric(df[gyr_z], errors="coerce"),
        }
    )
    gnss = pd.DataFrame(
        {
            "ts": ts,
            "lat": pd.to_numeric(df[lat], errors="coerce"),
            "lon": pd.to_numeric(df[lon], errors="coerce"),
            "alt_m": pd.to_numeric(df[alt], errors="coerce"),
            "speed_mps": pd.to_numeric(df[speed], errors="coerce") / 3.6,
            "heading_deg": pd.to_numeric(df[heading], errors="coerce"),
        }
    )
    if mag_x is not None:
        mag = pd.DataFrame(
            {
                "ts": ts,
                "mag_x": pd.to_numeric(df[mag_x], errors="coerce"),
                "mag_y": pd.to_numeric(df[mag_y], errors="coerce"),
                "mag_z": pd.to_numeric(df[mag_z], errors="coerce"),
            }
        )
    else:
        mag = None
    return imu, gnss, mag


def parse_io_vnbd_vehicle(file: Path) -> pd.DataFrame:
    df = pd.read_csv(file, encoding="latin-1")
    cols = list(df.columns)
    t_s = required_column(cols, ("timesincestartofday",), file)
    indicated = required_column(cols, ("indicatedvehiclespeed",), file)
    wheel = [
        find_columns(cols, f"wheelspeed{w}")
        for w in ("frontleft", "frontright", "rearleft", "rearright")
    ]
    wheels = [c for c in wheel if c]
    ts = pd.to_numeric(df[t_s], errors="coerce").to_numpy(float)
    ts = ts - float(np.nanmin(ts))
    out = pd.DataFrame(
        {"ts": ts, "wheel_speed_mps": pd.to_numeric(df[indicated], errors="coerce") / 3.6}
    )
    out["wheel_ts_count"] = int(len(wheels))
    return out


def preprocess_io_vnbd(raw: Path, config: Config) -> list[NavSequence]:
    csvs = sorted(raw.glob("src/**/*.csv"), key=lambda p: p.as_posix())
    vehicles: dict[str, list[pd.DataFrame]] = {}
    phone_runs: dict[str, list[tuple[Path, pd.DataFrame]]] = {}
    for file in csvs:
        kind = _classify_io_vnbd(file)
        stem = file.stem.upper()
        key = _route_key(stem)
        if kind == "V":
            try:
                vehicles.setdefault(key, []).append(parse_io_vnbd_vehicle(file))
            except Exception as exc:
                logger.warning(f"io_vnbd: skipping vehicle file {file}: {exc}")
        else:
            try:
                imu, gnss, mag = parse_io_vnbd_phone(file)
                phone_runs.setdefault(key, []).append((file, imu, gnss, mag))
            except Exception as exc:
                logger.warning(f"io_vnbd: skipping phone file {file}: {exc}")

    seqs: list[NavSequence] = []
    stem_count: dict[str, int] = {}
    for key, runs in sorted(phone_runs.items()):
        for file, raw_imu, raw_gnss, raw_mag in runs:
            stem_count[file.stem] = stem_count.get(file.stem, 0) + 1
            try:
                seq = _build_io_vnbd_sequence(
                    key, file, raw_imu, raw_gnss, raw_mag, vehicles, stem_count[file.stem], config
                )
                if seq is not None:
                    seqs.append(seq)
            except Exception as exc:
                logger.warning(f"io_vnbd: sequence for {file} failed: {exc}")
    return seqs


def _build_io_vnbd_sequence(
    key, file, raw_imu, raw_gnss, raw_mag, vehicles, occurrence, config
) -> NavSequence | None:
    d = config.data
    imu, native, dropped = normalize_imu(raw_imu, "ts", d.imu_hz, d.max_gap_ms / 1000.0)
    mag = None
    if raw_mag is not None and len(raw_mag):
        mag, _, _ = normalize_imu(raw_mag, "ts", d.mag_hz, d.max_gap_ms / 1000.0)
    gnss = raw_gnss.copy()
    gnss = gnss[np.isfinite(gnss["lat"]) & np.isfinite(gnss["lon"])].reset_index(drop=True)

    truth = enu_truth_from_gnss(gnss, d.truth_hz, d.earth_radius_m)
    t0, t1 = float(imu["ts"].min()), float(imu["ts"].max())
    outages = derive_outages(gnss["ts"].to_numpy(float), t0, t1, d.outage_min_gap_s)

    wheel = None
    if key in vehicles:
        merged = pd.concat([v for v in vehicles[key]], ignore_index=True)
        wheel = merged.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
        wheel = wheel[wheel["ts"] >= t0 - 1.0]

    lat_med = float(gnss["lat"].median()) if len(gnss) else float("nan")
    lon_med = float(gnss["lon"].median()) if len(gnss) else float("nan")
    region = _region_of(lat_med, lon_med) if math.isfinite(lat_med) else "unknown"
    vehicle = _driver_of(key)

    seq = NavSequence(
        seq_id=f"io_vnbd_{file.stem}_{occurrence}",
        source="io_vnbd",
        imu=imu,
        mag=mag,
        gnss=gnss,
        wheel=wheel,
        truth=truth,
        outages=outages or None,
        orientation_only=False,
        drive_id=f"io_vnbd:{vehicle}:{key}",
        region=region,
        vehicle=vehicle,
        metadata={"native_imu_hz": native, "dropped_samples": dropped, "file": file.name},
    )
    if not seq.passed_quality_gates(config):
        logger.info(f"io_vnbd: {seq.seq_id} filtered by quality gates")
        return None
    return seq


# ---------------------------------------------------------------------------
# ronin (HDF5, documented at ronin.cs.sfu.ca/README.txt)
# ---------------------------------------------------------------------------


def preprocess_ronin(raw: Path, config: Config) -> list[NavSequence]:
    seqs: list[NavSequence] = []
    for h5 in sorted(raw.glob("src/**/data.hdf5")):
        try:
            seq = _build_ronin_sequence(h5, config)
            if seq is not None:
                seqs.append(seq)
        except Exception as exc:
            logger.warning(f"ronin: {h5} failed: {exc}")
    return seqs


def _build_ronin_sequence(h5_path: Path, config: Config) -> NavSequence | None:
    d = config.data
    with h5py.File(h5_path, "r") as h5:
        if "synced" not in h5:
            raise LayoutError(
                f"{h5_path}: expected 'synced' group per ronin README.txt; actual groups {list(h5)}"
            )
        t = np.asarray(h5["synced/time"]).astype(float)
        acc = np.asarray(h5["synced/acce"]).astype(float)
        gyr = np.asarray(h5["synced/gyro"]).astype(float)
        has_mag = "magnet" in h5["synced"]
        mag = np.asarray(h5["synced/magnet"]).astype(float) if has_mag else None
        if "pose" in h5 and "tango_pos" in h5["pose"]:
            pos = np.asarray(h5["pose/tango_pos"]).astype(float)
            ori = np.asarray(h5["pose/tango_ori"]).astype(float)
        else:
            pos = ori = None

    t = t - t[0]
    imu_raw = pd.DataFrame(
        {
            "ts": t,
            "acc_x": acc[:, 0],
            "acc_y": acc[:, 1],
            "acc_z": acc[:, 2],
            "gyr_x": gyr[:, 0],
            "gyr_y": gyr[:, 1],
            "gyr_z": gyr[:, 2],
        }
    )
    mag_raw = None
    if mag is not None and mag.ndim == 2 and mag.shape[1] >= 3:
        mag_raw = pd.DataFrame(
            {"ts": t, "mag_x": mag[:, 0], "mag_y": mag[:, 1], "mag_z": mag[:, 2]}
        )
    mag_raw = mag_raw if mag_raw is not None and len(mag_raw) == len(t) else None

    imu, native, dropped = normalize_imu(imu_raw, "ts", d.imu_hz, d.max_gap_ms / 1000.0)

    truth = None
    if pos is not None and ori is not None and len(pos) > 1:
        pose_ts = np.linspace(t[0], t[-1], len(pos))
        pose = pd.DataFrame({"ts": pose_ts, "x_m": pos[:, 0], "y_m": pos[:, 1], "z_m": pos[:, 2]})
        truth = enu_truth_from_relative_pose(pose, d.truth_hz)
        qw, qx, qy, qz = (
            ori[:, 0].astype(float),
            ori[:, 1].astype(float),
            ori[:, 2].astype(float),
            ori[:, 3].astype(float),
        )
        heading = np.unwrap(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy**2 + qz**2)))
        truth["heading"] = np.interp(truth["ts"].to_numpy(), pose_ts, heading)

    seq_dir = h5_path.parent
    meta = {}
    info = seq_dir / "info.json"
    if info.exists():
        meta = json.loads(info.read_text(encoding="utf-8"))
    drive = meta.get("device", seq_dir.name)
    seq = NavSequence(
        seq_id=f"ronin_{seq_dir.name}",
        source="ronin",
        imu=imu,
        mag=mag_raw,
        baro=None,
        gnss=None,
        wheel=None,
        truth=truth,
        outages=None,
        orientation_only=False,
        drive_id=f"ronin:{drive}",
        region="indoor",
        vehicle=drive,
        metadata={"native_imu_hz": native, "dropped_samples": dropped, "info": meta},
    )
    if not seq.passed_quality_gates(config):
        return None
    return seq


# ---------------------------------------------------------------------------
# idol (feather; documented in the zenodo record 4484093)
# ---------------------------------------------------------------------------


def preprocess_idol(raw: Path, config: Config) -> list[NavSequence]:
    seqs: list[NavSequence] = []
    for feather in sorted(raw.glob("src/**/*.feather")):
        try:
            seq = _build_idol_sequence(feather, config)
            if seq is not None:
                seqs.append(seq)
        except Exception as exc:
            logger.warning(f"idol: {feather} failed: {exc}")
    return seqs


def _q_to_yaw(qw: float, qx: float, qy: float, qz: float) -> float:
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy**2 + qz**2))


def _idol_truth_from_pose(pose: pd.DataFrame, truth_hz: float) -> pd.DataFrame:
    truth = enu_truth_from_relative_pose(pose[["ts", "x_m", "y_m", "z_m"]], truth_hz)
    yaw = np.unwrap([_q_to_yaw(*row) for row in pose[["qw", "qx", "qy", "qz"]].to_numpy(float)])
    truth["heading"] = np.interp(truth["ts"].to_numpy(), pose["ts"].to_numpy(), yaw)
    return truth


def _build_idol_sequence(feather: Path, config: Config) -> NavSequence | None:
    d = config.data
    df = pd.read_feather(feather).sort_values("timestamp")
    t = df["timestamp"].to_numpy(float)
    t = t - t[0]
    acc = pd.DataFrame(
        {
            "ts": t,
            "acc_x": df["iphoneAccX"].to_numpy(float),
            "acc_y": df["iphoneAccY"].to_numpy(float),
            "acc_z": df["iphoneAccZ"].to_numpy(float),
            "gyr_x": df["iphoneGyroX"].to_numpy(float),
            "gyr_y": df["iphoneGyroY"].to_numpy(float),
            "gyr_z": df["iphoneGyroZ"].to_numpy(float),
            "mag_x": df["iphoneMagX"].to_numpy(float),
            "mag_y": df["iphoneMagY"].to_numpy(float),
            "mag_z": df["iphoneMagZ"].to_numpy(float),
        }
    )
    imu, native, dropped = normalize_imu(acc, "ts", d.imu_hz, d.max_gap_ms / 1000.0)

    pose = pd.DataFrame(
        {
            "ts": t,
            "x_m": df["processedPosX"].to_numpy(float),
            "y_m": df["processedPosY"].to_numpy(float),
            "z_m": df["processedPosZ"].to_numpy(float),
            "qw": df["orientW"].to_numpy(float),
            "qx": df["orientX"].to_numpy(float),
            "qy": df["orientY"].to_numpy(float),
            "qz": df["orientZ"].to_numpy(float),
        }
    )
    truth = _idol_truth_from_pose(pose, d.truth_hz)

    building = feather.parts[-2]
    seq = NavSequence(
        seq_id=f"idol_{feather.stem}",
        source="idol",
        imu=imu,
        mag=imu[["ts", "mag_x", "mag_y", "mag_z"]],
        baro=None,
        gnss=None,
        wheel=None,
        truth=truth,
        outages=None,
        orientation_only=True,
        drive_id=f"idol:{building}:{feather.stem}",
        region=str(building),
        vehicle=feather.stem,
        metadata={"native_imu_hz": native, "dropped_samples": dropped},
    )
    if not seq.passed_quality_gates(config):
        return None
    return seq


# ---------------------------------------------------------------------------
# self_collected (CSV runs per RECORDING_GUIDE.md)
# ---------------------------------------------------------------------------


def preprocess_self_collected(raw: Path, config: Config) -> list[NavSequence]:
    seqs: list[NavSequence] = []
    for run in sorted([p for p in raw.iterdir() if p.is_dir()]):
        try:
            seq = _build_self_collected_sequence(run, config)
            if seq is not None:
                seqs.append(seq)
        except Exception as exc:
            logger.warning(f"self_collected: run {run} failed: {exc}")
    return seqs


def _read_sensor_csv(run: Path, name: str) -> pd.DataFrame | None:
    path = run / name
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["ts"] = df["ts_ms"].to_numpy(float) / 1000.0
    df["ts"] = df["ts"] - df["ts"].min()
    return df.drop(columns=["ts_ms"])


def _build_self_collected_sequence(run: Path, config: Config) -> NavSequence | None:
    d = config.data
    meta = {}
    meta_path = run / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    acc = _read_sensor_csv(run, "accel.csv")
    gyr = _read_sensor_csv(run, "gyro.csv")
    if acc is None or gyr is None:
        raise LayoutError(f"{run}: requires accel.csv and gyro.csv")

    imu_raw = acc.merge(gyr, on="ts", how="inner")
    imu, native, dropped = normalize_imu(imu_raw, "ts", d.imu_hz, d.max_gap_ms / 1000.0)

    mag = _read_sensor_csv(run, "mag.csv")
    baro = _read_sensor_csv(run, "pressure.csv")
    gnss_raw = _read_sensor_csv(run, "gnss.csv")
    gnss = None
    truth = None
    outages = None
    if gnss_raw is not None and len(gnss_raw):
        gnss = gnss_raw[
            ["ts", "lat", "lon", "alt_m", "speed_mps", "heading_deg", "pdop", "cno_db_hz"]
        ].dropna(subset=["lat", "lon"])
        truth = enu_truth_from_gnss(gnss, d.truth_hz, d.earth_radius_m)
        t0, t1 = float(imu["ts"].min()), float(imu["ts"].max())
        outages = derive_outages(gnss["ts"].to_numpy(float), t0, t1, d.outage_min_gap_s)

    seq = NavSequence(
        seq_id=f"self_{run.name}",
        source="self_collected",
        imu=imu,
        mag=mag,
        baro=baro,
        gnss=gnss,
        wheel=None,
        truth=truth,
        outages=outages or None,
        orientation_only=False,
        drive_id=f"self_collected:{meta.get('vehicle', run.name)}",
        region="in",
        vehicle=meta.get("vehicle", run.name),
        metadata={"native_imu_hz": native, "dropped_samples": dropped, **meta},
    )
    if not seq.passed_quality_gates(config):
        return None
    return seq


PROCESSORS = {
    "io_vnbd": preprocess_io_vnbd,
    "ronin": preprocess_ronin,
    "idol": preprocess_idol,
    "self_collected": preprocess_self_collected,
}


def preprocess_source(source: str, config: Config) -> list[NavSequence]:
    raw = Path(config.paths.raw) / source
    return PROCESSORS[source](raw, config)
