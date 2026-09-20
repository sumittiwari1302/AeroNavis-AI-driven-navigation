"""Source handlers: offline-aware downloaders for every research dataset.

Each handler owns ``data/raw/<name>/`` and knows how to fetch, extract, and
verify its canonical layout. Handlers never guess: `download()` verifies the
extracted structure against an asserted layout and raises `LayoutError` (with
the observed reality recorded to PENDING.md) instead of silently proceeding.
"""

import json
import logging
import shutil
import urllib.request
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path

from aeronavis.config import Config

logger = logging.getLogger(__name__)

_MARKER = ".DOWNLOADED.json"


class DownloadError(RuntimeError):
    """Raised when a source cannot be fetched; message carries human steps."""


class LayoutError(RuntimeError):
    """Raised when an extracted dataset does not match the asserted layout."""


class SourceHandler(ABC):
    name: str

    def __init__(self, config: Config, root: Path | None = None) -> None:
        self.config = config
        raw_root = Path(config.paths.raw)
        if root is not None:
            raw_root = root
        self.root = raw_root / self.name

    # -- interface -----------------------------------------------------------

    @abstractmethod
    def urls(self) -> dict[str, str]:
        """Map of zip archive name -> downloadable URL."""

    @abstractmethod
    def expected_layout(self) -> list[str]:
        """Glob patterns a successful extraction must satisfy (relative)."""

    @abstractmethod
    def verify(self) -> None:
        """Raise LayoutError if the extracted files contradict the layout."""

    # -- concrete behavior ---------------------------------------------------

    def available(self) -> bool:
        marker = self.root / _MARKER
        if not marker.exists():
            return False
        return any(self.root.glob(p) for p in self.expected_layout())

    def download(self) -> Path:
        if self.available():
            logger.info(f"{self.name}: already downloaded, using cache")
            return self.root
        self.root.mkdir(parents=True, exist_ok=True)
        src_dir = self.root / "src"
        src_dir.mkdir(exist_ok=True)

        for name, url in self.urls().items():
            dest = self.root / "src" / name
            if dest.exists() and dest.stat().st_size > 0:
                logger.info(f"{self.name}: {name} already on disk, skipping fetch")
                continue
            self._fetch(url, dest)
            self._extract(dest, src_dir)

        try:
            self.verify()
        except LayoutError as exc:
            self._record_layout(str(exc))
            raise
        marker = self.root / _MARKER
        marker.write_text(json.dumps({"name": self.name, "urls": self.urls()}), encoding="utf-8")
        logger.info(f"{self.name}: download complete at {self.root}")
        return self.root

    def _fetch(self, url: str, dest: Path) -> None:
        logger.info(f"{self.name}: downloading {url}")
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as out:  # noqa: SIM115
                shutil.copyfileobj(resp, out, length=1 << 20)
        except Exception as exc:  # URLError / HTTPError / timeout
            raise DownloadError(
                f"{self.name}: failed to fetch {url} ({exc}).\n"
                f"Place the archive manually at {dest} and re-run."
            ) from exc
        tmp.replace(dest)
        logger.info(f"{self.name}: fetched {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")

    @staticmethod
    def _extract(archive: Path, dest: Path) -> None:
        if archive.suffix.lower() == ".zip" or zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(dest)
            return
        raise DownloadError(f"unsupported archive format: {archive.name}")

    def describe(self) -> dict:
        stats_file = self.root / "_stats.json"
        if not stats_file.exists():
            return {
                "ready": self.available(),
                "sequences": 0,
                "hours": 0.0,
                "km": 0.0,
                "outage_minutes": 0.0,
                "vehicles": [],
                "regions": [],
            }
        with stats_file.open(encoding="utf-8") as fh:
            stats = json.load(fh)
        stats["ready"] = self.available()
        return stats

    def _record_layout(self, message: str) -> None:
        log_path = self.root / "PENDING.md"
        entry = f"\n## Layout mismatch recorded by downloader\n\n{message}\n"
        if log_path.exists():
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(entry)
        else:
            log_path.write_text(
                "# PENDING — unverified source layout\n\n"
                "The extractor observed a real layout that does not match this "
                "repo's assumptions. Fix `preprocess` against it, never the "
                "reverse.\n",
                encoding="utf-8",
            )
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(entry)
        logger.error(f"{self.name}: layout mismatch recorded at {log_path}")


class IOVNBDHandler(SourceHandler):
    """Vehicle + phone wheel-speed/IMU benchmark (UK/NG/FR trucks)."""

    name = "io_vnbd"

    ZIPS = {
        "Synchronised.zip": (
            "https://media.githubusercontent.com/media/onyekpeu/IO-VNBD/master/"
            "Synchronised%20V%20abd%20S%20datasets.zip"
        ),
    }

    def urls(self) -> dict[str, str]:
        return dict(self.ZIPS)

    def expected_layout(self) -> list[str]:
        return ["src/**/*.csv"]

    def verify(self) -> None:
        csvs = list(self.root.glob("src/**/*.csv"))
        if not csvs:
            raise LayoutError(
                f"{self.name}: no CSV payload found under {self.root}/src. "
                "The canonical file tree differs from the asserted one."
            )
        logger.info(f"{self.name}: verified {len(csvs)} CSV files")


class RoninHandler(SourceHandler):
    """RoNIN (ICRA 2020): 42.7 h pedestrian phone IMU + Tango trajectories."""

    name = "ronin"

    # Federated Research Data Repository record DOI 10.20383/102.0543.
    # Each archive is a tar/zip of `sequence_name/{data.hdf5, info.json}` dirs.
    ZIPS = {
        "train_1.zip": "https://doi.org/10.20383/102.0543",
        "train_2.zip": "https://doi.org/10.20383/102.0543",
        "seen_test.zip": "https://doi.org/10.20383/102.0543",
        "unseen_test.zip": "https://doi.org/10.20383/102.0543",
    }

    def urls(self) -> dict[str, str]:
        return dict(self.ZIPS)

    def expected_layout(self) -> list[str]:
        return ["src/**/data.hdf5"]

    def verify(self) -> None:
        h5s = list(self.root.glob("src/**/data.hdf5"))
        if not h5s:
            raise LayoutError(
                f"{self.name}: no data.hdf5 found under {self.root}/src. "
                "Expected sequence dirs each holding data.hdf5 + info.json."
            )
        logger.info(f"{self.name}: verified {len(h5s)} HDF5 sequences")


class IDOLHandler(SourceHandler):
    """IDOL (AAAI 2021): indoor pedestrian orientation+IMU, .feather files."""

    name = "idol"

    ZIPS = {
        "building1.zip": ("https://zenodo.org/api/records/4484093/files/building1.zip/content"),
        "building2.zip": ("https://zenodo.org/api/records/4484093/files/building2.zip/content"),
        "building3.zip": ("https://zenodo.org/api/records/4484093/files/building3.zip/content"),
    }

    def urls(self) -> dict[str, str]:
        return dict(self.ZIPS)

    def expected_layout(self) -> list[str]:
        return ["src/**/*.feather"]

    def verify(self) -> None:
        feathers = list(self.root.glob("src/**/*.feather"))
        if not feathers:
            raise LayoutError(
                f"{self.name}: no .feather files found under {self.root}/src. "
                "IDOL stores trajectories as Apache Arrow feather files."
            )
        logger.info(f"{self.name}: verified {len(feathers)} feather trajectories")


class SelfCollectedHandler(SourceHandler):
    """No network. Writes the recording protocol for Indian-roads demos."""

    name = "self_collected"

    def urls(self) -> dict[str, str]:
        return {}

    def expected_layout(self) -> list[str]:
        return [f"{self.name}_guide.md", "*.csv"]

    def download(self) -> Path:
        if self.available():
            logger.info("self_collected: guide already present, using cache")
            return self.root
        self.root.mkdir(parents=True, exist_ok=True)
        self._write_guide()
        self._write_schema_csv()
        (self.root / _MARKER).write_text(
            json.dumps({"name": self.name, "urls": {}}), encoding="utf-8"
        )
        logger.info(f"self_collected: recording protocol written to {self.root}")
        return self.root

    def verify(self) -> None:
        if not (self.root / f"{self.name}_guide.md").exists():
            raise LayoutError("self_collected guide missing after write")

    # -- protocol artifacts --------------------------------------------------

    def _write_guide(self) -> None:
        guide = self.root / f"{self.name}_guide.md"
        guide.write_text(
            self._guide_text(),
            encoding="utf-8",
        )

    @staticmethod
    def _guide_text() -> str:
        return (
            "# NAV-X Self-Collection Protocol (Indian-roads demo data)\n"
            "\n"
            "## Purpose\n"
            "Captures phone IMU + GNSS on real Indian roads so the pipeline gets\n"
            "in-country dead-reckoning examples the public corpora do not have."
            "\n"
            "## App configuration (Android sensor logger)\n"
            "Use a sensor-logging app (e.g. Physics Toolbox / Sensor Logger) and "
            "set:\n"
            "\n"
            "| Sensor | Rate | Purpose |\n"
            "| --- | --- | --- |\n"
            "| Accelerometer (raw) | 100 Hz | IMU acc_x/y/z |\n"
            "| Gyroscope (raw) | 100 Hz | IMU gyr_x/y/z |\n"
            "| Geomagnetic field | 50 Hz | mag_x/y/z |\n"
            "| Pressure | 25 Hz | pressure_hpa |\n"
            "| GPS location | 1 s | gnss lat/lon/alt/speed/heading |\n"
            "| Camera | 10 fps (optional) | visual-odo texture frames |\n"
            "\n"
            "## Output format\n"
            "One CSV per sensor, columns `ts_ms` (epoch milliseconds) followed by "
            "the raw values. Camera frames `frame-<ms>.jpg` in a `camera/` "
            "folder.\n"
            "\n"
            "## Drive plan (three runs)\n"
            "1. Car, 5 min, GNSS always on — baseline corridor.\n"
            "2. Car, 5 min, 60 s tunnel/GNSS blackout — forced-outage eval.\n"
            "3. Two-wheeler, 5 min — phone placement variance.\n"
            "\n"
            "Record these in `metadata.json` for every run: vehicle model, phone "
            "model, phone placement (dashboard / pocket / handlebar), and the "
            "run mode (`gps` / `blackout`) with the blackout window."
            "\n"
            "## Run directory schema (ingested by aeronavis.data)\n"
            "```\n"
            "<run_id>/\n"
            "  metadata.json     vehicle, phone, placement, mode, blackout_start_s\n"
            "  accel.csv         ts_ms,acc_x,acc_y,acc_z\n"
            "  gyro.csv          ts_ms,gyr_x,gyr_y,gyr_z\n"
            "  mag.csv           ts_ms,mag_x,mag_y,mag_z\n"
            "  pressure.csv      ts_ms,pressure_hpa\n"
            "  gnss.csv          ts_ms,lat,lon,alt_m,speed_mps,heading_deg,\n"
            "                    pdop,cno_db_hz\n"
            "  camera/frame-*.jpg (optional)\n"
            "```\n"
            "Keep files clipped to a single continuous session; split multi-stop "
            "drives into separate run folders."
        )

    def _write_schema_csv(self) -> None:
        (self.root / "schema.csv").write_text(
            "\n".join(
                [
                    "sensor,column,type,unit",
                    "accel,ts_ms,float,ms",
                    "accel,acc_x,float,m/s^2",
                    "accel,acc_y,float,m/s^2",
                    "accel,acc_z,float,m/s^2",
                    "gyro,ts_ms,float,ms",
                    "gyro,gyr_x,float,rad/s",
                    "gyro,gyr_y,float,rad/s",
                    "gyro,gyr_z,float,rad/s",
                    "mag,ts_ms,float,ms",
                    "mag,mag_x,float,uT",
                    "mag,mag_y,float,uT",
                    "mag,mag_z,float,uT",
                    "pressure,ts_ms,float,ms",
                    "pressure,pressure_hpa,float,hPa",
                    "gnss,ts_ms,float,ms",
                    "gnss,lat,float,deg",
                    "gnss,lon,float,deg",
                    "gnss,alt_m,float,m",
                    "gnss,speed_mps,float,m/s",
                    "gnss,heading_deg,float,deg",
                    "gnss,pdop,float,dim",
                    "gnss,cno_db_hz,float,dB-Hz",
                    "truth,ts_ms,float,ms",
                    "truth,x_m,float,m",
                    "truth,y_m,float,m",
                    "truth,z_m,float,m",
                    "truth,heading,float,rad",
                ]
            ),
            encoding="utf-8",
        )


HANDLERS: dict[str, type[SourceHandler]] = {
    cls.name: cls for cls in (IOVNBDHandler, RoninHandler, IDOLHandler, SelfCollectedHandler)
}


def all_handlers(config: Config) -> list[SourceHandler]:
    return [cls(config) for cls in HANDLERS.values()]
