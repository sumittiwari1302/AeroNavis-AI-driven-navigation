"""Strict YAML configuration loader with a validated, frozen dataclass surface."""

import logging
from dataclasses import dataclass, fields, is_dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, get_args, get_origin

import yaml

logger = logging.getLogger(__name__)

_MISSING = "__aeronavis_missing__"


class ConfigError(ValueError):
    """Raised when config.yaml violates the dataclass spec."""


@dataclass(frozen=True)
class SensorConfig:
    imu_hz: int
    gnss_hz: int
    cam_hz: int


@dataclass(frozen=True)
class ModelConfig:
    window: int
    stride: int
    velocity_hidden: int
    max_params: int
    pseudo_odo_hidden: int
    pseudo_odo_max_params: int
    slip_hidden: int
    slip_max_params: int
    texture_gate_max_params: int
    visual_odo_max_params: int


@dataclass(frozen=True)
class TextureConfig:
    rich: float
    medium: float


@dataclass(frozen=True)
class FusionConfig:
    reseed_jump_max_m: float


@dataclass(frozen=True)
class PredictConfig:
    horizons: list[int]
    threshold: float


@dataclass(frozen=True)
class DataConfig:
    imu_hz: int
    gnss_hz: int
    wheel_hz: int
    truth_hz: int
    mag_hz: int
    baro_hz: int
    cam_hz: int
    min_seq_s: int
    min_gnss_coverage: float
    max_gap_ms: int
    outage_min_gap_s: float
    earth_radius_m: float


@dataclass(frozen=True)
class SplitsConfig:
    train_ratio: float
    val_ratio: float


@dataclass(frozen=True)
class PathsConfig:
    raw: str
    processed: str
    splits: str
    models: str
    benchmarks: str


@dataclass(frozen=True)
class Config:
    sensor: SensorConfig
    model: ModelConfig
    texture: TextureConfig
    fusion: FusionConfig
    predict: PredictConfig
    paths: PathsConfig
    data: DataConfig
    splits: SplitsConfig
    seed: int


# YAML namespace prefix each group dataclass is declared under. Prefixes need not
# equal the Config attribute name (e.g. texture keys carry an extra segment).
GROUP_PREFIX: dict[type, str] = {
    SensorConfig: "sensor",
    ModelConfig: "model",
    TextureConfig: "texture.threshold",
    FusionConfig: "fusion",
    PredictConfig: "predict",
    PathsConfig: "paths",
    DataConfig: "data",
    SplitsConfig: "splits",
}


def _validate_value(key: str, expected: Any, value: Any) -> Any:
    origin = get_origin(expected)
    if origin is list:
        (item_type,) = get_args(expected)
        if not isinstance(value, list):
            raise ConfigError(f"key {key!r}: expected list, got {type(value).__name__}")
        for item in value:
            if not isinstance(item, item_type):
                raise ConfigError(
                    f"key {key!r}: expected list of {item_type.__name__}, got {item!r}"
                )
        return value
    if not isinstance(value, expected):
        raise ConfigError(
            f"key {key!r}: expected {expected.__name__}, got {type(value).__name__} ({value!r})"
        )
    return value


def _build_spec() -> dict[str, tuple[type | None, str, Any]]:
    spec: dict[str, tuple[type | None, str, Any]] = {}
    for group_cls, prefix in GROUP_PREFIX.items():
        for member in fields(group_cls):
            key = f"{prefix}.{member.name}"
            spec[key] = (group_cls, member.name, member.type)
    spec["seed"] = (None, "seed", int)
    return spec


def load_config(path: str | Path) -> Config:
    """Parse config at *path* strictly against the dataclass spec."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError(f"config file {path}: expected a YAML mapping at the root")

    spec = _build_spec()
    unknown = set(raw) - set(spec)
    if unknown:
        raise ConfigError(f"unknown config key(s): {sorted(unknown)!r}")

    grouped: dict[type, dict[str, Any]] = {}
    root_seed = _MISSING
    for key, value in raw.items():
        group_cls, member_name, expected = spec[key]
        validated = _validate_value(key, expected, value)
        if group_cls is None:
            root_seed = validated
        else:
            grouped.setdefault(group_cls, {})[member_name] = validated

    kwargs: dict[str, Any] = {}
    for group in fields(Config):
        group_cls = group.type
        if is_dataclass(group_cls) and group_cls in grouped:
            kwargs[group.name] = group_cls(**grouped[group_cls])
        elif is_dataclass(group_cls):
            raise ConfigError(f"missing config section: {group.name!r}")
    if root_seed is _MISSING:
        raise ConfigError("missing required key: 'seed'")
    kwargs["seed"] = root_seed

    return Config(**kwargs)


@lru_cache(maxsize=1)
def get_config(path: str | Path = "config.yaml") -> Config:
    """Load once per path and hand out the same frozen object on every call."""
    return load_config(path)
