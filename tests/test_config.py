"""Proves the strict config loader: valid load, bad type, unknown key, singleton."""

from pathlib import Path

import pytest
import yaml

from aeronavis.config import Config, ConfigError, get_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_YAML = REPO_ROOT / "config.yaml"


def _write(raw: dict, tmp_path: Path) -> Path:
    target = tmp_path / "config.yaml"
    target.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return target


def test_loads_ok():
    config = get_config(str(CONFIG_YAML))
    assert isinstance(config, Config)
    assert config.sensor.imu_hz == 100
    assert config.sensor.gnss_hz == 1
    assert config.sensor.cam_hz == 10
    assert config.model.window == 200
    assert config.model.stride == 10
    assert config.model.velocity_hidden == 128
    assert config.model.max_params == 500000
    assert config.texture.rich == 2000.0
    assert config.texture.medium == 400.0
    assert config.fusion.reseed_jump_max_m == 0.3
    assert config.predict.horizons == [5, 10, 15]
    assert config.predict.threshold == 0.85
    assert config.paths.raw == "data/raw"
    assert config.paths.models == "models"
    assert config.seed == 42


def test_wrong_type_raises(tmp_path):
    raw = yaml.safe_load(CONFIG_YAML.read_text(encoding="utf-8"))
    raw["sensor.imu_hz"] = "not-an-int"
    path = _write(raw, tmp_path)
    with pytest.raises(ConfigError, match="sensor.imu_hz"):
        get_config.cache_clear()
        get_config(str(path))


def test_wrong_type_in_list_raises(tmp_path):
    raw = yaml.safe_load(CONFIG_YAML.read_text(encoding="utf-8"))
    raw["predict.horizons"] = [5, "ten"]
    path = _write(raw, tmp_path)
    with pytest.raises(ConfigError, match="horizons"):
        get_config(str(path))


def test_unknown_key_raises(tmp_path):
    raw = yaml.safe_load(CONFIG_YAML.read_text(encoding="utf-8"))
    raw["bogus.option"] = 1
    path = _write(raw, tmp_path)
    with pytest.raises(ConfigError, match="bogus.option"):
        get_config(str(path))


def test_missing_key_raises(tmp_path):
    raw = yaml.safe_load(CONFIG_YAML.read_text(encoding="utf-8"))
    del raw["seed"]
    path = _write(raw, tmp_path)
    with pytest.raises(ConfigError, match="seed"):
        get_config(str(path))


def test_singleton_identity():
    get_config.cache_clear()
    first = get_config(str(CONFIG_YAML))
    second = get_config(str(CONFIG_YAML))
    assert first is second
    get_config.cache_clear()


def test_flat_only_raises_on_nested_yaml(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("sensor:\n  imu_hz: 100\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="sensor"):
        get_config(str(path))
