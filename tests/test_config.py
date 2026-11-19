"""Defaults, YAML override and validation of ControlConfig."""
import pytest

from dronectl.config import ControlConfig, load_config, validate


def test_defaults_match_spec_values():
    cfg = ControlConfig()
    assert cfg.near_enter_q == 0.30 and cfg.far_enter_q == 0.20
    assert cfg.v_h == 1.5 and cfg.v_v == 1.0 and cfg.w_yaw_deg == 45.0
    assert cfg.ema_alpha == 0.35 and cfg.expo_alpha == 0.5
    assert cfg.classifier == "rules"
    assert cfg.takeoff_hold_near_s == 1.0 and cfg.land_hold_far_s == 1.5


def test_load_without_file_returns_defaults():
    assert load_config() == ControlConfig()


def test_yaml_file_overrides_values(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("v_h: 0.8\nfull_xy: 0.6\n", encoding="utf-8")
    cfg = load_config(str(path))
    assert cfg.v_h == 0.8 and cfg.full_xy == 0.6
    assert cfg.v_v == 1.0                       # untouched keys keep defaults


def test_overrides_dict_wins_over_file(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("v_h: 0.8\n", encoding="utf-8")
    assert load_config(str(path), overrides={"v_h": 2.0}).v_h == 2.0


def test_empty_yaml_file_gives_defaults(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    assert load_config(str(path)) == ControlConfig()


def test_yaml_must_be_a_mapping(tmp_path):
    path = tmp_path / "list.yaml"
    path.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(str(path))


def test_unknown_key_is_rejected():
    with pytest.raises(KeyError):
        load_config(overrides={"v_hh": 1.0})


def test_unsupported_classifier_is_rejected():
    with pytest.raises(ValueError):
        load_config(overrides={"classifier": "mlp"})


@pytest.mark.parametrize("bad", [
    {"far_enter_q": 0.4, "near_enter_q": 0.35},
    {"dead_xy": 0.9},
    {"full_z_ratio": 1.0},
    {"dead_yaw_deg": 40.0},
    {"ema_alpha": 0.0},
    {"v_h": -1.0},
])
def test_inconsistent_values_are_rejected(bad):
    with pytest.raises(ValueError):
        validate(ControlConfig(**bad))
