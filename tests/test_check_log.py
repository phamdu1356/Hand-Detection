"""tools/check_log.py on hand-made rows, plus the end-to-end exit codes."""
import csv
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "demo"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import check_log  # noqa: E402
import fly_sim  # noqa: E402


def _row(t, mode="HOVER", x=0.0, cmd=(0.0, 0.0, 0.0, 0.0), rng="near", lat=20.0, action=""):
    return {
        "t": repr(t), "mode": mode, "action": action, "range_mode": rng, "latency_ms": repr(lat),
        "cmd_forward": repr(cmd[0]), "cmd_right": repr(cmd[1]), "cmd_up": repr(cmd[2]),
        "cmd_yaw": repr(cmd[3]), "drone_x": repr(x), "drone_y": "0.0", "drone_z": "1.5",
    }


def test_m1_flags_command_outside_follow():
    rows = [_row(0.0), _row(1 / 30, mode="HOVER", cmd=(0.5, 0, 0, 0)), _row(2 / 30, mode="FOLLOW", cmd=(0.5, 0, 0, 0))]
    assert check_log.m1_violations(rows) == [1]


def test_m1_clean_when_moving_only_in_follow_or_point_move():
    rows = [_row(0.0, mode="FOLLOW", cmd=(0.5, 0, 0, 0)), _row(1 / 30, mode="POINT_MOVE", cmd=(0, 0.5, 0, 0))]
    assert check_log.m1_violations(rows) == []


def test_m2_brake_time_from_drone_speed():
    rows = []
    x = 0.0
    dt = 1 / 30
    # FOLLOW at 1 m/s for 0.5 s, then HOVER: decelerate linearly to zero over 0.4 s
    for i in range(15):
        rows.append(_row(i * dt, mode="FOLLOW", x=x, cmd=(1.0, 0, 0, 0)))
        x += dt
    v = 1.0
    t = 15 * dt
    for i in range(30):
        v = max(0.0, v - 2.5 * dt)
        x += v * dt
        rows.append(_row(t, mode="HOVER", x=x))
        t += dt
    times = check_log.m2_brake_times(rows)
    assert len(times) == 1
    assert 0.3 <= times[0] <= 0.5


def test_m8_median_p95_and_fps():
    rows = [_row(i / 30, lat=10.0 + i % 5) for i in range(60)]
    stats = check_log.m8_stats(rows)
    assert stats["median_ms"] == pytest.approx(12.0)
    assert stats["fps"] == pytest.approx(30.0, rel=1e-3)


def test_m9_counts_switches_in_any_10s_window():
    rows = [_row(0.0, rng="near"), _row(1.0, rng="far"), _row(2.0, rng="near"), _row(12.5, rng="far")]
    assert check_log.m9_max_switches_per_10s(rows) == 2  # changes at 1.0 and 2.0 fall in one window
    rows = [_row(0.0, rng="near"), _row(1.0, rng="far"), _row(12.5, rng="far")]
    assert check_log.m9_max_switches_per_10s(rows) == 1


def test_main_exit_codes(tmp_path):
    good = tmp_path / "good.csv"
    assert fly_sim.main(["--synthetic", "--backend", "mock", "--no-gui", "--log", str(good)]) == 0
    assert check_log.main([str(good)]) == 0

    bad = tmp_path / "bad.csv"
    rows = list(csv.DictReader(open(good, newline="", encoding="utf-8")))
    rows[3]["cmd_forward"] = "0.7"
    rows[3]["mode"] = "HOVER"
    with open(bad, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    assert check_log.main([str(bad)]) == 1


def test_schema_rejects_header_only_and_missing_columns(tmp_path):
    empty = tmp_path / "header_only.csv"
    empty.write_text("t,mode,range_mode,latency_ms,cmd_forward,cmd_right,cmd_up,cmd_yaw,drone_x,drone_y,drone_z\n")
    assert check_log.main([str(empty)]) == 1

    missing = tmp_path / "missing_cols.csv"
    missing.write_text("t,mode\n0.0,HOVER\n")
    assert check_log.main([str(missing)]) == 1


def test_m2_uses_logged_velocity_columns_when_present():
    rows = [
        {**_row(0.0, mode="FOLLOW"), "drone_vx": "0.0", "drone_vy": "-0.5", "drone_vz": "0.0"},
        {**_row(0.1, mode="HOVER"), "drone_vx": "0.0", "drone_vy": "-0.2", "drone_vz": "0.0"},
        {**_row(0.2, mode="HOVER"), "drone_vx": "0.0", "drone_vy": "-0.05", "drone_vz": "0.0"},
    ]
    assert check_log.m2_brake_times(rows) == [pytest.approx(0.1)]
