"""Check a flight log written by demo/fly_sim.py against the acceptance criteria of plan 0.2a.

Checked here:
  M1  no non-zero command outside FOLLOW and POINT_MOVE (0 violations required)
  M2  braking: from a move-to-HOVER transition to drone speed < 0.1 m/s, p95 <= 0.5 s
      (speed is a finite difference of drone_x/y/z, so it is approximate)
  M8  latency_ms: median <= 50 ms, p95 <= 80 ms, and frame rate >= 25 FPS
  M9  range_mode changes: at most 1 change in any 10 s window

Usage:
    python tools/check_log.py logs/run1.csv
Exit status 0 when every checked criterion passes, 1 otherwise. A criterion with no data
(for example M2 with no braking event) is reported as "n/a" and does not fail the run.
"""
from __future__ import annotations

import csv
import math
import sys

MOVING = ("FOLLOW", "POINT_MOVE")
BRAKE_SPEED = 0.1      # m/s
M2_P95 = 0.5           # s
M8_MEDIAN_MS = 50.0
M8_P95_MS = 80.0
M8_MIN_FPS = 25.0
M9_MAX_PER_10S = 1


def load(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _f(row, key, default=0.0):
    v = row.get(key)
    if v is None or v == "":
        return default
    return float(v)


def percentile(values, q):
    xs = sorted(values)
    if not xs:
        return math.nan
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def m1_violations(rows) -> list[int]:
    bad = []
    for i, r in enumerate(rows):
        cmd = [_f(r, "cmd_forward"), _f(r, "cmd_right"), _f(r, "cmd_up"), _f(r, "cmd_yaw")]
        if any(c != 0.0 for c in cmd) and r["mode"] not in MOVING:
            bad.append(i)
    return bad


def m2_brake_times(rows) -> list[float]:
    """Seconds from each move-to-HOVER transition to the first frame with speed < 0.1 m/s."""
    times = []
    speeds = []
    for i in range(1, len(rows)):
        dt = _f(rows[i], "t") - _f(rows[i - 1], "t")
        if dt <= 0:
            speeds.append(math.nan)
            continue
        d = [_f(rows[i], k) - _f(rows[i - 1], k) for k in ("drone_x", "drone_y", "drone_z")]
        speeds.append(math.sqrt(sum(x * x for x in d)) / dt)
    speeds = [0.0] + speeds
    for i in range(1, len(rows)):
        if rows[i - 1]["mode"] in MOVING and rows[i]["mode"] == "HOVER":
            t0 = _f(rows[i], "t")
            for j in range(i, len(rows)):
                s = speeds[j]
                if not math.isnan(s) and s < BRAKE_SPEED:
                    times.append(_f(rows[j], "t") - t0)
                    break
            else:
                times.append(math.inf)
    return times


def m8_stats(rows):
    lat = [_f(r, "latency_ms") for r in rows]
    ts = [_f(r, "t") for r in rows]
    span = ts[-1] - ts[0] if len(ts) > 1 else 0.0
    fps = (len(rows) - 1) / span if span > 0 else math.nan
    return {"median_ms": percentile(lat, 0.5), "p95_ms": percentile(lat, 0.95), "fps": fps}


def m9_max_switches_per_10s(rows) -> int:
    changes = []
    prev = None
    for r in rows:
        mode = r["range_mode"]
        if prev is not None and mode != prev and {mode, prev} <= {"near", "far"}:
            changes.append(_f(r, "t"))
        prev = mode
    worst = 0
    for i, t0 in enumerate(changes):
        n = sum(1 for t in changes[i:] if t - t0 < 10.0)
        worst = max(worst, n)
    return worst


def analyze(rows) -> dict:
    m1 = m1_violations(rows)
    brakes = m2_brake_times(rows)
    m8 = m8_stats(rows)
    m9 = m9_max_switches_per_10s(rows)
    results = {
        "M1": {"violations": len(m1), "pass": len(m1) == 0},
        "M2": {"events": len(brakes), "p95_s": percentile(brakes, 0.95) if brakes else math.nan,
               "pass": (percentile(brakes, 0.95) <= M2_P95) if brakes else None},
        "M8": {**m8, "pass": (m8["median_ms"] <= M8_MEDIAN_MS and m8["p95_ms"] <= M8_P95_MS
                              and m8["fps"] >= M8_MIN_FPS) if rows else None},
        "M9": {"max_switches_per_10s": m9, "pass": m9 <= M9_MAX_PER_10S},
    }
    return results


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) != 1:
        print(__doc__)
        return 2
    rows = load(argv[0])
    res = analyze(rows)
    failed = False
    m1 = res["M1"]
    print(f"M1 violations={m1['violations']} -> {'PASS' if m1['pass'] else 'FAIL'}")
    m2 = res["M2"]
    if m2["events"] == 0:
        print("M2 no braking events -> n/a")
    else:
        print(f"M2 events={m2['events']} p95={m2['p95_s']:.3f}s -> {'PASS' if m2['pass'] else 'FAIL'}")
    m8 = res["M8"]
    print(f"M8 median={m8['median_ms']:.1f}ms p95={m8['p95_ms']:.1f}ms fps={m8['fps']:.1f} -> "
          f"{'PASS' if m8['pass'] else ('n/a' if m8['pass'] is None else 'FAIL')}")
    m9 = res["M9"]
    print(f"M9 max range switches per 10 s={m9['max_switches_per_10s']} -> {'PASS' if m9['pass'] else 'FAIL'}")
    for key in ("M1", "M2", "M8", "M9"):
        if res[key]["pass"] is False:
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
