"""Shared schema, cryptographic/perceptual hashing and statistics for the 3-tier benchmark.

Imports only the standard library and numpy (OpenCV lazily for dHash64).
"""
from __future__ import annotations

import hashlib
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

# Exact 14 keypoints checked by dronectl.gestures.hand_command (must stay in sync).
CONTROL_KPT_INDICES: tuple[int, ...] = (0, 2, 4, 5, 6, 8, 9, 10, 12, 14, 16, 17, 18, 20)
PALM_INDICES: tuple[int, ...] = (0, 5, 9, 13, 17)
KPT_NAMES: tuple[str, ...] = (
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
)

CANONICAL_COMMANDS: tuple[str, ...] = (
    "takeoff",
    "point_right",
    "point_left",
    "point_up",
    "point_down",
    "stop",
    "land",
    "neutral",
)

MANIFEST_COLUMNS: tuple[str, ...] = (
    "participant_id", "session_id", "trial_id", "split", "interface", "source",
    "device", "distance_m", "lighting", "resolution", "video_path",
    "video_sha256", "labels_path", "log_path",
)
LABEL_COLUMNS: tuple[str, ...] = (
    "trial_id", "event_id", "kind", "command", "cue_s", "onset_s", "offset_s",
    "target_right_m", "target_up_m", "timeout_s",
)
LOG_REQUIRED_COLUMNS: tuple[str, ...] = (
    "t", "mode", "range_mode", "latency_ms",
    "cmd_forward", "cmd_right", "cmd_up", "cmd_yaw",
    "drone_x", "drone_y", "drone_z",
)

SIZE_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("<40", 0.0, 40.0),
    ("40-80", 40.0, 80.0),
    ("80-160", 80.0, 160.0),
    (">=160", 160.0, float("inf")),
)


def size_bucket(size_px: float) -> str:
    """Assign a hand pixel size max(bw, bh) to one of the standard reporting buckets."""
    for name, lo, hi in SIZE_BUCKETS:
        if lo <= size_px < hi:
            return name
    return ">=160"


def file_sha256(path: str | Path) -> str:
    """Return lowercase hex SHA-256 of a file, read in 1 MiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def git_commit_info(repo_root: str | Path) -> dict[str, Any]:
    """Return current git HEAD hash and dirty flag for audit reproducibility."""
    root = str(repo_root)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.DEVNULL
        )
        return {"commit": commit, "dirty": bool(status.strip())}
    except Exception:
        return {"commit": "unknown", "dirty": True}


def dhash64(image_or_path: str | Path | np.ndarray, hash_size: int = 8) -> int:
    """Compute a 64-bit difference hash (dHash) to detect recompressed near-duplicate images."""
    import cv2

    if isinstance(image_or_path, (str, Path)):
        gray = cv2.imread(str(image_or_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"cannot read image for dHash: {image_or_path}")
    else:
        arr = np.asarray(image_or_path)
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY) if arr.ndim == 3 else arr

    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    bits = diff.reshape(-1)
    val = 0
    for i, b in enumerate(bits):
        if b:
            val |= 1 << i
    return int(val)


def hamming_distance(a: int, b: int) -> int:
    """Bitwise Hamming distance between two integer hashes."""
    return (int(a) ^ int(b)).bit_count()


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score confidence interval (default 95%) for binomial proportion k / n."""
    if n <= 0:
        return (math.nan, math.nan)
    if k < 0 or k > n:
        raise ValueError(f"invalid binomial counts: k={k}, n={n}")
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def validate_manifest_rows(rows: list[dict], fieldnames: list[str] | None = None) -> list[str]:
    """Validate benchmark manifest schema and prevent dev/test leakage by video hash or participant."""
    errors: list[str] = []
    if fieldnames is not None:
        cols = set(fieldnames)
    elif rows:
        cols = set.intersection(*(set(r.keys()) for r in rows))
    else:
        cols = set()
    missing = [c for c in MANIFEST_COLUMNS if c not in cols]
    if missing:
        errors.append(f"missing manifest columns: {missing}")
    if not rows:
        errors.append("manifest contains no rows")
        return errors

    by_split_hash: dict[str, set[str]] = {}
    by_split_person: dict[str, set[str]] = {}
    seen_trials: set[str] = set()
    for i, r in enumerate(rows):
        tid = (r.get("trial_id") or "").strip()
        sid = (r.get("session_id") or "").strip()
        split = (r.get("split") or "").strip()
        vhash = (r.get("video_sha256") or "").strip()
        pid = (r.get("participant_id") or "").strip()
        iface = (r.get("interface") or "").strip()
        src = (r.get("source") or "").strip()
        vpath = (r.get("video_path") or "").strip()
        if not tid:
            errors.append(f"row {i}: empty trial_id")
        elif tid in seen_trials:
            errors.append(f"row {i}: duplicate trial_id {tid!r}")
        seen_trials.add(tid)
        if not sid:
            errors.append(f"row {i}: empty session_id")
        if split not in ("dev", "pilot", "test"):
            errors.append(f"row {i}: invalid split {split!r}")
        if not pid:
            errors.append(f"row {i}: empty participant_id")
        else:
            by_split_person.setdefault(split, set()).add(pid)
        if not vhash:
            errors.append(f"row {i}: empty video_sha256")
        else:
            by_split_hash.setdefault(split, set()).add(vhash)
        if not iface:
            errors.append(f"row {i}: empty interface")
        if not src:
            errors.append(f"row {i}: empty source")
        if not vpath:
            errors.append(f"row {i}: empty video_path")

    dev_hashes = by_split_hash.get("dev", set()) | by_split_hash.get("pilot", set())
    test_hashes = by_split_hash.get("test", set())
    leaked_hashes = sorted(dev_hashes & test_hashes)
    if leaked_hashes:
        errors.append(f"video_sha256 leaked between dev/pilot and test: {leaked_hashes}")

    dev_people = by_split_person.get("dev", set()) | by_split_person.get("pilot", set())
    test_people = by_split_person.get("test", set())
    leaked_people = sorted(dev_people & test_people)
    if leaked_people:
        errors.append(f"participant_id leaked between dev/pilot and test: {leaked_people}")
    return errors


def validate_label_rows(rows: list[dict], fieldnames: list[str] | None = None) -> list[str]:
    """Validate intent/neutral label rows: schema, timestamps, vocab, non-overlapping spans."""
    errors: list[str] = []
    if fieldnames is not None:
        cols = set(fieldnames)
    elif rows:
        cols = set.intersection(*(set(r.keys()) for r in rows))
    else:
        cols = set()
    missing = [c for c in LABEL_COLUMNS if c not in cols]
    if missing:
        errors.append(f"missing label columns: {missing}")
    if not rows:
        errors.append("labels file contains no rows")
        return errors

    intents_by_trial: dict[str, list[tuple[float, float, str]]] = {}
    neutrals_by_trial: dict[str, list[tuple[float, float, str]]] = {}
    seen_events: set[tuple[str, str]] = set()
    for i, r in enumerate(rows):
        kind = (r.get("kind") or "").strip()
        cmd = (r.get("command") or "").strip()
        tid = (r.get("trial_id") or "").strip()
        eid = (r.get("event_id") or "").strip()
        if not tid:
            errors.append(f"row {i}: empty trial_id")
        if not eid:
            errors.append(f"row {i}: empty event_id")
            eid = str(i)
        elif (tid, eid) in seen_events:
            errors.append(f"row {i}: duplicate event_id {eid!r} in trial {tid!r}")
        seen_events.add((tid, eid))
        if kind not in ("intent", "neutral"):
            errors.append(f"row {i}: invalid kind {kind!r}")
        if cmd not in CANONICAL_COMMANDS:
            errors.append(f"row {i}: unknown command {cmd!r}")
        if kind == "neutral" and cmd != "neutral":
            errors.append(f"row {i}: neutral row must use command='neutral', got {cmd!r}")
        if kind == "intent" and cmd == "neutral":
            errors.append(f"row {i}: intent row cannot use command='neutral'")
        try:
            onset = float(r["onset_s"])
            offset = float(r["offset_s"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"row {i}: non-numeric onset_s/offset_s")
            continue
        if not (math.isfinite(onset) and math.isfinite(offset)) or onset < 0.0 or onset >= offset:
            errors.append(f"row {i}: require 0 <= onset_s < offset_s, got ({onset}, {offset})")
            continue

        cue_raw = str(r.get("cue_s") or "").strip()
        if cue_raw:
            try:
                cue_val = float(cue_raw)
                if not math.isfinite(cue_val) or cue_val < 0.0 or cue_val > offset:
                    errors.append(f"row {i}: cue_s must satisfy 0 <= cue_s <= offset_s, got {cue_val}")
            except ValueError:
                errors.append(f"row {i}: non-numeric cue_s {cue_raw!r}")

        timeout_raw = str(r.get("timeout_s") or "").strip()
        if timeout_raw:
            try:
                to_val = float(timeout_raw)
                if not math.isfinite(to_val) or to_val <= 0.0:
                    errors.append(f"row {i}: timeout_s must be positive, got {to_val}")
            except ValueError:
                errors.append(f"row {i}: non-numeric timeout_s {timeout_raw!r}")

        if kind == "intent":
            intents_by_trial.setdefault(tid, []).append((onset, offset, eid))
        elif kind == "neutral":
            neutrals_by_trial.setdefault(tid, []).append((onset, offset, eid))

    for tid, spans in intents_by_trial.items():
        spans.sort()
        for (a0, a1, ea), (b0, b1, eb) in zip(spans, spans[1:]):
            if b0 < a1 - 1e-9:
                errors.append(f"trial {tid!r}: overlapping intents {ea!r} [{a0},{a1}] and {eb!r} [{b0},{b1}]")
        for n0, n1, en in neutrals_by_trial.get(tid, []):
            for a0, a1, ea in spans:
                if max(a0, n0) < min(a1, n1) - 1e-9:
                    errors.append(
                        f"trial {tid!r}: neutral {en!r} [{n0},{n1}] overlaps intent {ea!r} [{a0},{a1}]"
                    )
    return errors
