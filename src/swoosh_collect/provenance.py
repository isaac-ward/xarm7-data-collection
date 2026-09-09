"""What was true when this run was recorded.

`action` is stick x translation_rate_mm_s x dt. Change the rate, the deadzone, the
expo, the home pose or the frame matrix between sessions and identical action values
mean different physics -- with nothing in the data saying so. A world model trained
across such a mix learns an average of two different robots.

So every run records the config that produced it, the git SHA, the frame matrix and
its verification status, and what the controller says about its own TCP setup.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from .frames import R_WORLD_FROM_BASE


def matrix_hash() -> str:
    """Short hash of the frame matrix, so a verification can be tied to one matrix."""
    return hashlib.sha256(
        np.asarray(R_WORLD_FROM_BASE, dtype=np.float64).tobytes()
    ).hexdigest()[:12]


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[2], capture_output=True,
            text=True, timeout=5,
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def git_dirty() -> bool | None:
    """True/False, or None when git could not be consulted.

    This returned False on failure, which reads as "the tree was clean" -- the most
    reassuring possible answer to a question we did not actually get to ask. Inside the
    container git refused the bind-mounted repo as dubious ownership, so every run
    recorded git_sha "unknown" alongside git_dirty False, i.e. no provenance while
    looking like good provenance.
    """
    try:
        r = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).resolve().parents[2], capture_output=True,
            text=True, timeout=5,
        )
        if r.returncode != 0:
            return None
        return bool(r.stdout.strip())
    except Exception:
        return None


def _verify_path() -> Path:
    return Path(__file__).resolve().parents[2] / "campaigns" / ".frame_verification.json"


def save_frame_verification(result: dict[str, Any]) -> Path:
    p = _verify_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "matrix_hash": matrix_hash(),
        "when_unix": time.time(),
        "result": result,
    }, indent=2))
    return p


def load_frame_verification() -> dict[str, Any] | None:
    p = _verify_path()
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def frame_is_verified() -> tuple[bool, str]:
    """(ok, human reason). The matrix hash must match the verified one."""
    v = load_frame_verification()
    if v is None:
        return False, ("the 45-degree frame has never been checked against the arm. "
                       "Run the 'verify 45 frame' sanity check (it jogs the arm 20 mm "
                       "along each world axis).")
    if v.get("matrix_hash") != matrix_hash():
        return False, (f"the frame matrix changed since it was verified "
                       f"({v.get('matrix_hash')} -> {matrix_hash()}). Re-verify.")
    if not v.get("result", {}).get("all_correct"):
        return False, "the last frame verification FAILED; fix frames.py before collecting."
    age_h = (time.time() - float(v.get("when_unix", 0))) / 3600.0
    return True, f"frame verified {age_h:.1f} h ago against matrix {matrix_hash()}"


def snapshot(cfg: dict[str, Any], arm: Any = None) -> dict[str, Any]:
    """Everything needed to interpret this run's numbers later."""
    out: dict[str, Any] = {
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "matrix_hash": matrix_hash(),
        "R_WORLD_FROM_BASE": np.asarray(R_WORLD_FROM_BASE).tolist(),
        "frame_verification": load_frame_verification(),
        "config": {k: v for k, v in cfg.items() if k != "_config_path"},
        "measurements": load_measurements(),   # servo lag, camera latency
    }
    if arm is not None and getattr(arm, "api", None) is not None:
        for name, fn in (("tcp_offset", "get_tcp_offset"),
                         ("tcp_load", "get_tcp_load"),
                         ("version", "get_version")):
            try:
                code, val = getattr(arm.api, fn)()
                out[name] = val if code == 0 else None
            except Exception:
                out[name] = None
    return out


def _measure_path() -> Path:
    return Path(__file__).resolve().parents[2] / "campaigns" / ".measurements.json"


def save_measurement(key: str, value: Any) -> Path:
    """One-off rig measurements (servo lag, camera latency) that describe the setup
    rather than any single run. Copied into every run's provenance."""
    p = _measure_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    cur: dict[str, Any] = {}
    if p.is_file():
        try:
            cur = json.loads(p.read_text())
        except Exception:
            cur = {}
    cur[key] = {"value": value, "when_unix": time.time()}
    p.write_text(json.dumps(cur, indent=2))
    return p


def load_measurements() -> dict[str, Any]:
    p = _measure_path()
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}
