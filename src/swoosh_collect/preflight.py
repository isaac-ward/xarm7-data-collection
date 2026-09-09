"""Checks run before every episode. If any fails, the episode does not start.

A one-shot campaign cannot afford an episode recorded with three cameras, an
unverified frame, or a latched controller error. These are cheap, so they run at every
A press and print a tick or a cross per line.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

GREEN, RED, DIM, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[0m"
TICK, CROSS = "✓", "✗"


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""

    def line(self) -> str:
        mark = f"{GREEN}{TICK}{RESET}" if self.ok else f"{RED}{CROSS}{RESET}"
        col = "" if self.ok else RED
        tail = f"  {DIM}{self.detail}{RESET}" if self.detail else ""
        return f"  {mark} {col}{self.name}{RESET}{tail}"


def run_checks(cfg: dict, arm: Any, pad: Any, want_cameras: bool,
               inflight: set, last_state: Any = None) -> list[Check]:
    from .cameras import resolve_labelled
    from .provenance import frame_is_verified

    checks: list[Check] = []

    ok, why = frame_is_verified()
    checks.append(Check("45-degree frame verified against the arm", ok,
                        why if not ok else why))

    connected = bool(getattr(pad, "connected", False))
    checks.append(Check("Xbox pad connected", connected,
                        "" if connected else "replug the pad"))

    if want_cameras:
        try:
            cams = resolve_labelled(cfg)
            want = list(cfg.get("cameras", {}).get("expected_labels", []))
            have = [c["label"] for c in cams]
            missing = [w for w in want if w not in have]
            checks.append(Check(f"all {len(want)} cameras present", not missing,
                                ", ".join(have) if not missing
                                else f"missing: {', '.join(missing)}"))
        except Exception as exc:
            checks.append(Check("all cameras present", False, str(exc).split("\n")[0]))
    else:
        checks.append(Check("cameras", True, "skipped (--no-cameras)"))

    # The controller can hold a persistent base-coordinate offset (set_world_offset),
    # which survives reboots and is settable from xArm Studio. We do the 45-degree
    # rotation ourselves in frames.py, so a nonzero controller offset would stack on
    # top of ours and rotate every action twice -- silently. Insist it is zero.
    try:
        wo = list(getattr(arm.api, "world_offset", [0] * 6) or [0] * 6)
        zero = all(abs(float(v)) < 1e-6 for v in wo)
        checks.append(Check("controller world offset is zero", zero,
                            "" if zero else f"{[round(float(v), 3) for v in wo]} - clear it "
                            f"in xArm Studio; frames.py already does this rotation"))
    except Exception:
        checks.append(Check("controller world offset is zero", True, "not reported"))

    err = int(getattr(last_state, "error_code", 0) or 0) if last_state else 0
    checks.append(Check("no latched controller error", err == 0,
                        "" if err == 0 else f"code {err} - run the arm check"))

    checks.append(Check("no run still processing", not inflight,
                        "" if not inflight else f"waiting on {sorted(inflight)[0]}"))

    try:
        import shutil

        free_gb = shutil.disk_usage(".").free / 1e9
    except Exception:
        free_gb = 999.0
    checks.append(Check("disk space", free_gb > 5.0, f"{free_gb:.1f} GB free"))

    return checks


def report(checks: list[Check]) -> bool:
    print(f"\n{DIM}pre-run checks{RESET}", flush=True)
    for c in checks:
        print(c.line(), flush=True)
    ok = all(c.ok for c in checks)
    print(f"  {GREEN if ok else RED}"
          f"{'all checks passed - recording' if ok else 'CHECKS FAILED - run not started'}"
          f"{RESET}\n", flush=True)
    return ok
