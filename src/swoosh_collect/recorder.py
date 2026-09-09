"""Raw stream logging. This is the authoritative record; the LeRobot export is derived.

FOUR SEPARATE STREAMS, deliberately never merged at write time:

  controller.jsonl   what the OPERATOR did   -- the world-model action signal
  commanded.jsonl    what WE decided         -- target pose in world and base frames
  xarm_command.jsonl what the SDK was TOLD   -- the exact arguments to set_servo_cartesian
  arm_state.jsonl    what the ROBOT reports  -- joints, measured pose, gripper, errors

Keeping `commanded` and `xarm_command` apart looks redundant until the frame maths is
wrong: the first is intent in world coordinates, the second is the literal bytes sent
after the 45-degree rotation and workspace clamp. When they disagree, the difference
localises the bug. Merging them would hide exactly the class of error that cost weeks
on lego_assemblies, where the published action turned out to be raw controller pose in
a frame nobody recorded.

Every row carries `t`, seconds since the run's t0, on one monotonic clock shared with
the camera frame timestamps. Wall-clock is recorded once in run.json; monotonic is used
for everything else because it cannot jump.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, TextIO


class StreamWriter:
    """Line-buffered JSONL. One file per stream."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fh: TextIO = path.open("w", buffering=1, encoding="utf-8")
        self.rows = 0

    def write(self, row: dict[str, Any]) -> None:
        self._fh.write(json.dumps(row, separators=(",", ":")) + "\n")
        self.rows += 1

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


class RunRecorder:
    """All raw streams for one A-to-B episode."""

    STREAMS = ("controller", "commanded", "xarm_command", "arm_state", "tick")

    def __init__(self, run_dir: Path, t0: float) -> None:
        self.run_dir = run_dir
        self.raw_dir = run_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.t0 = t0
        self.wall_start = time.time()
        self.w = {n: StreamWriter(self.raw_dir / f"{n}.jsonl") for n in self.STREAMS}

    # one method per stream so a caller cannot write the wrong shape to the wrong file
    def controller(self, row: dict[str, Any]) -> None:
        self.w["controller"].write(row)

    def commanded(self, t: float, world_xyz, rpy_deg, yaw_world_deg: float,
                  gripper_closure: float, clamped: bool,
                  reanchored: bool = False) -> None:
        self.w["commanded"].write({
            "t": t,
            "target_world_xyz_mm": [float(v) for v in world_xyz],
            # base-frame euler, despite the neutral key name -- see the export, which
            # ships it as action.commanded_rpy_base
            "target_rpy_deg": [float(v) for v in rpy_deg],
            # yaw ACCUMULATED SINCE THE LAST RE-SEED, not an absolute world yaw: it is
            # reset to 0 by re-home, clear-errors and servo recovery. Absolute
            # orientation lives in target_rpy_deg.
            "target_yaw_world_deg": float(yaw_world_deg),
            "gripper_closure": float(gripper_closure),
            "clamped_by_workspace": bool(clamped),
            # True on the tick where the target was snapped back to the measured pose
            # because the arm stopped following. Without this the action says "still
            # pushing" while the commanded pose teleports backwards, which is
            # indistinguishable from intent.
            "reanchored": bool(reanchored),
        })

    def xarm_command(self, t: float, pose_base: list[float], servo_code: Any,
                     gripper_raw: float | None,
                     gripper_sent: dict | None = None) -> None:
        """Exactly what the SDK was handed, plus what it returned.

        The servo pose is literally the argument, on this tick. The gripper is not: it
        goes through a worker because the SDK has no non-blocking call, so
        `set_gripper_position_queued` is what the loop ASKED for and
        `gripper_sent_*` is what the worker actually handed the SDK, when, and what
        came back. Those are different facts and conflating them made this stream a
        statement of intent for the gripper.
        """
        row = {
            "t": t,
            "set_servo_cartesian_pose_base": [float(v) for v in pose_base],
            "servo_code": None if servo_code is None else int(servo_code),
            "set_gripper_position_queued": (None if gripper_raw is None
                                            else float(gripper_raw)),
        }
        if gripper_sent:
            row["gripper_sent_position"] = gripper_sent.get("sent_position")
            # already on the run's clock (the arm is told t_loop0)
            st = gripper_sent.get("sent_t")
            row["gripper_sent_t"] = st
            row["gripper_sent_age_s"] = None if st is None else t - float(st)
            row["gripper_sent_code"] = gripper_sent.get("sent_code")
            if gripper_sent.get("sent_error"):
                row["gripper_sent_error"] = gripper_sent["sent_error"]
        self.w["xarm_command"].write(row)

    def tick(self, t: float, dt: float, n: int, servo_code: Any, pad_ok: bool) -> None:
        """Loop health. Without this a degraded loop rate is invisible, and the action
        (stick * rate * dt) silently means something different when the loop is slow."""
        self.w["tick"].write({
            "t": t, "dt": dt, "n": n,
            "servo_code": None if servo_code is None else int(servo_code),
            "pad_connected": bool(pad_ok),
        })

    def arm_state(self, row: dict[str, Any]) -> None:
        self.w["arm_state"].write(row)

    def counts(self) -> dict[str, int]:
        return {n: w.rows for n, w in self.w.items()}

    def close(self, stopped_by: str, extra: dict[str, Any] | None = None) -> dict:
        for w in self.w.values():
            w.close()
        # MERGE, don't overwrite: new_run_dir() already wrote `status: recording` and
        # the start timestamp so the dashboard could show the run immediately.
        existing: dict[str, Any] = {}
        rj = self.run_dir / "run.json"
        if rj.is_file():
            try:
                existing = json.loads(rj.read_text())
            except Exception:
                existing = {}
        meta = {
            **existing,
            "t0_monotonic": self.t0,
            "wall_start_unix": self.wall_start,
            "wall_end_unix": time.time(),
            "duration_s": time.time() - self.wall_start,
            "stopped_by": stopped_by,
            "clock_origin": "t_loop0 -- ALL streams, cameras included, share this origin",
            "stream_rows": self.counts(),
            "status": "processing",   # the collector flips this to `ready` when the
                                      # summary mp4 and plots have been built
            **(extra or {}),
        }
        rj.write_text(json.dumps(meta, indent=2))
        return meta
