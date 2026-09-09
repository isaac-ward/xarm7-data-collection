"""Simulated hardware: drop-in stand-ins for the arm, the pad and the cameras.

`swoosh-collect --simulate` swaps these in for RightArm / XboxPad / CameraRig. Nothing
else changes, so the simulator exercises the REAL control loop, the REAL recorder, the
REAL dashboard and the REAL export -- which is the whole point. An earlier "preview
mode" pushed fake numbers straight into the dashboard's LiveState and therefore tested
none of that.

THE PHYSICS IS DELIBERATELY CRUDE. The arm follows the commanded pose with a
first-order lag and the joint angles are smooth bounded wobble, not an IK solution.
That is enough to drive every screen and to exercise record -> process -> play back ->
export end to end. It is NOT a robot model and nothing here is used on hardware.

The simulated cameras DO write real mp4s and real timestamp files, so a full A-to-B
episode in simulation produces a genuine run folder that summarises, plays back and
exports exactly like a real one.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .arm import ArmState
from .config import get
from .frames import base_to_world
from .xbox import PadSnapshot


class _FakeApi:
    """Just enough of XArmAPI for preflight and sanity to interrogate."""

    world_offset = [0.0] * 6
    mode = 1
    state = 0

    def get_inverse_kinematics(self, pose, **_k):
        """Crude reachability stand-in: inside a plausible xArm7 shell -> code 0.

        Enough to exercise the workspace-reach button in simulation; the real check
        uses the controller's own solver.
        """
        import numpy as np

        r = float(np.linalg.norm(np.asarray(pose[:3], dtype=float)))
        return (0 if 200.0 < r < 900.0 else 1), [0.0] * 7

    def get_tcp_offset(self): return 0, [0.0] * 6
    def set_mode(self, *_a, **_k): return 0
    def set_state(self, *_a, **_k): return 0
    def set_gripper_position(self, *_a, **_k): return 0
    def get_gripper_position(self): return 0, 400.0


class SimulatedArm:
    """Follows the commanded pose with a lag. Same interface as RightArm.

    `is_simulated` lets checks that can only be meaningful against real hardware --
    the FK check, in particular -- say so instead of reporting a false alarm.
    """

    is_simulated = True

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.ip = "simulated"
        self.api = _FakeApi()
        self._t0 = time.monotonic()
        # Start where the real home pose roughly puts the TCP, in the BASE frame.
        self._pose = np.array([330.0, 0.0, 621.0, 180.0, 0.0, 0.0])
        self._target = self._pose.copy()
        self._grip_cmd = 0.0
        self._grip_pos = float(get(cfg, "gripper.open_position", 850))
        self._lock = threading.Lock()

    # -- lifecycle -----------------------------------------------------------
    def connect(self) -> None:
        print("[sim] simulated arm connected (no hardware)", flush=True)

    def go_home(self, announce: bool = True) -> None:
        with self._lock:
            self._pose = np.array([330.0, 0.0, 621.0, 180.0, 0.0, 0.0])
            self._target = self._pose.copy()

    def start_streaming(self) -> None: ...
    def start_gripper_worker(self) -> None: ...

    def shutdown(self) -> None:
        print("[sim] simulated arm shut down", flush=True)

    # -- io ------------------------------------------------------------------
    def servo_pose(self, pose_base: list[float]) -> int:
        with self._lock:
            self._target = np.asarray(pose_base, dtype=np.float64)
        return 0

    def set_gripper(self, closure: float) -> float | None:
        op = float(get(self.cfg, "gripper.open_position", 850))
        cl = float(get(self.cfg, "gripper.closed_position", 0))
        with self._lock:
            self._grip_cmd = max(0.0, min(1.0, float(closure)))
        return op + (cl - op) * self._grip_cmd

    def _advance(self) -> None:
        """First-order lag toward the target. Called on every read."""
        with self._lock:
            self._pose += (self._target - self._pose) * 0.25
            op = float(get(self.cfg, "gripper.open_position", 850))
            cl = float(get(self.cfg, "gripper.closed_position", 0))
            want = op + (cl - op) * self._grip_cmd
            self._grip_pos += (want - self._grip_pos) * 0.25

    def read_state(self, t: float) -> ArmState:
        self._advance()
        el = time.monotonic() - self._t0
        with self._lock:
            pose = self._pose.copy()
            grip = self._grip_pos
        st = ArmState(t=t)
        # Smooth, bounded wobble. NOT an IK solution -- see the module docstring.
        st.joints_deg = [
            float(np.clip(30.0 * math.sin(0.35 * el + j * 0.9), -170.0, 170.0))
            for j in range(7)
        ]
        st.pose_base = [float(v) for v in pose]
        st.pose_world_xyz = base_to_world(pose[:3]).tolist()
        st.gripper_pos = float(grip)
        st.state, st.mode, st.error_code, st.warn_code = 0, 1, 0, 0
        return st

    def current_pose_world(self):
        with self._lock:
            pose = self._pose.copy()
        return base_to_world(pose[:3]), [float(v) for v in pose]


class SimulatedPad:
    """Smooth sinusoid sticks. Buttons come from the dashboard, not from here."""

    def __init__(self, cfg: dict[str, Any], idle: bool = False) -> None:
        self.cfg = cfg
        self.connected = True
        self.idle = idle          # True -> all zeros, for a "connected but still" rig
        self._t0 = time.monotonic()
        self._device = type("Dev", (), {"name": "simulated pad"})()

    def start(self) -> bool:
        return True

    def stop(self) -> None: ...

    def drain_events(self) -> list[tuple[float, str]]:
        return []                 # the dashboard's A/B/Y are the only button source

    def snapshot(self, t: float) -> PadSnapshot:
        if self.idle:
            return PadSnapshot(t=t, connected=True)
        e = time.monotonic() - self._t0
        return PadSnapshot(
            t=t,
            move_x=0.55 * math.sin(0.40 * e),
            move_y=0.45 * math.cos(0.31 * e),
            height=0.30 * math.sin(0.23 * e),
            yaw=0.35 * math.sin(0.53 * e),
            gripper=0.5 + 0.5 * math.sin(0.27 * e),
            connected=True,
            raw={},
        )


@dataclass
class _SimCap:
    """Mirrors cameras.CameraCapture closely enough for the rig and the dashboard."""

    label: str
    out_dir: Path
    cfg: dict
    stop_event: threading.Event
    index: int
    latest: Any = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    timestamps: list[float] = field(default_factory=list)
    frames_written: int = 0
    error: str = ""
    actual: dict = field(default_factory=dict)
    _thread: threading.Thread | None = None
    _t0: float = 0.0

    @property
    def fps(self) -> float:
        n = len(self.timestamps)
        if n < 2:
            return 0.0
        span = self.timestamps[-1] - self.timestamps[0]
        return (n - 1) / span if span > 0 else 0.0

    def start(self, t0: float) -> None:
        self._t0 = t0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def join(self, timeout: float = 10.0) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _run(self) -> None:
        import imageio.v2 as imageio

        w = int(get(self.cfg, "cameras.width", 640))
        h = int(get(self.cfg, "cameras.height", 480))
        fps = float(get(self.cfg, "cameras.fps", 30.0))
        self.actual = {"width": w, "height": h, "fps": fps}
        writer = imageio.get_writer(str(self.out_dir / f"{self.label}.mp4"), fps=fps,
                                    codec="libx264", quality=6, macro_block_size=1)
        period = 1.0 / fps
        nxt = time.monotonic()
        try:
            while not self.stop_event.is_set():
                now = time.monotonic()
                if now < nxt:
                    time.sleep(min(0.004, nxt - now))
                    continue
                nxt += period
                el = now - self._t0
                f = np.zeros((h, w, 3), np.uint8)
                f[:, :, self.index % 3] = int(35 + 165 * abs(math.sin(0.7 * el + self.index)))
                x = int(40 + (w - 120) * (0.5 + 0.5 * math.sin(0.8 * el + 0.4 * self.index)))
                f[h // 2 - 45:h // 2 + 45, x:x + 55] = 245
                with self.lock:
                    self.latest = f
                writer.append_data(f)
                self.timestamps.append(now - self._t0)
                self.frames_written += 1
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                writer.close()
            except Exception:
                pass


class SimulatedCameraRig:
    """Writes real mp4s and real timestamp files, so a simulated episode is a real
    run folder that summarises, plays back and exports like any other."""

    def __init__(self, cfg: dict[str, Any], out_dir: Path) -> None:
        self.cfg = cfg
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.stop_event = threading.Event()
        self.caps: list[_SimCap] = []

    def start(self, t0: float, blocking: bool = True) -> None:
        labels = list(get(self.cfg, "cameras.expected_labels", []) or [])
        stagger = float(get(self.cfg, "cameras.open_stagger_sec", 0.25))

        def work() -> None:
            for i, label in enumerate(labels):
                if self.stop_event.is_set():
                    return
                cc = _SimCap(label=label, out_dir=self.out_dir, cfg=self.cfg,
                             stop_event=self.stop_event, index=i)
                self.caps.append(cc)
                cc.start(t0)
                time.sleep(stagger)     # mirror the real USB stagger

        if blocking:
            work()
        else:
            threading.Thread(target=work, daemon=True).start()

    def stop(self) -> None:
        self.stop_event.set()
        for c in self.caps:
            c.join()

    def write_timestamps(self) -> None:
        import json

        from .cameras import _count_frames

        for c in self.caps:
            (self.out_dir / f"{c.label}_frame_times.json").write_text(json.dumps({
                "label": c.label, "device": "simulated", "frames": c.frames_written,
                "measured_fps": round(c.fps, 3), "error": c.error, "actual": c.actual,
                "frames_in_mp4": _count_frames(self.out_dir / f"{c.label}.mp4"),
                "t": [round(t, 6) for t in c.timestamps],
            }))

    def status(self) -> list[dict[str, Any]]:
        return [{"label": c.label, "frames": c.frames_written,
                 "fps": round(c.fps, 1), "error": c.error} for c in self.caps]
