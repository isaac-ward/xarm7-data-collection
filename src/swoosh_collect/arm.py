"""RIGHT xArm 7 driver: connect, home, stream Cartesian targets, read state.

The connect/enable ordering here is not arbitrary -- it is the sequence that
sisl/manipulation-mono arrived at against this exact hardware, and the comments say
why each step is where it is. Deviating from it produces controller error 31 on a
cold start.
"""

from __future__ import annotations

import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import get
from .frames import base_to_world, world_to_base

SDK_PORT = 502  # plain TCP probe before opening XArmAPI, so an unreachable arm fails
                # fast with a clear message instead of a 30 s SDK timeout


@dataclass
class ArmState:
    """One sample of everything the arm can tell us, on the caller's clock."""

    t: float = 0.0
    joints_deg: list[float] = field(default_factory=list)   # 7
    pose_base: list[float] = field(default_factory=list)    # [x,y,z,r,p,y] mm/deg
    pose_world_xyz: list[float] = field(default_factory=list)
    gripper_pos: float = float("nan")
    state: int = -1
    mode: int = -1
    error_code: int = 0
    warn_code: int = 0

    def as_row(self) -> dict[str, Any]:
        return {
            "t": self.t,
            "joints_deg": self.joints_deg,
            "pose_base_mm_deg": self.pose_base,
            "pose_world_xyz_mm": self.pose_world_xyz,
            "gripper_pos": self.gripper_pos,
            "state": self.state,
            "mode": self.mode,
            "error_code": self.error_code,
            "warn_code": self.warn_code,
        }


def probe_reachable(ip: str, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((ip, SDK_PORT), timeout=timeout):
            return True
    except OSError:
        return False


class RightArm:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.ip = str(get(cfg, "arm.ip", "192.168.1.199"))
        self.api: Any = None
        self._last_gripper_cmd: float | None = None
        # Gripper I/O is TWO modbus RTU round trips over RS-485 (5-15 ms each). On the
        # control thread that alone can blow the 10 ms budget at 100 Hz -- and it would
        # slow the loop ONLY while the operator squeezes, so the effective stick gain
        # would change with what they are doing. Worker + mailbox instead.
        self._grip_want: float | None = None
        self._grip_sent: float | None = None
        self._grip_pos: float = float("nan")
        self._grip_lock = threading.Lock()
        self._grip_stop = threading.Event()
        self._grip_thread: threading.Thread | None = None

    # -- lifecycle -----------------------------------------------------------
    def connect(self) -> None:
        if not probe_reachable(self.ip):
            raise ConnectionError(
                f"xArm at {self.ip}:{SDK_PORT} is not reachable.\n"
                f"  - is the controller powered and booted? (xArm Studio at "
                f"http://{self.ip} should load)\n"
                f"  - does this host have an address on that subnet? expected "
                f"{get(self.cfg, 'arm.host_ip_cidr')} on "
                f"{get(self.cfg, 'arm.nic')}"
            )
        from xarm.wrapper import XArmAPI  # lazy so --help works without the SDK

        arm = XArmAPI(self.ip, is_radian=False)
        arm.clean_warn()
        arm.clean_error()
        # Sensitivity BEFORE motion_enable: otherwise the controller's torque model
        # can trip error 31 ("Collision Caused Abnormal Current") the instant the
        # servos power on, because the gripper's mass isn't in the payload model.
        try:
            arm.set_collision_sensitivity(int(get(self.cfg, "arm.collision_sensitivity", 0)))
        except Exception:
            pass
        arm.motion_enable(enable=True)
        arm.set_mode(0)   # position mode, for the initial joint move
        arm.set_state(0)
        # motion_enable can re-latch a fault that was just cleared, so clean again
        # or the first move is silently refused.
        try:
            arm.clean_warn()
            arm.clean_error()
        except Exception:
            pass
        try:
            arm.set_gripper_mode(0)
            arm.set_gripper_enable(True)
            arm.set_gripper_speed(int(get(self.cfg, "gripper.speed", 2000)))
        except Exception:
            pass
        self.api = arm

    def start_gripper_worker(self) -> None:
        def work() -> None:
            last_poll = 0.0
            while not self._grip_stop.is_set():
                with self._grip_lock:
                    want = self._grip_want
                    self._grip_want = None
                if want is not None:
                    try:
                        self.api.set_gripper_position(int(round(want)), wait=False,
                                                      auto_enable=True)
                        with self._grip_lock:
                            self._grip_sent = want
                    except Exception:
                        pass
                now = time.monotonic()
                if now - last_poll > 0.1:      # position at <=10 Hz; it is a slow device
                    last_poll = now
                    try:
                        code, pos = self.api.get_gripper_position()
                        if code == 0 and pos is not None:
                            with self._grip_lock:
                                self._grip_pos = float(pos)
                    except Exception:
                        pass
                time.sleep(0.005)

        self._grip_thread = threading.Thread(target=work, daemon=True)
        self._grip_thread.start()

    def go_home(self, announce: bool = True) -> None:
        """Move to the configured home joint pose. MOVES THE ARM."""
        from .utils.safety import emit_motion_warning

        if announce:
            emit_motion_warning(robot_name="SWOOSH", countdown=3)
        self.api.set_mode(0)
        self.api.set_state(0)
        self.api.set_servo_angle(
            angle=list(get(self.cfg, "arm.home_joints")),
            speed=float(get(self.cfg, "arm.home_speed_deg_s", 20.0)),
            mvacc=float(get(self.cfg, "arm.home_acc_deg_s2", 100.0)),
            wait=True,
            is_radian=False,
        )

    def start_streaming(self) -> None:
        """Enter servo-cartesian streaming mode. Required before servo_pose()."""
        self.api.set_mode(1)
        self.api.set_state(0)
        time.sleep(0.1)  # let the mode change settle

    def shutdown(self) -> None:
        self._grip_stop.set()
        if self.api is None:
            return
        try:
            self.api.set_state(4)  # stop
        except Exception:
            pass
        try:
            self.api.disconnect()
        except Exception:
            pass

    # -- io ------------------------------------------------------------------
    def read_state(self, t: float) -> ArmState:
        st = ArmState(t=t)
        try:
            code, angles = self.api.get_servo_angle(is_radian=False)
            if code == 0 and angles is not None:
                st.joints_deg = [float(a) for a in angles]
        except Exception:
            pass
        try:
            code, pose = self.api.get_position(is_radian=False)
            if code == 0 and pose is not None:
                st.pose_base = [float(p) for p in pose]
                st.pose_world_xyz = base_to_world(np.array(st.pose_base[:3])).tolist()
        except Exception:
            pass
        with self._grip_lock:      # cached by the worker; never block the loop on modbus
            st.gripper_pos = self._grip_pos
        for attr, name in ((("state",), "state"), (("mode",), "mode")):
            try:
                setattr(st, name, int(getattr(self.api, attr[0], -1)))
            except Exception:
                pass
        try:
            ec, ew = self.api.get_err_warn_code(show=False)
            if ec == 0 and ew:
                st.error_code = int(ew[0])
                st.warn_code = int(ew[1]) if len(ew) > 1 else 0
        except Exception:
            pass
        return st

    def servo_pose(self, pose_base: list[float]) -> int | None:
        """Stream one Cartesian target in the ARM BASE frame. Returns the SDK code.

        Code 1 means the controller is latched in HAS_ERROR -- every subsequent
        command is refused until it's cleared. The caller decides what to do; this
        does not auto-recover, because a ~1 s recovery window with a live operator
        produces a startling jump.
        """
        try:
            return self.api.set_servo_cartesian(
                pose_base,
                speed=float(get(self.cfg, "arm.speed_mm_s", 200.0)),
                mvacc=float(get(self.cfg, "arm.acc_mm_s2", 2000.0)),
                is_radian=False,
            )
        except Exception:
            return None

    def set_gripper(self, closure: float) -> float | None:
        """Proportional gripper. `closure` in [0,1]: 0 fully open, 1 fully closed.

        Returns the commanded raw position, or None if nothing was sent. The gripper
        is a slow serial device, so a command is only issued when it differs from the
        last by more than `gripper.min_command_delta` -- flooding it stalls the loop.
        """
        op = float(get(self.cfg, "gripper.open_position", 850))
        cl = float(get(self.cfg, "gripper.closed_position", 0))
        closure = max(0.0, min(1.0, float(closure)))
        target = op + (cl - op) * closure
        if (
            self._last_gripper_cmd is not None
            and abs(target - self._last_gripper_cmd)
            < float(get(self.cfg, "gripper.min_command_delta", 15))
        ):
            return None
        self._last_gripper_cmd = target
        with self._grip_lock:
            self._grip_want = target       # the worker does the modbus round trips
        return target

    def current_pose_world(self) -> tuple[np.ndarray, list[float]]:
        """(world xyz, full base pose). Used to seed the target at startup."""
        code, pose = self.api.get_position(is_radian=False)
        if code != 0 or pose is None:
            raise RuntimeError(f"get_position failed with code {code}")
        pose = [float(p) for p in pose]
        return base_to_world(np.array(pose[:3])), pose

    def world_target_to_base(self, world_xyz: np.ndarray, rpy_deg: list[float]) -> list[float]:
        return [*world_to_base(world_xyz).tolist(), *rpy_deg]
