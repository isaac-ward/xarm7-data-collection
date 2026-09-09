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
    # the SAME 7 joints read with is_real=True (servo feedback) rather than the SDK
    # default (controller's planned position). Both are logged so the difference is
    # measurable from the data -- see read_state.
    joints_real_deg: list[float] = field(default_factory=list)
    # False when the controller's report stream has gone silent (E-stop, link drop),
    # in which case api.angles reads all-zeros and any pose is suspect
    report_alive: bool = True
    # when gripper_pos was measured, on the SAME clock as `t` (see read_state)
    gripper_pos_t: float = float("nan")
    # how old that reading was when this row was built, in seconds
    gripper_pos_age_s: float = float("nan")
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
            "joints_real_deg": self.joints_real_deg,
            "report_alive": self.report_alive,
            "pose_base_mm_deg": self.pose_base,
            "pose_world_xyz_mm": self.pose_world_xyz,
            "gripper_pos": self.gripper_pos,
            "gripper_pos_t": self.gripper_pos_t,
            "gripper_pos_age_s": self.gripper_pos_age_s,
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
        self._grip_pos_t: float = float("nan")
        self._grip_sent_t: float = float("nan")
        self._grip_sent_code: int = 0
        self._grip_sent_err: str = ""
        self._grip_suspended: bool = False
        self._clock_origin: float = 0.0   # set by set_clock_origin()
        self._grip_poll_period: float = 1.0 / max(
            1.0, float(get(self.cfg, "gripper.poll_hz", 20.0)))
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
        import time as _t
        _t.sleep(0.6)          # let the report stream publish before reading codes
        arm.clean_warn()
        arm.clean_error()
        # Error 1 LATCHES: it stays set after the button is released and only goes away
        # once cleared. So the test has to be "still 1 AFTER clearing", never "1 on
        # arrival" -- an earlier version refused before attempting the clear, which
        # made a released E-stop impossible to recover from without an edit.
        if int(arm.error_code or 0) == self.ESTOP_CODE:
            _t.sleep(0.6)
            arm.clean_error()
            _t.sleep(0.4)
        if int(arm.error_code or 0) == self.ESTOP_CODE:
            raise RuntimeError(
                f"xArm at {self.ip} still reports error 1 after clearing: the "
                f"EMERGENCY STOP is physically ENGAGED.\n"
                f"  Release/twist the E-stop button on the control box (check the "
                f"teach pendant too) -- it cannot be cleared from software while it "
                f"is held in.\n"
                f"  Then re-run."
            )
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
                    # Record what the SDK was actually handed and what it returned.
                    # Previously the recorder logged the value the control loop QUEUED,
                    # which the worker could coalesce or fail to send, with the return
                    # code discarded and exceptions swallowed -- so the "commands sent
                    # to the arm" stream was, for the gripper, a statement of intent.
                    try:
                        code = self.api.set_gripper_position(
                            int(round(want)), wait=False, auto_enable=True)
                        with self._grip_lock:
                            self._grip_sent = want
                            self._grip_sent_t = time.monotonic()
                            self._grip_sent_code = int(code) if code is not None else -1
                    except Exception as exc:
                        with self._grip_lock:
                            self._grip_sent_code = -2
                            self._grip_sent_err = f"{type(exc).__name__}: {exc}"
                now = time.monotonic()
                # Poll rate is a knob because it is a real trade: every gripper read is
                # two modbus round trips holding the SAME SDK lock the 100 Hz servo
                # command needs, so asking more often buys a faster gripper column at
                # the cost of loop jitter. Watch tick.jsonl dt after changing it.
                if now - last_poll > self._grip_poll_period:
                    last_poll = now
                    try:
                        code, pos = self.api.get_gripper_position()
                        if code == 0 and pos is not None:
                            with self._grip_lock:
                                self._grip_pos = float(pos)
                                # When this value was actually measured. The cached
                                # reading is copied into every 50 Hz state row, so
                                # without its own timestamp the column looks like a
                                # 50 Hz signal that happens to repeat -- indistinguish-
                                # able from the sample-and-hold that ruined the last
                                # dataset's observation.state.
                                self._grip_pos_t = now
                    except Exception:
                        pass
                time.sleep(0.005)

        self._grip_thread = threading.Thread(target=work, daemon=True)
        self._grip_thread.start()

    ESTOP_CODE = 1      # UFACTORY: emergency stop engaged

    def estop_engaged(self) -> bool:
        """Is the PHYSICAL emergency stop latched?

        Error 1 cannot be cleared in software -- clean_error returns code=1, the report
        stream stops publishing (api.angles reads all-zeros), and every command is
        refused. The button has to be released by hand first. Worth naming explicitly,
        because "code=1" on its own looks like any other SDK failure.
        """
        return int(self.api.error_code or 0) == self.ESTOP_CODE

    def clear_errors(self) -> tuple[int, int]:
        """Clear latched errors and warnings and get back into servo streaming.

        Returns the (error, warn) codes that WERE set, so the operator learns what
        happened rather than just seeing it go away. Clearing alone is not enough:
        a cleared error leaves the controller out of servo mode, where
        set_servo_cartesian is ignored while still returning 0.
        """
        err, warn = int(self.api.error_code or 0), int(self.api.warn_code or 0)
        self.api.clean_warn()
        self.api.clean_error()
        # Judge the E-stop only AFTER trying to clear -- error 1 latches past release.
        if int(self.api.error_code or 0) == self.ESTOP_CODE:
            time.sleep(0.5)
            self.api.clean_error()
            time.sleep(0.3)
            if int(self.api.error_code or 0) == self.ESTOP_CODE:
                raise RuntimeError(
                    "EMERGENCY STOP is still engaged after clearing -- release/twist "
                    "the button on the control box (and check the teach pendant); it "
                    "cannot be cleared from software while it is held in."
                )
        self.api.motion_enable(enable=True)
        self.start_streaming()
        return err, warn

    def joints_now(self, timeout_s: float = 3.0) -> list[float]:
        """Current joint angles, waited for rather than read blind.

        `api.angles` is published by the report thread and reads ALL ZEROS for roughly
        the first 100 ms after connect. All-zeros is not a harmless default here: J4=0
        fully extends the elbow into the base, so anything that reads joints straight
        after connecting and commands them drives the arm into itself. Anything feeding
        a joint read back into a command must come through here.
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            a = list(self.api.angles or [])
            if len(a) >= 7 and any(abs(float(v)) > 1e-6 for v in a):
                return [float(v) for v in a]
            time.sleep(0.02)
        raise RuntimeError(
            "joint angles never arrived from the report stream (still all-zeros after "
            f"{timeout_s}s). Refusing to report a pose that would extend the elbow "
            "into the base if commanded."
        )

    def go_home(self, announce: bool = True, countdown: int = 3) -> None:
        """Move to the configured home joint pose. MOVES THE ARM."""
        from .utils.safety import emit_motion_warning

        if announce and countdown > 0:
            emit_motion_warning(robot_name="SWOOSH", countdown=countdown)
        self.api.set_mode(0)
        self.api.set_state(0)
        self.api.set_servo_angle(
            angle=list(get(self.cfg, "arm.home_joints")),
            speed=float(get(self.cfg, "arm.home_speed_deg_s", 20.0)),
            mvacc=float(get(self.cfg, "arm.home_acc_deg_s2", 100.0)),
            wait=True,
            is_radian=False,
        )

    def start_streaming(self, timeout_s: float = 3.0) -> None:
        """Enter servo-cartesian streaming mode, and VERIFY it actually took.

        This used to be set_mode(1) / set_state(0) / sleep(0.1) and trust it. It is not
        trustworthy, and the failure is silent in the worst possible way:
        `set_servo_cartesian` in mode 0 is IGNORED by the controller but still returns
        0. The SDK knows -- xarm/x3/base.py::_check_code logs "The mode may be
        incorrect" and then `return 0`, with the line that would have returned
        MODE_IS_NOT_CORRECT commented out immediately below it.

        So the loop streams targets at 100 Hz, every code says success, the arm never
        moves, and the recorder writes actions for motion that did not happen. That is
        the same class of defect as the lego dataset's dead `action` column.

        `mode` comes from the report stream, so it lags a set_mode by tens of ms --
        hence poll rather than sleep-and-hope, and re-issue in case it was rejected
        because the controller was still settling out of the previous mode.
        """
        deadline = time.monotonic() + timeout_s
        attempt = 0
        while True:
            self.api.set_mode(1)
            self.api.set_state(0)
            attempt += 1
            # `mode` is published by the report thread, so immediately after a
            # set_mode it can still hold the PREVIOUS value. Reading it too early is
            # therefore unsafe in both directions: a stale 1 left over from before a
            # set_mode(0) would satisfy this check instantly and defeat the whole
            # point. Wait out at least one report period before believing anything.
            time.sleep(0.2)
            for _ in range(20):
                if self.api.mode == 1:
                    return
                time.sleep(0.02)
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"arm did not enter servo streaming mode after {attempt} attempts "
                    f"(mode={self.api.mode}, state={self.api.state}, "
                    f"err={self.api.error_code}). set_servo_cartesian would return 0 "
                    f"and be silently ignored, so refusing to stream."
                )

    def streaming_ok(self) -> bool:
        """Is the controller still in servo mode? A drift back to 0 silently deadens
        every servo command while the SDK keeps returning 0."""
        return self.api.mode == 1

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
        # RECORD BOTH. The SDK's default is_radian=False leaves is_real=False, which
        # routes to get_joint_pos() -- the controller's PLANNED position -- while
        # is_real=True routes to get_joint_states(num=1), the servo feedback. UFACTORY
        # added the distinction for a reason, and which one this controller returns is
        # not determinable from the docs. If we logged only the default and it turned
        # out to be the plan, observation.state would follow the command through a
        # physical stall (collision sensitivity is 0) and that is UNRECOVERABLE.
        # Logging both makes the question answerable from the dataset instead of
        # needing to be settled before collecting. Costs one extra round trip per
        # state read; watch tick.jsonl dt if the loop rate suffers.
        try:
            code, real = self.api.get_servo_angle(is_radian=False, is_real=True)
            if code == 0 and real is not None:
                st.joints_real_deg = [float(a) for a in real]
        except Exception:
            pass
        try:
            code, pose = self.api.get_position(is_radian=False)
            if code == 0 and pose is not None:
                st.pose_base = [float(p) for p in pose]
                st.pose_world_xyz = base_to_world(np.array(st.pose_base[:3])).tolist()
        except Exception:
            pass
        # Is the controller's report stream actually publishing? It goes silent on an
        # E-stop and the SDK then hands back all-zeros for the joints rather than
        # admitting it does not know -- and all-zeros is not neutral on this arm, it is
        # the elbow driven into its own base. Recording this per row means a dead window
        # is visible in the data instead of looking like a real pose.
        try:
            a = self.api.angles or []
            st.report_alive = bool(
                len(a) >= 7 and any(abs(float(v)) > 1e-6 for v in a))
        except Exception:
            st.report_alive = False
        with self._grip_lock:      # cached by the worker; never block the loop on modbus
            st.gripper_pos = self._grip_pos
            # EPOCH. The worker stamps with absolute time.monotonic(); every `t` in a
            # run is relative to t_loop0, and the arm does not know t_loop0. Writing
            # the absolute stamp put gripper_pos_t 61,615 s away from every other
            # timestamp in the file, so the export resampled the gripper against a
            # clock it shared nothing with. Convert to an AGE here -- valid because
            # both sides are absolute -- then express it on the row's own clock.
            gp_t = self._grip_pos_t
        if gp_t == gp_t:                                # not NaN
            # Exact, via the run's clock origin. Reconstructing it as
            # `t - (monotonic() - gp_t)` was on the right clock but jittered a couple
            # of ms, because `t` and monotonic() are read at slightly different
            # instants -- so a value held between 20 Hz polls was never bit-identical
            # and the held-fraction check read the rate as 50 Hz.
            st.gripper_pos_t = gp_t - self._clock_origin
            st.gripper_pos_age_s = t - st.gripper_pos_t
        else:
            st.gripper_pos_age_s = float("nan")
            st.gripper_pos_t = float("nan")
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

    def gripper_command_record(self) -> dict:
        """What the gripper worker last actually SENT, not what was queued.

        `sent_age_s` rather than a timestamp, for the same epoch reason as
        gripper_pos_t: the worker's clock is absolute and the recorder's is relative to
        t_loop0. An age is meaningful without knowing either epoch, and the recorder
        turns it back into a time on the row's own clock.
        """
        with self._grip_lock:
            t_sent = self._grip_sent_t
            rec = {
                "sent_position": (None if self._grip_sent is None
                                  else float(self._grip_sent)),
                "sent_code": int(self._grip_sent_code),
                "sent_error": self._grip_sent_err,
            }
        rec["sent_t"] = (t_sent - self._clock_origin) if t_sent == t_sent else None
        return rec

    def set_clock_origin(self, t0: float) -> None:
        """Tell the arm the run's t_loop0, so its own stamps land on the shared clock.

        Without it the arm only has absolute time.monotonic() and every timestamp it
        produced had to be reconstructed by the caller -- which is how gripper_pos_t
        ended up 61,615 s adrift, and later merely jittery.
        """
        self._clock_origin = float(t0)

    def gripper_suspend(self, on: bool) -> None:
        """Stop the control loop from overwriting someone else's gripper command.

        `_grip_want` is a ONE-SLOT mailbox and the control loop writes it every tick at
        100 Hz. Anything else that queues a position is therefore overwritten before the
        worker can send it -- which is exactly why the camera-latency check appeared to
        do nothing: it queued "closed", and 10 ms later the loop queued "open" again
        from a resting trigger.
        """
        self._grip_suspended = bool(on)

    def set_gripper(self, closure: float, force: bool = False) -> float | None:
        """Proportional gripper. `closure` in [0,1]: 0 fully open, 1 fully closed.

        Returns the commanded raw position, or None if nothing was sent. The gripper
        is a slow serial device, so a command is only issued when it differs from the
        last by more than `gripper.min_command_delta` -- flooding it stalls the loop.
        """
        if self._grip_suspended and not force:
            return None      # a sanity task owns the gripper right now
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
