"""Wired Xbox pad via evdev. No ROS, no pygame, no display needed.

Runs a daemon reader thread that keeps a snapshot of every axis and button. The
control loop samples the snapshot; it never blocks on input. Buttons are exposed as
EDGE events (a queue of presses) so a single A press starts exactly one episode even
though the loop ticks at 100 Hz.

The pad is matched by NAME, not by /dev/input/eventN, because the event number moves
between boots and replugs.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .config import get


@dataclass
class PadSnapshot:
    """Normalised pad state. Sticks in [-1,1], triggers in [0,1]."""

    t: float = 0.0
    move_x: float = 0.0      # world +X forward
    move_y: float = 0.0      # world +Y left
    height: float = 0.0      # world +Z up
    yaw: float = 0.0         # about world +Z
    gripper: float = 0.0     # 0 open .. 1 closed
    connected: bool = False
    raw: dict[str, float] = field(default_factory=dict)   # every axis, pre-shaping
    # axis -> AGE IN SECONDS of that axis's current value at snapshot time. Stored as
    # an age, not a timestamp: the kernel's stamps are absolute CLOCK_MONOTONIC while
    # every `t` in this project is relative to t_loop0, and mixing the two produced
    # nonsense. An age is meaningful without knowing either epoch.
    raw_age_s: dict[str, float] = field(default_factory=dict)
    # age of the FRESHEST driving axis, in seconds -- see _freshest_age
    input_age_s: float = float("nan")

    def as_row(self) -> dict[str, Any]:
        return {
            "t": self.t,
            "move_x": self.move_x,
            "move_y": self.move_y,
            "height": self.height,
            "yaw": self.yaw,
            "gripper": self.gripper,
            "connected": self.connected,
            "raw": self.raw,
            "raw_age_s": self.raw_age_s,
            "input_age_s": self.input_age_s,
        }


def _shape(value: float, deadzone: float, expo: float) -> float:
    """Deadzone then expo. Rescales so the live band still reaches +-1: without the
    rescale a 0.12 deadzone would cap the achievable rate at 0.88."""
    a = abs(value)
    if a <= deadzone:
        return 0.0
    scaled = (a - deadzone) / (1.0 - deadzone)
    return (1.0 if value > 0 else -1.0) * (scaled ** float(expo))


def _ages(now_abs: float, raw_t: dict) -> dict:
    """Per-axis age in seconds, from absolute kernel stamps."""
    import math

    return {k: round(now_abs - v, 6)
            for k, v in raw_t.items() if not math.isnan(v)}


def _freshest_age(now_abs: float, raw_t: dict, codes: list) -> float:
    """Age in seconds of the FRESHEST driving axis.

    Not the oldest: an axis nobody is touching reports nothing, so its last value is
    legitimately minutes old and swamped the statistic (an untouched trigger made this
    read 139 s). The freshest axis is the one that just moved, which is the latency
    that matters. Per-axis ages are in `raw_age_s` for anything finer.

    `now_abs` MUST be absolute time.monotonic() -- the kernel's stamps are absolute,
    while every `t` in a run is relative to t_loop0.
    """
    import math

    ages = [now_abs - raw_t[c] for c in codes
            if c and c in raw_t and not math.isnan(raw_t[c])]
    return min(ages) if ages else float("nan")


class XboxPad:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.name_match = str(get(cfg, "controller.name_match", "Xbox")).lower()
        self.axis_map = dict(get(cfg, "controller.axes", {}) or {})
        self.button_map = dict(get(cfg, "controller.buttons", {}) or {})
        self.invert = dict(get(cfg, "controller.invert", {}) or {})
        self.deadzone = float(get(cfg, "control.deadzone", 0.12))
        self.expo = float(get(cfg, "control.expo", 2.0))

        self._lock = threading.Lock()
        self._raw: dict[str, float] = {}       # ecode name -> normalised value
        self._absinfo: dict[str, tuple[float, float]] = {}
        self._events: list[tuple[float, str]] = []   # (t, logical button name)
        self._raw_t: dict[str, float] = {}   # axis -> kernel event time
        self.event_clock_monotonic = False
        # (t, ecode name, logical or None) -- every press, for logging only
        self._presses: list[tuple[float, str, str | None]] = []
        self._device: Any = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.connected = False
        # logical name -> ecode name, inverted for lookup
        self._button_by_ecode = {v: k for k, v in self.button_map.items()}

    # -- device --------------------------------------------------------------
    def find_device(self) -> Any:
        import evdev

        for path in evdev.list_devices():
            try:
                dev = evdev.InputDevice(path)
            except OSError:
                continue
            if self.name_match in dev.name.lower():
                return dev
            caps = dev.capabilities()
            # Fall back to "has sticks and face buttons" for pads with odd names.
            if evdev.ecodes.EV_ABS in caps and evdev.ecodes.EV_KEY in caps:
                abs_codes = {c for c, _ in caps[evdev.ecodes.EV_ABS]}
                if {evdev.ecodes.ABS_X, evdev.ecodes.ABS_RX}.issubset(abs_codes):
                    return dev
        return None

    def _use_monotonic_clock(self, dev: Any) -> bool:
        """Ask the kernel to stamp this device's events on CLOCK_MONOTONIC.

        By default evdev events carry CLOCK_REALTIME, which is not comparable to the
        time.monotonic() clock every other stream in this project uses -- so the pad's
        own event time was unusable and got discarded, leaving the action stamped with
        the loop tick and up to ~18 ms of unmeasured staleness. EVIOCSCLOCKID switches
        the device to the monotonic clock, after which event.timestamp() can be
        subtracted from our own t directly.
        """
        import fcntl
        import struct
        EVIOCSCLOCKID = (1 << 30) | (4 << 16) | (0x45 << 8) | 0xA0   # _IOW('E',0xa0,int)
        try:
            fcntl.ioctl(dev.fd, EVIOCSCLOCKID, struct.pack("i", 1))  # 1 = CLOCK_MONOTONIC
            return True
        except OSError:
            return False

    def start(self) -> bool:
        import evdev

        dev = self.find_device()
        if dev is None:
            return False
        self._device = dev
        self.event_clock_monotonic = self._use_monotonic_clock(dev)
        for code, info in dev.capabilities().get(evdev.ecodes.EV_ABS, []):
            name = evdev.ecodes.ABS[code]
            if isinstance(name, list):
                name = name[0]
            self._absinfo[name] = (float(info.min), float(info.max))
        self.connected = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        import evdev

        dev = self._device
        try:
            for event in dev.read_loop():
                if self._stop.is_set():
                    break
                now = time.monotonic()
                if event.type == evdev.ecodes.EV_ABS:
                    name = evdev.ecodes.ABS.get(event.code)
                    if isinstance(name, list):
                        name = name[0]
                    if name is None:
                        continue
                    lo, hi = self._absinfo.get(name, (-32768.0, 32767.0))
                    span = (hi - lo) or 1.0
                    if name in ("ABS_Z", "ABS_RZ"):        # triggers: 0..1
                        val = (float(event.value) - lo) / span
                    else:                                   # sticks/dpad: -1..1
                        val = 2.0 * (float(event.value) - lo) / span - 1.0
                    with self._lock:
                        self._raw[name] = val
                        # The kernel's own stamp for THIS value. Comparable to
                        # time.monotonic() once EVIOCSCLOCKID has been set; NaN if the
                        # kernel refused, so a consumer can tell the difference rather
                        # than silently trusting a realtime clock.
                        self._raw_t[name] = (
                            float(event.timestamp())
                            if self.event_clock_monotonic else float("nan"))
                elif event.type == evdev.ecodes.EV_KEY and event.value == 1:
                    names = evdev.ecodes.BTN.get(event.code) or evdev.ecodes.KEY.get(event.code)
                    # evdev returns a TUPLE of aliases for buttons that have several
                    # names (0x130 -> ('BTN_A','BTN_GAMEPAD','BTN_SOUTH')) and a bare
                    # str for the rest. Handling only `list` meant every face button
                    # was silently ignored and only BTN_START ever fired.
                    if isinstance(names, (list, tuple)):
                        candidates = list(names)
                    elif names:
                        candidates = [names]
                    else:
                        candidates = []
                    logical = None
                    for ec in candidates:
                        logical = self._button_by_ecode.get(ec)
                        if logical:
                            with self._lock:
                                self._events.append((now, logical))
                            break
                    # Every press is also reported for LOGGING, mapped or not. An
                    # unbound button used to produce nothing at all, so an operator
                    # pressing the wrong one got silence and no way to tell that from
                    # a dead pad. This queue never drives the arm.
                    with self._lock:
                        self._presses.append(
                            (now, candidates[0] if candidates else f"code_{event.code}",
                             logical)
                        )
        except OSError:
            # Pad unplugged. Mark disconnected; the control loop keeps the arm still.
            self.connected = False

    # -- sampling ------------------------------------------------------------
    def snapshot(self, t: float) -> PadSnapshot:
        with self._lock:
            raw = dict(self._raw)
            raw_t = dict(self._raw_t)
        def axis(logical: str) -> float:
            ec = self.axis_map.get(logical)
            v = raw.get(ec, 0.0) if ec else 0.0
            if self.invert.get(logical):
                v = -v
            return v

        if not self.connected:
            # The pad went away. Return neutral rather than the last values it sent --
            # otherwise the arm keeps driving on a stale deflection and the recorded
            # action says the operator was holding the stick when they were not.
            return PadSnapshot(t=t, connected=False, raw=raw,
                               raw_age_s=_ages(time.monotonic(), raw_t))

        return PadSnapshot(
            t=t,
            move_x=_shape(axis("move_x"), self.deadzone, self.expo),
            move_y=_shape(axis("move_y"), self.deadzone, self.expo),
            height=_shape(axis("height"), self.deadzone, self.expo),
            yaw=_shape(axis("yaw"), self.deadzone, self.expo),
            # Trigger: deadzone only, no expo -- proportional gripper should track the
            # finger linearly.
            # Deadzone on the trigger too: a pad resting at 0.02 with noise would
            # otherwise cross the gripper's min_command_delta every tick and flood the
            # slow modbus link, stalling the loop.
            gripper=(lambda g: 0.0 if g < 0.03 else g)(max(0.0, min(1.0, axis("gripper")))),
            connected=self.connected,
            raw=raw,
            raw_age_s=_ages(time.monotonic(), raw_t),
            # How stale the driving axes are, measured rather than assumed. The five
            # axes that move the arm; NaN if the kernel would not give us its clock.
            input_age_s=_freshest_age(time.monotonic(), raw_t, [
                self.axis_map.get(k) for k in
                ("move_x", "move_y", "height", "yaw", "gripper")]),
        )

    def drain_presses(self) -> list[tuple[float, str, str | None]]:
        """Pop every button press since the last call, as (t, ecode_name, logical).

        `logical` is None for a button with no binding. Logging only -- the control
        loop drives off drain_events().
        """
        with self._lock:
            out = self._presses
            self._presses = []
        return out

    def drain_events(self) -> list[tuple[float, str]]:
        """Pop all button presses since the last call. Edge-triggered, so one physical
        press yields exactly one event no matter the loop rate."""
        with self._lock:
            out = self._events
            self._events = []
        return out


def main() -> int:
    """`swoosh-pad`-style probe: print live pad state so the mapping can be checked."""
    from .config import load_config

    cfg = load_config()
    pad = XboxPad(cfg)
    if not pad.start():
        print("No gamepad found. Is it plugged in? Try: ls -l /dev/input/event*")
        return 1
    print(f"reading {pad._device.name!r} -- move sticks, press buttons, Ctrl-C to stop\n")
    try:
        while True:
            s = pad.snapshot(time.monotonic())
            for t, name in pad.drain_events():
                print(f"\n  BUTTON {name}")
            print(
                f"\r  x{s.move_x:+.2f} y{s.move_y:+.2f} z{s.height:+.2f} "
                f"yaw{s.yaw:+.2f} grip{s.gripper:.2f}   ",
                end="",
                flush=True,
            )
            time.sleep(0.05)
    except KeyboardInterrupt:
        print()
    finally:
        pad.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
