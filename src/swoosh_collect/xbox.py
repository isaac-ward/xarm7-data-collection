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
        }


def _shape(value: float, deadzone: float, expo: float) -> float:
    """Deadzone then expo. Rescales so the live band still reaches +-1: without the
    rescale a 0.12 deadzone would cap the achievable rate at 0.88."""
    a = abs(value)
    if a <= deadzone:
        return 0.0
    scaled = (a - deadzone) / (1.0 - deadzone)
    return (1.0 if value > 0 else -1.0) * (scaled ** float(expo))


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

    def start(self) -> bool:
        import evdev

        dev = self.find_device()
        if dev is None:
            return False
        self._device = dev
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
                    for ec in candidates:
                        logical = self._button_by_ecode.get(ec)
                        if logical:
                            with self._lock:
                                self._events.append((now, logical))
                            break
        except OSError:
            # Pad unplugged. Mark disconnected; the control loop keeps the arm still.
            self.connected = False

    # -- sampling ------------------------------------------------------------
    def snapshot(self, t: float) -> PadSnapshot:
        with self._lock:
            raw = dict(self._raw)
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
            return PadSnapshot(t=t, connected=False, raw=raw)

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
        )

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
