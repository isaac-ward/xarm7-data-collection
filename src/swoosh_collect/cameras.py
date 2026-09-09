"""Four Arducams: discovery pinned by USB port path, plus timestamped capture.

WHY USB PATH AND NOT /dev/videoN. Enumeration order is not stable across reboots or
replugs, so a labels-in-order scheme silently swaps two cameras and corrupts a dataset
in a way that is very hard to notice later. The USB port path
(/sys/class/video4linux/videoN/device -> .../usb1/1-3/1-3.2/...) is a property of which
socket the cable is in, so it survives both.

Capture writes an MP4 per camera AND a per-frame timestamp log on the shared monotonic
clock. The timestamps are what make the 30 Hz export honest -- without them there is no
way to know which robot state a frame belongs to.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import get, load_config, set_value

SYSFS = Path("/sys/class/video4linux")


def _node_index(p: Path) -> int:
    m = re.search(r"(\d+)$", p.name)
    return int(m.group(1)) if m else -1


def _usb_port_path(realpath: str) -> str:
    """Reduce a sysfs device path to the stable USB port chain, e.g. '1-3.2'.

    The full realpath contains the interface suffix (':1.0') and the whole PCI
    prefix, neither of which we want: the first changes per interface, the second is
    noise. What identifies the physical socket is the trailing usb port chain.
    """
    parts = [p for p in realpath.split("/") if re.fullmatch(r"\d+-[\d.]+", p)]
    return parts[-1] if parts else realpath


def discover(name_filter: str = "Arducam") -> list[dict[str, Any]]:
    """One entry per physical USB camera matching the filter, lowest node first.

    A UVC camera exposes several /dev/videoN nodes (capture + metadata); only the
    lowest-numbered one of each physical device streams, hence the dedupe.
    """
    if not SYSFS.is_dir():
        return []
    nf = (name_filter or "").lower()
    by_dev: dict[str, dict[str, Any]] = {}
    for entry in sorted(SYSFS.iterdir(), key=_node_index):
        name_file = entry / "name"
        if not name_file.is_file():
            continue
        sysfs_name = name_file.read_text(errors="ignore").strip()
        if nf and nf not in sysfs_name.lower():
            continue
        try:
            real = os.path.realpath(entry / "device")
        except OSError:
            continue
        port = _usb_port_path(real)
        if port in by_dev:
            continue
        by_dev[port] = {
            "device": f"/dev/{entry.name}",
            "sysfs_name": sysfs_name,
            "usb_port": port,
            "node_index": _node_index(entry),
        }
    return sorted(by_dev.values(), key=lambda d: d["node_index"])


def resolve_labelled(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Discovered cameras joined to their configured labels by USB port path.

    Raises if the assignment is missing or stale, rather than guessing -- guessing is
    how you get a dataset with the wrist camera labelled as the scene camera.
    """
    mapping = dict(get(cfg, "cameras.by_usb_path", {}) or {})
    found = discover(str(get(cfg, "cameras.name_filter", "Arducam")))
    if not mapping:
        raise RuntimeError(
            "cameras.by_usb_path is empty -- no camera has been named yet.\n"
            "Run:  swoosh-cameras --assign"
        )
    out = []
    for cam in found:
        label = mapping.get(cam["usb_port"])
        if label:
            out.append({**cam, "label": label})
    missing = set(mapping.values()) - {c["label"] for c in out}
    if missing:
        raise RuntimeError(
            f"cameras configured but not found on the bus: {sorted(missing)}\n"
            f"  present USB ports: {[c['usb_port'] for c in found]}\n"
            f"  configured ports:  {sorted(mapping)}\n"
            "Re-seat the cable or re-run: swoosh-cameras --assign"
        )
    # Two ports mapped to one label would start two capture threads writing the same
    # mp4 and the same *_frame_times.json -- interleaved frames from two physical
    # cameras under one name, which is unrecoverable after the fact.
    seen: dict[str, str] = {}
    for c in out:
        if c["label"] in seen:
            raise RuntimeError(
                f"label {c['label']!r} is mapped to two USB ports "
                f"({seen[c['label']]} and {c['usb_port']}). Both would write the same "
                f"mp4. Fix cameras.by_usb_path -- run: swoosh-cameras --assign"
            )
        seen[c["label"]] = c["usb_port"]
    order = list(get(cfg, "cameras.expected_labels", []) or [])
    out.sort(key=lambda c: order.index(c["label"]) if c["label"] in order else 99)
    return out


# -- capture -----------------------------------------------------------------
@dataclass
class CameraCapture:
    """One camera's capture thread. Writes an mp4 and records frame timestamps."""

    label: str
    device: str
    cfg: dict[str, Any]
    out_dir: Path
    stop_event: threading.Event
    # False = preview only: publish frames for the dashboard, write no mp4 and record
    # no timestamps. Lets the operator SEE the cameras before pressing A, which
    # previously was impossible because the rig only existed during a recording.
    record: bool = True
    # Flip vertically the instant the frame leaves the device, before latest/mp4/
    # timestamps -- so the preview, the recording and the export are all the same
    # single orientation and nothing downstream has to compensate.
    flip_vertical: bool = False
    latest: Any = None
    # A read failure used to `break` with nothing set, so a camera dying mid-run was
    # indistinguishable from a clean stop. The exporter then truncated the episode to
    # the death time and still reported success.
    stopped_early: bool = False
    stopped_at_s: float = -1.0
    frames_seen: int = 0        # read off the device, whether or not we are recording
    # "v4l2_buffer" (kernel capture time) or "post_read" (when read() returned).
    timestamp_source: str = "post_read"
    # median(post_read - kernel) in seconds: the exposure + transfer + decode bias that
    # post_read stamping silently carries.
    read_lag_s: float = -1.0
    _lags: list = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    timestamps: list[float] = field(default_factory=list)
    frames_written: int = 0
    error: str = ""
    actual: dict = field(default_factory=dict)
    _thread: threading.Thread | None = None
    _t0: float = 0.0

    def start(self, t0: float) -> None:
        self._t0 = t0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def join(self, timeout: float = 10.0) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    @property
    def fps(self) -> float:
        n = len(self.timestamps)
        if n < 2:
            return 0.0
        span = self.timestamps[-1] - self.timestamps[0]
        return (n - 1) / span if span > 0 else 0.0

    def _run(self) -> None:
        import cv2
        import imageio.v2 as imageio

        c = self.cfg
        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            self.error = f"could not open {self.device}"
            return
        cap.set(cv2.CAP_PROP_FOURCC,
                cv2.VideoWriter_fourcc(*str(get(c, "cameras.fourcc", "MJPG"))))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(get(c, "cameras.width", 640)))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(get(c, "cameras.height", 480)))
        cap.set(cv2.CAP_PROP_FPS, float(get(c, "cameras.fps", 30.0)))
        # Depth-1 queue: with the default 4-deep V4L2 queue a stalled encoder lets
        # frames pile up and the timestamp we take after read() lags by up to 4 frames
        # (133 ms) with nothing recording that it happened.
        buf_ok = False
        try:
            buf_ok = bool(cap.set(cv2.CAP_PROP_BUFFERSIZE, 1))
        except Exception:
            pass
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        self.actual = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS)),
            # The buffersize request's result was previously discarded. If the driver
            # ignored it the V4L2 queue stays 4 deep and every frame we stamp is up to
            # 4 frame periods (~133 ms) stale -- so whether it took has to be on record.
            "buffersize_set": buf_ok,
            "buffersize": int(cap.get(cv2.CAP_PROP_BUFFERSIZE)),
            "fourcc": "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)),
        }

        writer = None
        try:
            for _ in range(int(get(c, "cameras.warmup_frames", 25))):
                if self.stop_event.is_set():
                    return
                if not cap.read()[0]:
                    self.error = "warmup read failed"
                    return
            if self.record:
                path = self.out_dir / f"{self.label}.mp4"
                writer = imageio.get_writer(
                    str(path),
                    fps=float(get(c, "cameras.fps", 30.0)),
                    codec="libx264",
                    quality=int(get(c, "cameras.mp4_quality", 8)),
                    macro_block_size=1,
                )
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if ok and frame is not None and self.flip_vertical:
                    # -1 = BOTH axes = a 180 degree ROTATION, which is what an
                    # upside-down mount actually is.
                    # This was cv2.flip(frame, 0), a vertical MIRROR. That corrects
                    # up/down but reverses chirality: the recorded view disagreed with
                    # the other three cameras, and with move_y, about which way is
                    # left. Two cameras on a vertical baseline cannot disagree on
                    # azimuth, which is how it was caught -- the block stack sat
                    # right-of-centre in one gripper view and left-of-centre in the
                    # other.
                    frame = cv2.flip(frame, -1)
                if not ok or frame is None:
                    # Not a normal exit: stop_event is how a run ends. Getting here
                    # means the device went away (USB re-enumeration is the usual
                    # cause on this hub) or the driver failed a read.
                    self.stopped_early = True
                    self.stopped_at_s = (
                        self.timestamps[-1] if self.timestamps else 0.0)
                    self.error = self.error or (
                        f"read failed after {self.frames_written} frames "
                        f"at t={self.stopped_at_s:.2f}s -- camera stopped early")
                    break
                now = time.monotonic()
                # V4L2 stamps each buffer on CLOCK_MONOTONIC -- the same clock as
                # time.monotonic() -- at capture, before transfer and MJPG decode.
                # Stamping after read() folds all of that in as a bias. Use the kernel
                # value when it is sane, and record the measured difference either way.
                cap_t = now
                try:
                    pos_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))
                except Exception:
                    pos_ms = 0.0
                if pos_ms > 0.0:
                    kt = pos_ms / 1000.0
                    lag = now - kt
                    # Sane means: same epoch as our clock, and the kernel time is not
                    # in the future. A backend reporting a stream-relative position
                    # fails this and we keep post_read stamping.
                    if 0.0 <= lag < 0.5:
                        cap_t = kt
                        self.timestamp_source = "v4l2_buffer"
                        if len(self._lags) < 900:
                            self._lags.append(lag)
                            self.read_lag_s = float(sorted(self._lags)[len(self._lags) // 2])
                with self.lock:
                    self.latest = frame
                self.frames_seen += 1        # counted in preview too, unlike
                #                              frames_written, so camera health is
                #                              observable before a run starts
                if writer is None:
                    continue    # preview: nothing is written and nothing is stamped
                # Append the timestamp only AFTER the write succeeds, so a failed
                # encode can never leave len(timestamps) != frames_written.
                writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                self.timestamps.append(cap_t - self._t0)
                self.frames_written += 1
        except Exception as exc:
            # A thread dying by exception (writer/pipe failure) is just as much an
            # early stop as a failed read. The first version of this guard only set
            # stopped_early on the read path, so an exception still produced a
            # silently truncated episode that export accepted and validate passed.
            self.error = f"{type(exc).__name__}: {exc}"
            self.stopped_early = True
            self.stopped_at_s = self.timestamps[-1] if self.timestamps else 0.0
        finally:
            if writer is not None:
                try:
                    writer.close()
                except Exception:
                    pass
            cap.release()


def _count_frames(path: Path) -> int:
    """Decode-count an mp4. Used to reconcile against the timestamp list."""
    try:
        import imageio.v2 as imageio

        with imageio.get_reader(str(path)) as r:
            return sum(1 for _ in r)
    except Exception:
        return -1


class CameraRig:
    """All four cameras as one unit, sharing a clock origin and a stop event."""

    def __init__(self, cfg: dict[str, Any], out_dir: Path, record: bool = True) -> None:
        self.cfg = cfg
        self.out_dir = out_dir
        self.record = record
        self.stop_event = threading.Event()
        self.caps: list[CameraCapture] = []

    def start(self, t0: float, blocking: bool = True) -> None:
        """Open every camera. `t0` is the SHARED clock origin -- the same value every
        other stream is timestamped against.

        With blocking=False the staggered opens happen on a worker thread, because the
        stagger sleeps used to run inside the 100 Hz control loop and stalled the arm
        for ~1 s at every A press.
        """
        cams = resolve_labelled(self.cfg)   # raises here, on the caller's thread
        stagger = float(get(self.cfg, "cameras.open_stagger_sec", 0.25))

        def work() -> None:
            for cam in cams:
                if self.stop_event.is_set():
                    return
                # Keyed by USB PORT deliberately: a flip corrects how the camera is
                # BOLTED ON, so it must survive relabelling. Keying it by label meant
                # swapping two labels silently un-flipped one view and flipped the
                # other. Labels are still accepted so old configs keep working.
                _flip_cfg = set(get(self.cfg, "cameras.flip_vertical", []) or [])
                flip = (cam["usb_port"] in _flip_cfg) or (cam["label"] in _flip_cfg)
                cc = CameraCapture(
                    label=cam["label"], device=cam["device"], cfg=self.cfg,
                    out_dir=self.out_dir, stop_event=self.stop_event,
                    record=self.record, flip_vertical=flip,
                )
                self.caps.append(cc)
                cc.start(t0)
                # Stagger VIDIOC_STREAMON so four cameras don't saturate one USB hub
                # at the same instant.
                time.sleep(stagger)

        if blocking:
            work()
        else:
            threading.Thread(target=work, daemon=True).start()

    def stop(self) -> None:
        self.stop_event.set()
        for c in self.caps:
            c.join()

    def write_timestamps(self) -> None:
        for c in self.caps:
            (self.out_dir / f"{c.label}_frame_times.json").write_text(
                json.dumps(
                    {
                        "label": c.label,
                        "device": c.device,
                        "frames": c.frames_written,
                        "measured_fps": round(c.fps, 3),
                        "error": c.error,
                        "stopped_early": c.stopped_early,
                        "stopped_at_s": round(c.stopped_at_s, 3),
                        "timestamp_source": c.timestamp_source,
                        "read_lag_s": round(c.read_lag_s, 4),
                        # the frames in this mp4 are already flipped; do not flip again
                        "flipped_vertical": c.flip_vertical,
                        "actual": c.actual,
                        # Decoded afterwards and compared: a single dropped frame would
                        # shift every later timestamp by one frame period in the export.
                        "frames_in_mp4": _count_frames(self.out_dir / f"{c.label}.mp4"),
                        # seconds since the run's t0, same clock as every other stream
                        "t": [round(t, 6) for t in c.timestamps],
                    }
                )
            )

    def expected_missing(self) -> list[str]:
        """Configured labels that produced no capture thread at all.

        `start()` builds threads on a worker, so a device that could not be opened
        simply never appears in `caps` -- and every later consumer counts what IS
        there rather than what was expected.
        """
        want = set(get(self.cfg, "cameras.expected_labels", []) or [])
        mapped = set((get(self.cfg, "cameras.by_usb_path", {}) or {}).values())
        return sorted((want & mapped) - {c.label for c in self.caps})

    def status(self) -> list[dict[str, Any]]:
        return [
            {"label": c.label, "frames": c.frames_written,
             "seen": c.frames_seen,
             "fps": round(c.fps, 1), "error": c.error,
             "timestamp_source": c.timestamp_source,
             "read_lag_s": round(c.read_lag_s, 4),
             "flipped_vertical": c.flip_vertical,
             "stopped_early": c.stopped_early,
             "stopped_at_s": round(c.stopped_at_s, 3)}
            for c in self.caps
        ]


# -- CLI ---------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Discover Arducams and pin labels to USB port paths."
    )
    ap.add_argument("--assign", action="store_true",
                    help="interactively name each camera and save to the config")
    ap.add_argument("--preview", action="store_true",
                    help="with --assign, show a frame from each camera while naming")
    args = ap.parse_args()

    cfg = load_config()
    found = discover(str(get(cfg, "cameras.name_filter", "Arducam")))
    if not found:
        print(f"No cameras matching {get(cfg, 'cameras.name_filter')!r} found under {SYSFS}.")
        print("Plugged in? Try: v4l2-ctl --list-devices")
        return 1

    mapping = dict(get(cfg, "cameras.by_usb_path", {}) or {})
    print(f"{len(found)} camera(s):\n")
    print(f"  {'device':14s} {'usb port':12s} {'label':18s} name")
    for c in found:
        print(f"  {c['device']:14s} {c['usb_port']:12s} "
              f"{mapping.get(c['usb_port'], '-'):18s} {c['sysfs_name']}")

    if not args.assign:
        if not mapping:
            print("\nNothing assigned yet. Run with --assign to name them.")
        return 0

    expected = list(get(cfg, "cameras.expected_labels", []) or [])
    print(f"\nName each camera. Suggested: {', '.join(expected)}")
    print("Blank skips a camera.\n")
    new: dict[str, str] = {}
    for c in found:
        if args.preview:
            _show_one(c["device"], c["usb_port"])
        cur = mapping.get(c["usb_port"], "")
        prompt = f"  {c['device']} (usb {c['usb_port']})"
        prompt += f" [{cur}]: " if cur else ": "
        try:
            label = input(prompt).strip() or cur
        except EOFError:
            label = cur
        if label:
            new[c["usb_port"]] = label
    cfg["cameras"]["by_usb_path"] = new
    path = set_value("cameras.by_usb_path", new)
    print(f"\nsaved {len(new)} assignment(s) to {path}")
    for port, label in new.items():
        print(f"  {port:12s} -> {label}")
    return 0


def _show_one(device: str, port: str) -> None:
    """Grab and display one frame so the operator can tell which camera is which."""
    try:
        import cv2
    except ImportError:
        return
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    try:
        for _ in range(10):
            cap.read()
        ok, frame = cap.read()
        if ok and frame is not None:
            win = f"usb {port} -- press any key"
            cv2.imshow(win, frame)
            cv2.waitKey(0)
            cv2.destroyWindow(win)
    except Exception:
        pass
    finally:
        cap.release()


if __name__ == "__main__":
    raise SystemExit(main())
