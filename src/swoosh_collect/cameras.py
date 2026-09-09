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
    latest: Any = None
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
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self.actual = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        }

        writer = None
        try:
            for _ in range(int(get(c, "cameras.warmup_frames", 25))):
                if self.stop_event.is_set():
                    return
                if not cap.read()[0]:
                    self.error = "warmup read failed"
                    return
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
                if not ok or frame is None:
                    break
                now = time.monotonic()
                with self.lock:
                    self.latest = frame
                # Append the timestamp only AFTER the write succeeds, so a failed
                # encode can never leave len(timestamps) != frames_written.
                writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                self.timestamps.append(now - self._t0)
                self.frames_written += 1
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
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

    def __init__(self, cfg: dict[str, Any], out_dir: Path) -> None:
        self.cfg = cfg
        self.out_dir = out_dir
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
                cc = CameraCapture(
                    label=cam["label"], device=cam["device"], cfg=self.cfg,
                    out_dir=self.out_dir, stop_event=self.stop_event,
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
                        "actual": c.actual,
                        # Decoded afterwards and compared: a single dropped frame would
                        # shift every later timestamp by one frame period in the export.
                        "frames_in_mp4": _count_frames(self.out_dir / f"{c.label}.mp4"),
                        # seconds since the run's t0, same clock as every other stream
                        "t": [round(t, 6) for t in c.timestamps],
                    }
                )
            )

    def status(self) -> list[dict[str, Any]]:
        return [
            {"label": c.label, "frames": c.frames_written,
             "fps": round(c.fps, 1), "error": c.error}
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
