"""Localhost dashboard. Stdlib only -- no Flask, no websockets, no build step.

Three transports, each chosen because the browser speaks it natively:

  GET /                       the page (one self-contained HTML file)
  GET /api/events             Server-Sent Events: controller + proprio + run status
  GET /api/campaign           campaign name and its runs, as JSON
  POST /api/campaign/new      create a campaign folder under campaigns/
  POST /api/open              xdg-open a campaign or run folder in the file manager
  GET /stream/<label>         MJPEG (multipart/x-mixed-replace) -- renders in a plain <img>
  GET /api/run/<name>         one run's raw streams, thinned, for client-side playback
  GET /media/<run>/<file>     the recorded mp4s and pngs

IT MUST NOT SLOW THE ARM DOWN. The 100 Hz servo loop only ever writes into a
`LiveState` under a short lock; every encode, copy and socket write happens on server
threads. Camera preview is throttled and downscaled, because JPEG-encoding four 640x480
frames at 30 Hz would cost more CPU than the control loop itself.
"""

from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from .campaign import Campaign
from .config import get


class SanityRunner:
    """Runs one pre-flight task at a time and buffers its output for the browser.

    Output is captured line by line into a ring buffer the page polls, so the embedded
    terminal shows progress live rather than only at the end. Only one task runs at a
    time -- these touch the arm, and two concurrent jogs would be a bad afternoon.
    """

    MAX_LINES = 4000

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.lines: list[str] = []
        self.running: str | None = None
        self.exit_code: int | None = None

    def log(self, text: str) -> None:
        with self._lock:
            for ln in str(text).rstrip("\n").split("\n"):
                self.lines.append(ln)
            if len(self.lines) > self.MAX_LINES:
                del self.lines[: len(self.lines) - self.MAX_LINES]

    def tail(self, since: int) -> tuple[list[str], int]:
        with self._lock:
            return self.lines[since:], len(self.lines)

    def clear(self) -> None:
        with self._lock:
            self.lines = []
            self.exit_code = None

    def start(self, name: str, argv: list[str]) -> bool:
        """Run a console command as a subprocess, streaming stdout+stderr."""
        import subprocess

        with self._lock:
            if self.running:
                return False
            self.running = name
            self.exit_code = None

        def work() -> None:
            self.log(f"$ {' '.join(argv)}")
            try:
                proc = subprocess.Popen(
                    argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                )
                assert proc.stdout is not None
                for line in proc.stdout:
                    self.log(line.rstrip("\n"))
                proc.wait()
                code = proc.returncode
            except FileNotFoundError:
                self.log(f"command not found: {argv[0]}")
                code = 127
            except Exception as exc:
                self.log(f"{type(exc).__name__}: {exc}")
                code = 1
            self.log(f"[exit {code}]")
            with self._lock:
                self.running = None
                self.exit_code = code

        threading.Thread(target=work, daemon=True).start()
        return True

    def start_fn(self, name: str, fn: Any) -> bool:
        """Run a python callable, capturing anything it prints."""
        import contextlib
        import io

        with self._lock:
            if self.running:
                return False
            self.running = name
            self.exit_code = None

        def work() -> None:
            buf = io.StringIO()
            code = 0
            try:
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    fn(self.log)
            except Exception as exc:
                import traceback
                self.log(f"{type(exc).__name__}: {exc}")
                self.log(traceback.format_exc())
                code = 1
            if buf.getvalue().strip():
                self.log(buf.getvalue())
            self.log(f"[exit {code}]")
            with self._lock:
                self.running = None
                self.exit_code = code

        threading.Thread(target=work, daemon=True).start()
        return True


class LiveState:
    """The single hand-off point between the control loop and the server."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.controller: dict[str, Any] = {}
        self.proprio: dict[str, Any] = {}
        self.status: dict[str, Any] = {}
        self.frames: dict[str, Any] = {}      # label -> latest BGR frame (numpy)
        self.seq = 0
        # Button presses injected from the dashboard. The control loop drains these
        # alongside the physical pad's, so an on-screen A behaves exactly like a real A.
        self._buttons: list[str] = []
        self.checks: list[dict] = []      # last pre-run check result, for the dashboard
        self.preview = False              # synthetic data, off by default
        self.workspace: dict | None = None   # live edits from the dashboard

    def publish(self, controller: dict, proprio: dict, status: dict) -> None:
        with self._lock:
            self.controller = controller
            self.proprio = proprio
            self.status = status
            self.seq += 1

    def put_frame(self, label: str, frame: Any) -> None:
        with self._lock:
            self.frames[label] = frame

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {"controller": dict(self.controller), "proprio": dict(self.proprio),
                    "status": dict(self.status), "checks": list(self.checks),
                    "seq": self.seq}

    def get_frame(self, label: str) -> Any:
        with self._lock:
            return self.frames.get(label)

    def set_workspace(self, box: dict) -> None:
        with self._lock:
            self.workspace = dict(box)

    def take_workspace(self) -> dict | None:
        with self._lock:
            b, self.workspace = self.workspace, None
            return b

    def toggle_preview(self) -> bool:
        with self._lock:
            self.preview = not self.preview
            return self.preview

    def set_checks(self, checks: list[dict]) -> None:
        with self._lock:
            self.checks = list(checks)

    def press(self, button: str) -> None:
        with self._lock:
            self._buttons.append(button)

    def drain_buttons(self) -> list[str]:
        with self._lock:
            out, self._buttons = self._buttons, []
            return out


def _thin(rows: list[dict], hz: float, dur_hint: float = 0.0) -> list[dict]:
    """Downsample a raw stream to ~hz for playback, so a 5-minute run doesn't ship
    30k rows to the browser."""
    if not rows:
        return []
    out, next_t = [], -1e9
    step = 1.0 / max(hz, 1.0)
    for r in rows:
        t = float(r.get("t", 0.0))
        if t >= next_t:
            out.append(r)
            next_t = t + step
    return out


class Handler(BaseHTTPRequestHandler):
    server_version = "swoosh/1.0"
    live: LiveState
    campaign_ref: dict[str, Any]
    cfg: dict[str, Any]
    sanity: SanityRunner
    arm_ref: dict[str, Any]

    def log_message(self, *a: Any) -> None:  # keep the console clean for the operator
        pass

    # -- helpers -------------------------------------------------------------
    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None,
              gzip_ok: bool = False) -> None:
        extra = dict(extra or {})
        # The 3D view pulls ~4.4 MB (three.module.js unminified + 3.1 MB of STL), which
        # is most of the wait before the arm appears. These compress well and the
        # stdlib server does nothing automatically.
        if gzip_ok and len(body) > 1400 and "gzip" in (
                self.headers.get("Accept-Encoding") or "").lower():
            import gzip as _gzip
            body = _gzip.compress(body, 6)
            extra["Content-Encoding"] = "gzip"
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        # A caller-supplied Cache-Control has to WIN. This sent no-store
        # unconditionally and THEN appended the caller's header, so /vendor/'s
        # max-age was dead on arrival and the whole 4.4 MB was re-fetched on every
        # page load, not just the first.
        if "Cache-Control" not in extra:
            self.send_header("Cache-Control", "no-store")
        for k, v in extra.items():
            self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, obj: Any, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    @property
    def campaign(self) -> Campaign:
        return self.campaign_ref["campaign"]

    # -- routes --------------------------------------------------------------
    def do_GET(self) -> None:
        path = unquote(urlparse(self.path).path)
        if path == "/":
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif path == "/api/campaign":
            self._json(self._campaign_json())
        elif path == "/api/workspace":
            self._json({"box": get(self.cfg, "control.workspace_box_mm", {})})
        elif path == "/api/campaign/list":
            from .campaign import Campaign as _C
            cur = self.campaign.path.name
            self._json({"current": cur, "campaigns": [
                {"slug": c.path.name, "name": c.meta.get("name", c.path.name),
                 "runs": len(c.runs()), "active": c.path.name == cur}
                for c in _C.list_all(self.cfg)]})
        elif path == "/api/root":
            from .campaign import campaigns_root, host_view
            # host_view: the operator reads this on the HOST, where /workspace/... does
            # not exist and cannot be pasted anywhere useful.
            self._json({"configured": str(get(self.cfg, "recording.campaigns_dir", "campaigns")),
                        "resolved": host_view(campaigns_root(self.cfg))})
        elif path == "/api/events":
            self._events()
        elif path.startswith("/snapshot/"):
            self._snapshot(unquote(path[len("/snapshot/"):].split("?")[0]))
        elif path.startswith("/stream/"):
            self._mjpeg(path.split("/", 2)[2])
        elif path.startswith("/api/run/"):
            self._run_json(path.split("/", 3)[3])
        elif path.startswith("/vendor/"):
            self._vendor_file(path[len("/vendor/"):])
        elif path.startswith("/media/"):
            self._media(path[len("/media/"):])
        elif path == "/api/sanity/log":
            from urllib.parse import parse_qs
            q = parse_qs(urlparse(self.path).query)
            since = int((q.get("since") or ["0"])[0])
            lines, total = self.sanity.tail(since)
            self._json({"lines": lines, "total": total,
                        "running": self.sanity.running,
                        "exit_code": self.sanity.exit_code})
        elif path == "/api/sanity/cameras":
            from .cameras import discover
            found = discover(str(get(self.cfg, "cameras.name_filter", "Arducam")))
            self._json({"found": found,
                        "assigned": dict(get(self.cfg, "cameras.by_usb_path", {}) or {}),
                        "expected": list(get(self.cfg, "cameras.expected_labels", []) or [])})
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        path = unquote(urlparse(self.path).path)
        n = int(self.headers.get("Content-Length", 0) or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            body = {}
        if path == "/api/campaign/new":
            try:
                name = str(body.get("name", "")).strip()
                if not name:
                    raise ValueError("a campaign needs a name")
                self.campaign_ref["campaign"] = Campaign.create(name, self.cfg)
                self._json({"ok": True, "campaign": self._campaign_json()})
            except Exception as exc:
                self._json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 400)
        elif path == "/api/run/delete":
            try:
                import shutil
                name = str(body.get("run", "")).strip()
                run_dir = (self.campaign.path / name).resolve()
                base = self.campaign.path.resolve()
                # Never let a crafted name escape the campaign, and never delete a run
                # that is still being written -- deleting mid-recording would leave the
                # collector writing into a hole.
                if not name or not str(run_dir).startswith(str(base) + "/"):
                    raise ValueError("not a run in this campaign")
                if not run_dir.is_dir():
                    raise FileNotFoundError(name)
                run = next((r for r in self.campaign.runs()
                            if r.path.name == name), None)
                if run is not None and run.status != "ready":
                    raise ValueError(f"run is {run.status} -- stop it and let it "
                                     f"finish processing before deleting")
                shutil.rmtree(run_dir)
                self.sanity.log(f"[dash] deleted run {name}")
                self._json({"ok": True, "campaign": self._campaign_json()})
            except Exception as exc:
                self._json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 400)
        elif path == "/api/campaign/load":
            try:
                slug = str(body.get("slug", "")).strip()
                if not slug:
                    raise ValueError("pick a campaign")
                self.campaign_ref["campaign"] = Campaign.open(slug, self.cfg)
                self._json({"ok": True, "campaign": self._campaign_json()})
            except Exception as exc:
                self._json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 400)
        elif path == "/api/open":
            self._open_folder(str(body.get("run", "")).strip())
        elif path == "/api/sanity/start":
            self._sanity_start(body)
        elif path == "/api/sanity/clear":
            self.sanity.clear()
            self._json({"ok": True})
        elif path == "/api/sanity/cameras":
            self._save_cameras(body)
        elif path == "/api/workspace":
            self._set_workspace(body)
        elif path == "/api/root":
            self._set_root(body)
        elif path == "/api/button":
            name = str(body.get("button", ""))
            if name not in {"start_episode", "stop_episode", "rehome", "quit",
                            "clear_errors"}:
                self._json({"ok": False, "error": f"unknown button {name!r}"}, 400)
            else:
                self.live.press(name)
                self._json({"ok": True})
        else:
            self._send(404, b"not found", "text/plain")

    # These open a SECOND XArmAPI session and run clean_error / set_mode(0) /
    # set_state(4), or open the cameras. Doing that during a recording freezes the arm
    # while set_servo_cartesian keeps returning 0, so the action column goes on moving
    # against a stationary arm -- and the cameras cannot be opened twice at all.
    # SanityRunner only ever stopped two sanity jobs overlapping, never this.
    _EXCLUSIVE = {"arm", "frame", "home", "gripper", "cameras", "fkcheck", "reach",
                  "camlatency"}

    def _sanity_start(self, body: dict) -> None:
        import sys

        action = str(body.get("action", ""))
        if action in self._EXCLUSIVE and bool(self.live.status.get("recording")):
            self._json({"ok": False, "error":
                        f"'{action}' takes over the arm and cameras -- stop the run "
                        f"with B first. Running it now would freeze the arm mid-episode "
                        f"while the recorded action kept moving."}, 409)
            return
        py = sys.executable
        jobs: dict[str, list[str]] = {
            "arm":        [py, "-m", "swoosh_collect.sanity"],
            "frame":      [py, "-m", "swoosh_collect.sanity", "--verify-frame"],
            "home":       [py, "-m", "swoosh_collect.sanity", "--home"],
            "gripper":    [py, "-m", "swoosh_collect.sanity", "--gripper"],
            "cameras":    [py, "-m", "swoosh_collect.cameras"],
        }
        if action == "lag":
            ok = self.sanity.start_fn("lag", self._lag_check)
        elif action == "fkcheck":
            ok = self.sanity.start_fn("fkcheck", self._fk_check)
        elif action == "reach":
            ok = self.sanity.start_fn("reach", self._reach_check)
        elif action == "camlatency":
            ok = self.sanity.start_fn("camlatency", self._camera_latency)
        elif action == "validate":
            ok = self.sanity.start(action, [py, "-m", "swoosh_collect.validate",
                                            "--campaign", self.campaign.path.name])
        elif action == "preview":
            on = self.live.toggle_preview()
            self.sanity.log(f"preview mode {'ON - synthetic data, no hardware' if on else 'OFF'}")
            self._json({"ok": True, "preview": on})
            return
        elif action in jobs:
            ok = self.sanity.start(action, jobs[action])
        else:
            self._json({"ok": False, "error": f"unknown action {action!r}"}, 400)
            return
        self._json({"ok": ok,
                    "error": None if ok else f"{self.sanity.running} is already running"})

    def _lag_check(self, log: Any) -> None:
        """Does the commanded pose predict the measured pose one servo lag later?

        This is the check that lego_assemblies did not have. A correct action stream
        produces a U-shaped error curve with a clear minimum at the servo lag; a flat
        curve means the action is not aligned to the observation it caused.
        """
        import numpy as np

        from .summarize import _read_jsonl

        runs = [r for r in self.campaign.runs()
                if (r.path / "raw" / "commanded.jsonl").is_file()]
        if not runs:
            log("no runs with data yet -- record one first (A then B)")
            return
        r = runs[-1]
        log(f"checking {r.path.name}")
        cmd = _read_jsonl(r.path / "raw" / "commanded.jsonl")
        stt = _read_jsonl(r.path / "raw" / "arm_state.jsonl")
        if len(cmd) < 50 or len(stt) < 50:
            log(f"not enough rows (commanded={len(cmd)}, arm_state={len(stt)})")
            return
        ct = np.array([x["t"] for x in cmd])
        cx = np.array([x["target_world_xyz_mm"] for x in cmd], dtype=float)
        st = np.array([x["t"] for x in stt])
        sx = np.array([x.get("pose_world_xyz_mm") or [np.nan] * 3 for x in stt], dtype=float)
        log("")
        log("  lag (ms)   median |commanded(t) - measured(t+lag)|  (mm)")
        best, best_lag = 1e18, None
        for lag_ms in range(0, 401, 25):
            lag = lag_ms / 1000.0
            idx = np.searchsorted(st, ct + lag)
            idx = np.clip(idx, 0, len(st) - 1)
            d = np.linalg.norm(cx - sx[idx], axis=1)
            d = d[np.isfinite(d)]
            if not len(d):
                continue
            med = float(np.median(d))
            bar = "#" * min(46, int(med / 2))
            log(f"   {lag_ms:5d}      {med:8.2f}  {bar}")
            if med < best:
                best, best_lag = med, lag_ms
        log("")
        from .provenance import save_measurement

        if best_lag is not None:
            save_measurement("servo_lag_ms", {"lag_ms": best_lag, "residual_mm": best,
                                              "run": r.path.name})
            log(f"saved servo_lag_ms = {best_lag} into campaigns/.measurements.json")
        if best_lag is None:
            log("no usable samples")
        elif best_lag in (0, 400):
            log(f"WARNING: minimum at the EDGE of the sweep ({best_lag} ms, {best:.2f} mm).")
            log("A correct action stream gives a U with the minimum INSIDE the range.")
            log("A flat or edge-pinned curve is the lego_assemblies misalignment signature.")
        else:
            log(f"OK: minimum {best:.2f} mm at {best_lag} ms -- U-shaped, "
                f"minimum inside the sweep.")
            log("This is the check lego_assemblies lacked; it is what catches a")
            log("time-misaligned action before the data is used.")

    def _fk_check(self, log: Any) -> None:
        """Is the kinematic chain behind the 3D view actually right?

        Compares FK(joint angles) with the TCP pose the CONTROLLER reports for the same
        instant. They should agree to within the TCP offset (the controller reports the
        tool point; our FK stops at the flange). A large or pose-dependent disagreement
        means the table is wrong and the 3D arm is decorative fiction.
        """
        import numpy as np

        from .kinematics import fk_pose_mm

        arm = self.arm_ref.get("arm")
        if arm is None:
            log("no arm in this process -- run this from swoosh-collect")
            return
        if getattr(arm, "is_simulated", False):
            log("SIMULATED arm: its joint angles are smooth wobble, not an IK solution")
            log("for its pose, so FK and the reported pose are unrelated BY DESIGN.")
            log("This check only means something against the real robot.")
            return
        st = arm.read_state(0.0)
        if not st.joints_deg or not st.pose_base:
            log("arm did not report joints and pose")
            return
        fk = fk_pose_mm(st.joints_deg)
        rep = np.asarray(st.pose_base[:3], dtype=float)
        d = rep - fk
        log(f"  joints  {[round(v, 1) for v in st.joints_deg]}")
        log(f"  FK flange   {[round(float(v), 1) for v in fk]} mm")
        log(f"  controller  {[round(float(v), 1) for v in rep]} mm")
        log(f"  difference  {[round(float(v), 1) for v in d]} mm "
            f"(norm {float(np.linalg.norm(d)):.1f})")
        try:
            code, off = arm.api.get_tcp_offset()
            log(f"  controller TCP offset: {off if code == 0 else 'unavailable'}")
            log("  -> the difference SHOULD be about the TCP offset, since our FK stops")
            log("     at the flange and the controller reports the tool point.")
        except Exception:
            pass
        n = float(np.linalg.norm(d))
        if n < 200:
            log(f"OK: FK and the controller agree to {n:.0f} mm -- plausible for a "
                f"flange-vs-tool difference. The chain looks right.")
        else:
            log(f"WARNING: {n:.0f} mm apart. That is too large to be a TCP offset; the "
                f"chain in kinematics.py / armfk.js is probably wrong, so the 3D "
                f"arm's pose is not trustworthy.")

    def _reach_check(self, log: Any) -> None:
        """Are all eight corners of the safety box actually reachable?

        A box corner outside the arm's workspace lets the target integrate somewhere
        the arm can never follow: the arm freezes and the recorded action does nothing.
        """
        import itertools

        arm = self.arm_ref.get("arm")
        if arm is None:
            log("no arm in this process -- run this from swoosh-collect")
            return
        box = get(self.cfg, "control.workspace_box_mm", {})
        from .frames import world_to_base
        import numpy as np

        bad = 0
        for cx, cy, cz in itertools.product(box["x"], box["y"], box["z"]):
            base = world_to_base(np.array([cx, cy, cz], dtype=float))
            pose = [*base.tolist(), 180.0, 0.0, 0.0]
            try:
                code, _ = arm.api.get_inverse_kinematics(pose, input_is_radian=False,
                                                         return_is_radian=False)
            except Exception as exc:
                log(f"  IK call failed: {exc}")
                return
            ok = (code == 0)
            bad += not ok
            log(f"  world ({cx:6.0f},{cy:6.0f},{cz:6.0f}) -> "
                f"{'reachable' if ok else f'UNREACHABLE (code {code})'}")
        if bad:
            log(f"{bad}/8 corners unreachable. Shrink the box in the 3D scene panel, or "
                f"the target can run away where the arm cannot follow.")
        else:
            log("all 8 corners reachable")

    def _camera_latency(self, log: Any) -> None:
        """Measure ACTION -> PIXEL latency by commanding the gripper and watching for it.

        Closes the gripper, then watches a wrist camera for the first frame that differs
        materially from the pre-command baseline. The gap is exposure + USB + decode +
        the gripper's own travel. Without this number nobody knows how stale the frames
        a world model conditions on actually are.
        """
        import time

        import numpy as np

        from .provenance import save_measurement

        wrist = [l for l in (get(self.cfg, "cameras.expected_labels", []) or [])
                 if "gripper" in l]
        if not wrist:
            log("no wrist camera configured")
            return
        label = wrist[0]
        if self.live.get_frame(label) is None:
            log(f"no live frames from {label}. Start `swoosh-collect` so the cameras "
                f"are streaming, then press this again.")
            return
        arm = self.arm_ref.get("arm")
        if arm is None:
            log("no arm connected in this process; run this from swoosh-collect")
            return

        log(f"watching {label}; commanding the gripper closed")
        # Take the gripper off the control loop first, or the loop's next tick queues
        # the resting trigger value straight over the top of our command.
        arm.gripper_suspend(True)
        hit = None
        try:
            arm.set_gripper(0.0, force=True)     # start from fully OPEN, known state
            time.sleep(1.2)                      # let it finish travelling
            base = self.live.get_frame(label).astype("float32")
            arm.set_gripper(1.0, force=True)     # now CLOSE, and watch for it
            t0 = time.monotonic()
            while time.monotonic() - t0 < 3.0:
                f = self.live.get_frame(label)
                if f is not None:
                    d = float(np.abs(f.astype("float32") - base).mean())
                    if d > 6.0:
                        hit = time.monotonic() - t0
                        break
                time.sleep(0.005)
            time.sleep(0.4)
            arm.set_gripper(0.0, force=True)     # reopen, back to a known state
            time.sleep(0.8)
        finally:
            arm.gripper_suspend(False)           # hand it back to the operator
        if hit is None:
            log("no visible change within 3 s -- is the gripper in view?")
            return
        log(f"first visible change {hit*1000:.0f} ms after the command")
        log("(this is exposure + USB + decode + gripper travel, so it is an UPPER bound")
        log(" on pure camera latency, and the right number for action->pixel staleness)")
        save_measurement("camera_latency_s", {"seconds": hit, "camera": label})
        log("saved camera_latency_s into campaigns/.measurements.json")

    def _save_cameras(self, body: dict) -> None:
        from .config import set_value

        mapping = body.get("mapping")
        if not isinstance(mapping, dict) or not mapping:
            self._json({"ok": False, "error": "no mapping given"}, 400)
            return
        self.cfg.setdefault("cameras", {})["by_usb_path"] = {
            str(k): str(v) for k, v in mapping.items() if str(v).strip()
        }
        try:
            path = set_value("cameras.by_usb_path",
                             self.cfg["cameras"]["by_usb_path"])
            self.sanity.log(f"saved {len(mapping)} camera assignment(s) to {path}")
            for k, v in self.cfg["cameras"]["by_usb_path"].items():
                self.sanity.log(f"  {k:12s} -> {v}")
            self._json({"ok": True})
        except Exception as exc:
            self._json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 500)

    def _set_workspace(self, body: dict) -> None:
        """Edit the safety box live. Clamping means the commanded pose PRESSES AGAINST
        the wall rather than halting -- motion along the other axes continues."""
        from .config import set_value

        box = body.get("box") or {}
        try:
            clean = {k: [float(box[k][0]), float(box[k][1])] for k in ("x", "y", "z")}
            for k, (lo, hi) in clean.items():
                if hi <= lo:
                    raise ValueError(f"{k}: max must exceed min")
            self.cfg.setdefault("control", {})["workspace_box_mm"] = clean
            set_value("control.workspace_box_mm", clean)
            self.live.set_workspace(clean)
            self._json({"ok": True, "box": clean})
        except Exception as exc:
            self._json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 400)

    def _set_root(self, body: dict) -> None:
        """Point campaigns/ somewhere else -- an external drive, say. A relative path
        resolves against the repo root; an absolute one is used as-is."""
        from pathlib import Path as _P

        from .campaign import campaigns_root
        from .config import set_value

        raw = str(body.get("path", "")).strip()
        if not raw:
            self._json({"ok": False, "error": "no path given"}, 400)
            return
        try:
            self.cfg.setdefault("recording", {})["campaigns_dir"] = raw
            root = campaigns_root(self.cfg)
            root.mkdir(parents=True, exist_ok=True)
            if not os.access(root, os.W_OK):
                raise PermissionError(f"{root} is not writable")
            set_value("recording.campaigns_dir", raw)
            self.sanity.log(f"campaign root -> {root}")
            self._json({"ok": True, "configured": raw, "resolved": str(root)})
        except Exception as exc:
            self._json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 400)

    def _open_folder(self, run: str) -> None:
        """Open a folder in the desktop file manager, or hand back the host path.

        The collector normally runs INSIDE the container, where there is no desktop
        session and no file manager: xdg-open's Popen succeeded, so this reported
        ok=true, and nothing whatsoever happened. The path it would have opened was a
        container path (/workspace/...) that does not exist on the host either.

        So: only attempt xdg-open when not containerised, and otherwise return the
        HOST-side path for the operator to paste. Saying "here is the path, I cannot
        open it from in here" beats silently doing nothing.
        """
        import subprocess
        from pathlib import Path as _P

        from .campaign import host_view

        base = self.campaign.path.resolve()
        target = (base / run).resolve() if run else base
        if not str(target).startswith(str(base)) or not target.is_dir():
            self._json({"ok": False, "error": "no such folder"}, 404)
            return
        host_path = host_view(target)
        if _P("/.dockerenv").exists():
            self._json({"ok": False, "containerised": True, "path": host_path,
                        "error": "The collector is running inside Docker, which has "
                                 "no desktop session -- it cannot open your file "
                                 "manager. The folder on this machine is:"})
            return
        try:
            subprocess.Popen(["xdg-open", str(target)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._json({"ok": True, "path": host_path})
        except Exception as exc:
            self._json({"ok": False, "error": f"{type(exc).__name__}: {exc}",
                        "path": host_path}, 500)

    # -- payloads ------------------------------------------------------------
    def _campaign_json(self) -> dict[str, Any]:
        c = self.campaign
        runs = []
        for r in c.runs():
            runs.append({
                "name": r.path.name,
                "index": r.index,
                "duration_s": round(r.duration_s, 1),
                "stopped_by": r.stopped_by,
                "complete": r.complete,
                "status": r.status,          # recording | processing | ready
                "timestamp": r.timestamp,
                "error": r.meta.get("processing_error"),
            })
        return {"name": c.name, "slug": c.path.name, "path": str(c.path),
                "runs": runs, "complete": sum(r["complete"] for r in runs)}

    def _events(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        last = -1
        try:
            while True:
                snap = self.live.snapshot()
                if snap["seq"] != last:
                    last = snap["seq"]
                    self.wfile.write(f"data: {json.dumps(snap)}\n\n".encode())
                    self.wfile.flush()
                time.sleep(1.0 / 15.0)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _snapshot(self, label: str) -> None:
        """One JPEG, then the connection closes.

        The four previews used to be long-lived MJPEG streams, one open connection
        each. With the SSE stream that is five of a browser's six-per-origin budget,
        on an HTTP/1.0 server with no keep-alive, while ~20 vendor files and the log
        poll competed for what was left -- so the last two panes (the gripper pair,
        last in expected_labels) never got a connection and sat frozen. Polling single
        frames holds nothing open, and the preview was already throttled to 10 fps by
        design, so nothing is lost.
        """
        try:
            import cv2
        except ImportError:
            self._send(500, b"opencv missing", "text/plain")
            return
        frame = self.live.get_frame(label)
        if frame is None:
            self._send(503, b"no frame yet", "text/plain")
            return
        small = cv2.resize(frame, (320, 240))
        ok, buf = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ok:
            self._send(500, b"encode failed", "text/plain")
            return
        self._send(200, buf.tobytes(), "image/jpeg")

    def _mjpeg(self, label: str) -> None:
        try:
            import cv2
        except ImportError:
            self._send(500, b"opencv missing", "text/plain")
            return
        boundary = "swooshframe"
        self.send_response(200)
        self.send_header(
            "Content-Type", f"multipart/x-mixed-replace; boundary={boundary}")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        # Deliberately slow: the operator needs to see the scene, not every frame.
        # Four 640x480 JPEGs at 30 Hz would cost more CPU than the servo loop.
        period = 1.0 / 10.0
        try:
            while True:
                frame = self.live.get_frame(label)
                if frame is not None:
                    small = cv2.resize(frame, (320, 240))
                    ok, buf = cv2.imencode(".jpg", small,
                                           [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                    if ok:
                        data = buf.tobytes()
                        self.wfile.write(
                            f"--{boundary}\r\nContent-Type: image/jpeg\r\n"
                            f"Content-Length: {len(data)}\r\n\r\n".encode())
                        self.wfile.write(data)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                time.sleep(period)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _run_json(self, name: str) -> None:
        from .summarize import _read_jsonl

        run_dir = self.campaign.path / name
        if not run_dir.is_dir():
            self._json({"error": "no such run"}, 404)
            return
        run = next((r for r in self.campaign.runs() if r.path.name == name), None)
        if run is not None and run.status != "ready":
            # The UI turns this into a wiggle rather than a dialog.
            self._json({"error": f"run is {run.status}", "status": run.status}, 409)
            return
        ctl = _thin(_read_jsonl(run_dir / "raw" / "controller.jsonl"), 30.0)
        stt = _thin(_read_jsonl(run_dir / "raw" / "arm_state.jsonl"), 30.0)
        cam_meta = [json.loads(m.read_text())
                    for m in sorted((run_dir / "video").glob("*_frame_times.json"))]

        # RUN-RELATIVE TIME, TRIMMED TO THE FULLY-COVERED WINDOW.
        #
        # Raw timestamps are measured from t_loop0 -- the start of the whole session --
        # so a run that began 111 s in has rows at t=111.6..127.2. The player drives
        # `apply(t)` with t in 0..duration, so every lookup landed before the first row
        # and returned it: the page rendered one frozen sample and nothing moved.
        #
        # And the cameras open on a staggered worker, so each mp4 begins up to ~1.8 s
        # after the control streams. Starting the timeline at the earliest stream meant
        # the first couple of seconds had some panes with no video at all, which is the
        # glitchy opening. Start at the LATEST start and end at the EARLIEST end -- the
        # window where every stream is present -- which is also exactly the window the
        # exporter keeps, so playback shows what the dataset will contain.
        starts = [r["t"] for r in (ctl[:1] + stt[:1])]
        starts += [(d.get("t") or [0.0])[0] for d in cam_meta]
        ends = [r["t"] for r in (ctl[-1:] + stt[-1:])]
        ends += [(d.get("t") or [0.0])[-1] for d in cam_meta]
        t0 = max(starts) if starts else 0.0
        t1 = min(ends) if ends else t0
        covered = max(0.0, t1 - t0)

        # keep only rows inside the covered window
        ctl = [r for r in ctl if t0 <= r["t"] <= t1]
        stt = [r for r in stt if t0 <= r["t"] <= t1]

        cams = [{"label": d["label"],
                 # seconds into this camera's mp4 that the covered window begins, so
                 # the player seeks each video by (t + skip) and they stay in step
                 "skip": max(0.0, t0 - (d.get("t") or [0.0])[0]),
                 "url": f"/media/{name}/video/{d['label']}.mp4"}
                for d in cam_meta]
        meta = {}
        if (run_dir / "run.json").is_file():
            meta = json.loads((run_dir / "run.json").read_text())
        self._json({
            "name": name,
            # the COVERED duration, not the recorded one: playback spans only the
            # window where every stream has data
            "duration_s": covered,
            "recorded_duration_s": meta.get("duration_s", 0.0),
            "trimmed_s": max(0.0, meta.get("duration_s", 0.0) - covered),
            # t0 is kept so a consumer can get back to the raw clock if it needs to
            "t_origin": t0,
            "controller": [
                {"t": r["t"] - t0, "mx": r.get("move_x", 0), "my": r.get("move_y", 0),
                 "h": r.get("height", 0), "yaw": r.get("yaw", 0),
                 "g": r.get("gripper", 0)} for r in ctl],
            "proprio": [
                {"t": r["t"] - t0, "j": r.get("joints_deg") or [],
                 "w": r.get("pose_world_xyz_mm") or [],
                 "gp": r.get("gripper_pos")} for r in stt],
            "cameras": cams,
        })

    def _vendor_file(self, rel: str) -> None:
        base = (Path(__file__).resolve().parent / "vendor").resolve()
        target = (base / rel).resolve()
        if not str(target).startswith(str(base)) or not target.is_file():
            self._send(404, b"not found", "text/plain")
            return
        ctype = {"js": "text/javascript", "css": "text/css", "json": "application/json",
                 "stl": "model/stl"}.get(target.suffix.lstrip("."), "application/octet-stream")
        # `immutable` was wrong here and cost an afternoon: it is only correct for
        # content-addressed URLs, and /vendor/arm3d.js is a stable path whose CONTENT
        # we edit. Browsers then served a stale module for a week and the 3D view
        # silently kept running the previous code.
        # An ETag gets both properties: a conditional request per file (trivial on
        # localhost) and a 304 that re-sends none of the 4.4 MB when nothing changed.
        data = target.read_bytes()
        st = target.stat()
        etag = f'W/"{st.st_mtime_ns:x}-{st.st_size:x}"'
        if self.headers.get("If-None-Match") == etag:
            self.send_response(304)
            self.send_header("ETag", etag)
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            return
        self._send(200, data, ctype,
                   {"Cache-Control": "no-cache", "ETag": etag}, gzip_ok=True)

    def _media(self, rel: str) -> None:
        base = self.campaign.path.resolve()
        target = (base / rel).resolve()
        if not str(target).startswith(str(base)) or not target.is_file():
            self._send(404, b"not found", "text/plain")
            return
        ctype = {"mp4": "video/mp4", "png": "image/png",
                 "json": "application/json"}.get(target.suffix.lstrip("."),
                                                 "application/octet-stream")
        size = target.stat().st_size

        # RANGE REQUESTS. This used to answer "Accept-Ranges: none" and send the whole
        # file, which meant the browser could not SEEK: assigning video.currentTime was
        # ignored, so scrubbing did nothing and the playback panes sat on frame 0 while
        # the numbers moved. A <video> being driven to a timestamp needs ranges.
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            try:
                lo_s, _, hi_s = rng[len("bytes="):].partition("-")
                lo = int(lo_s) if lo_s else 0
                hi = int(hi_s) if hi_s else size - 1
                hi = min(hi, size - 1)
                if lo > hi or lo >= size:
                    raise ValueError("unsatisfiable")
            except ValueError:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            with target.open("rb") as fh:
                fh.seek(lo)
                chunk = fh.read(hi - lo + 1)
            self.send_response(206)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(chunk)))
            self.send_header("Content-Range", f"bytes {lo}-{hi}/{size}")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        self._send(200, target.read_bytes(), ctype, {"Accept-Ranges": "bytes"})


class DashboardServer:
    def __init__(self, cfg: dict, campaign: Campaign, live: LiveState,
                 port: int = 8770, host: str = "127.0.0.1") -> None:
        self.cfg = cfg
        self.live = live
        self.campaign_ref = {"campaign": campaign}
        self.sanity = SanityRunner()
        self.arm_ref: dict[str, Any] = {}
        self.port = int(port)
        self.host = str(host)
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def campaign(self) -> Campaign:
        return self.campaign_ref["campaign"]

    def start(self, open_browser: bool = True) -> str:
        handler = type("BoundHandler", (Handler,), {
            "live": self.live, "campaign_ref": self.campaign_ref, "cfg": self.cfg,
            "sanity": self.sanity, "arm_ref": self.arm_ref})
        for port in range(self.port, self.port + 20):
            try:
                self._httpd = ThreadingHTTPServer((self.host, port), handler)
                self.port = port
                break
            except OSError:
                continue
        if self._httpd is None:
            raise RuntimeError("no free port for the dashboard")
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        url = f"http://{'127.0.0.1' if self.host in ('0.0.0.0', '') else self.host}:{self.port}/"
        if open_browser:
            threading.Thread(target=self._open, args=(url,), daemon=True).start()
        return url

    @staticmethod
    def _open(url: str) -> None:
        import shutil
        import subprocess
        import webbrowser

        time.sleep(0.4)
        for exe in ("google-chrome", "chromium", "chromium-browser"):
            if shutil.which(exe):
                try:
                    subprocess.Popen([exe, url],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    return
                except Exception:
                    pass
        try:
            webbrowser.open(url)
        except Exception:
            pass

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()


def _vendor(name: str) -> str:
    """Inline a vendored asset so the page stays a single self-contained file --
    the lab machine may have no internet, and a CDN fetch that fails would take the
    resizable panes with it."""
    p = Path(__file__).resolve().parent / "vendor" / name
    try:
        return p.read_text(encoding="utf-8")
    except OSError:
        return ""


PAGE = r"""<!doctype html><html><head><meta charset=utf-8>
<title>swoosh collect</title>
<style>
:root{
  --bg:#ffffff; --panel:#f6f7f9; --fg:#14171a; --dim:#6b7280; --line:#d7dbe0;
  --blue:#1668dc; --red:#d92d20; --green:#0f9d58; --amber:#b25e02;
}
*{box-sizing:border-box}
html,body{height:100%;margin:0;overflow:hidden}
body{display:flex;flex-direction:column;font:13px/1.45 ui-sans-serif,system-ui,-apple-system,sans-serif;
  background:var(--bg);color:var(--fg)}
#main{display:flex;flex:1;min-height:0;width:100%}
/* every control is a hard-edged square-cornered rectangle */
button,input,select{border-radius:0!important;font:inherit}
button{background:#fff;color:var(--fg);border:1px solid var(--line);padding:6px 11px;cursor:pointer}
button:hover:not(:disabled){border-color:var(--blue);color:var(--blue)}
button:disabled{opacity:.45;cursor:default}
button.blue{background:var(--blue);color:#fff;border-color:var(--blue)}
button.blue:hover{background:#0f5ac2;color:#fff}
button.red{background:var(--red);color:#fff;border-color:var(--red)}
button.red:hover{background:#b3251a;color:#fff}
button.green{background:var(--green);color:#fff;border-color:var(--green)}
button.green:hover{background:#0b7d45;color:#fff}
.split{display:flex;flex-direction:row;height:100%;width:100%}
.gutter{background:var(--line);background-repeat:no-repeat;background-position:50%}
.gutter:hover{background:var(--blue)}
.gutter.gutter-horizontal{cursor:col-resize;width:5px}
.gutter.gutter-vertical{cursor:row-resize;height:5px}
#left{display:flex;flex-direction:column;height:100%;min-width:0;overflow:hidden}
#right{display:flex;flex-direction:column;height:100%;min-width:0;overflow:hidden;padding:0 10px 0}
#imgblock{display:flex;flex-direction:column;min-height:0;overflow:hidden}
#panelblock{display:flex;flex-direction:column;min-height:0;overflow:hidden;padding-bottom:8px;
  container-type:inline-size}
#campaignpane{display:flex;flex-direction:column;min-height:0;overflow:hidden}
#sanity{display:flex;flex-direction:column;min-height:0;overflow:hidden;background:var(--panel);
  border-top:1px solid var(--line);padding:8px 10px}
.hd{padding:12px 14px;border-bottom:1px solid var(--line);background:var(--panel)}
h1{margin:0;font-size:16px} .sub{color:var(--dim);font-size:12px}
#runs{flex:1;overflow:auto;padding:7px}
/* Campaign total sits below the scrolling run list, always visible. */
#camptotal{flex:none;border-top:1px solid var(--line);background:var(--panel);
  padding:7px 11px;display:flex;justify-content:space-between;align-items:baseline;
  font-variant-numeric:tabular-nums}
#camptotal .lb{font-size:10px;text-transform:uppercase;letter-spacing:.07em;
  color:var(--dim);font-weight:700}
#camptotal .tv{font-size:15px;font-weight:700}
#camptotal .sub2{font-size:11px;color:var(--dim);font-weight:400;margin-left:6px}
.run{padding:7px 9px;border:1px solid var(--line);background:#fff;margin-bottom:5px;cursor:pointer;
  display:flex;gap:8px;align-items:center}
.run:hover{border-color:var(--blue)} .run.sel{border-color:var(--blue);background:#eaf2fe}
.run .n{font-weight:700;font-variant-numeric:tabular-nums}
.run .d{color:var(--dim);margin-left:auto;font-variant-numeric:tabular-nums}
.run .ts{color:var(--dim);font-size:11px;font-variant-numeric:tabular-nums}
.run .meta{display:flex;flex-direction:column;gap:1px;min-width:0}
.dot{width:7px;height:7px;background:var(--green);flex:none}
.dot.bad{background:var(--dim)}
.tag{font-size:10px;font-weight:700;letter-spacing:.04em;padding:1px 6px;text-transform:uppercase;color:#fff}
.tag.recording{background:var(--red);animation:pulse 1.1s ease-in-out infinite}
.tag.processing{background:var(--amber)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.45}}
@keyframes wig{0%,100%{transform:translateX(0)}15%{transform:translateX(-7px)}30%{transform:translateX(6px)}
 45%{transform:translateX(-5px)}60%{transform:translateX(4px)}75%{transform:translateX(-2px)}}
.run.wiggle{animation:wig .42s ease;border-color:var(--red)!important;background:#fdeceb}
.iconbtn{font-size:11px;padding:2px 6px;color:var(--dim)}
.iconbtn.del{color:var(--red);border-color:var(--line)}
.iconbtn.del:hover{background:var(--red);color:#fff;border-color:var(--red)}
#sanity h3{margin:0 0 6px;font-size:10px;text-transform:uppercase;letter-spacing:.07em;color:var(--dim);
  font-weight:700;display:flex;align-items:center;gap:7px;flex:none}
#sanity h3 .sp{margin-left:auto;display:flex;gap:5px}
.sbtns{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:6px;flex:none}
.sbtn{font-size:11px;padding:3px 8px}
#term{flex:1;min-height:0;background:#000;border:1px solid #000;overflow:auto;padding:6px 8px;
  font:11px/1.35 ui-monospace,Menlo,Consolas,monospace;color:#d6dde3;white-space:pre-wrap;word-break:break-word}
#term .e{color:#ff6b6b} #term .g{color:#4ade80} #term .c{color:#60a5fa} #term .w{color:#fbbf24}
.spin{display:inline-block;width:8px;height:8px;background:var(--blue);animation:pulse .9s ease-in-out infinite}
/* ticker: flush to the panel edges, scrolls forever like a stock ticker */
#ticker{flex:none;overflow:hidden;white-space:nowrap;border-bottom:1px solid var(--line);
  margin:0 -10px;padding:5px 0;font-weight:800;letter-spacing:.16em;font-size:12px}
#ticker .t{display:inline-block;animation:scroll 52s linear infinite;will-change:transform}
@keyframes scroll{from{transform:translateX(0)}to{transform:translateX(-50%)}}
#ticker.blue{color:var(--blue);background:#eaf2fe}
#ticker.green{color:var(--green);background:#e9f7ef}
#ticker.red{color:#fff;background:var(--red)}
.mode{padding:8px 0;display:flex;align-items:center;gap:9px;flex:none}
.badge{padding:2px 9px;font-size:11px;font-weight:700;letter-spacing:.05em;color:#fff}
.badge.live{background:var(--blue)} .badge.rec{background:var(--red)} .badge.rep{background:var(--green)}
#cams{display:grid;grid-template-columns:1fr 1fr;gap:6px;min-height:0;flex:1 1 auto}
.cam{position:relative;background:#000;overflow:hidden;min-height:0}
/* contain, never cover: `cover` crops whatever does not fit the pane's aspect,
   which silently hid the edges of every feed -- and the edges are where you check
   whether the gripper is actually in frame. Letterbox against the black instead.
   width/height 100% + contain re-fits on every resize with no JS. */
.cam img,.cam video{width:100%;height:100%;object-fit:contain;display:block}
.cam .lb{position:absolute;left:5px;top:4px;font-size:10px;background:#000a;color:#fff;padding:1px 5px}
/* Three panes with draggable boundaries (Split.js sets the widths inline), so the
   operator decides how much goes to the pad, the numbers and the 3D view. That
   replaces a fixed grid whose column widths could never suit every screen. */
.panels{display:flex;flex:1 1 auto;min-height:0;padding-top:8px}
.panels>.card{min-width:0;min-height:0}
/* Controller stacks downward -- pad, then face buttons, then triggers -- so it needs
   only the canvas's width instead of sitting beside the buttons. */
.ctlstack{display:flex;flex-direction:column;gap:7px;align-items:flex-start;
  min-height:0;overflow:hidden}
.ctlstack #trig{width:100%}
#pad{max-width:100%;height:auto}
/* arm3d fills; the workspace box fields are a horizontal strip beneath it. */
.scene{display:flex;flex-direction:column;gap:7px;flex:1 1 auto;min-height:0;min-width:0}
#wsfields{flex:none;display:flex;flex-wrap:wrap;gap:4px 7px;align-items:center}
#wsfields label{font-size:9px;letter-spacing:.02em;color:var(--dim);display:flex;
  align-items:center;gap:3px;white-space:nowrap}
#wsfields input{width:50px;padding:2px 3px;font-size:11px;border:1px solid var(--line);
  font-variant-numeric:tabular-nums}
#arm3d{flex:1 1 auto;min-width:0;min-height:0;width:100%;height:100%;
  background:#f6f7f9;border:1px solid var(--line)}
/* three.js is called with setSize(w,h,false) -- updateStyle FALSE -- so it sets
   the drawing buffer but not the CSS size. With setPixelRatio(2) on a HiDPI
   screen the element then defaults to its attribute size, twice the container,
   and the scene gets cut off. Pin the CSS size to the box instead. */
#arm3d canvas{display:block;width:100%!important;height:100%!important}
.card{background:var(--panel);border:1px solid var(--line);padding:9px}
.card{display:flex;flex-direction:column;min-height:0;min-width:0;overflow:hidden}
.card h2{margin:0 0 7px;font-size:10px;text-transform:uppercase;letter-spacing:.06em;color:var(--dim);font-weight:700}
.row{display:flex;justify-content:space-between;gap:10px;font-variant-numeric:tabular-nums;padding:1px 0}
/* label over value, for readings too wide to sit beside their name */
.stackrow{display:flex;flex-direction:column;padding:1px 0 3px;
  font-variant-numeric:tabular-nums}
.stackrow .k{color:var(--dim);font-size:11px}
.stackrow .v{color:#000;font-weight:600}
.row span:last-child{color:#000;font-weight:600}
.bar{height:4px;background:#e3e6ea;overflow:hidden;margin-top:2px}
.bar>div{height:100%;background:#000}
.abxy{display:grid;grid-template-columns:repeat(3,30px);grid-template-rows:repeat(3,30px);gap:3px}
/* face buttons beside a legend saying what each one does */
.padrow{display:flex;gap:11px;align-items:flex-start}
.btnkey{display:grid;grid-template-columns:auto 1fr;gap:2px 6px;margin:0;
  align-content:start;align-items:baseline}
.btnkey dt{font:700 10px/1.5 ui-sans-serif,system-ui,sans-serif;color:#fff;
  text-align:center;padding:0 4px;min-width:15px}
.btnkey dd{margin:0;font-size:11px;color:var(--dim);white-space:nowrap}
.btnkey .ka{background:#107C10} .btnkey .kb{background:#D13438}
.btnkey .kx{background:#0A5C96} .btnkey .ky{background:#F7B500;color:#3a2c00}
.abxy button{width:30px;height:30px;padding:0;font-weight:800;font-size:13px;color:#fff}
/* real Xbox face-button colours */
.abxy .a{background:#107C10;border-color:#107C10}
.abxy .b{background:#D13438;border-color:#D13438}
.abxy .x{background:#0A5C96;border-color:#0A5C96}
.abxy .y{background:#F7B500;border-color:#F7B500;color:#3a2c00}
.abxy button:hover:not(:disabled){filter:brightness(1.12);color:#fff}
.abxy .y:hover{color:#3a2c00}
.abxy button:disabled{opacity:.5}
#pbbar{flex:none;border:1px solid var(--line);margin:6px 0 0;padding:7px 9px;display:flex;
  align-items:center;gap:10px;background:var(--panel)}
#pbbar.off{opacity:.4;pointer-events:none}
#pbslider{flex:1;height:8px}
.noUi-connect{background:var(--blue)!important}
.noUi-handle{border-radius:0!important;box-shadow:none!important;width:14px!important;
  height:18px!important;right:-7px!important;top:-6px!important;cursor:grab}
.noUi-handle:before,.noUi-handle:after{display:none}
.abxy .y{grid-area:1/2} .abxy .x{grid-area:2/1} .abxy .b{grid-area:2/3} .abxy .a{grid-area:3/2}
.pill{font-size:10px;padding:3px 7px;border:1px solid var(--line);background:#fff;cursor:pointer}
#pb{display:none;padding:7px 0;align-items:center;gap:9px;flex:none}
#pb input[type=range]{flex:1}
#modal{position:fixed;inset:0;background:#0006;display:none;align-items:center;justify-content:center;z-index:50}
#modal.on{display:flex}
.mbox{background:#fff;border:1px solid var(--line);padding:18px 20px;min-width:420px;max-width:660px;
  max-height:80vh;overflow:auto;box-shadow:0 16px 44px #0003}
.mbox h4{margin:0 0 4px;font-size:15px} .mbox p{margin:0 0 14px;color:var(--dim);font-size:12px}
.mrow{display:flex;align-items:center;gap:9px;margin-bottom:7px}
.mrow label{flex:1;font-size:12px;font-variant-numeric:tabular-nums}
.mrow input,.mrow select{border:1px solid var(--line);padding:5px 8px;min-width:170px;background:#fff;color:var(--fg)}
.mact{display:flex;gap:8px;justify-content:flex-end;margin-top:15px}
</style></head><body>
<div id=main>
<div id=left>
  <div id=campaignpane>
    <div class=hd>
      <h1 id=cname>...</h1>
      <div class=sub id=csub></div>
      <div style="margin-top:8px;display:flex;gap:6px;flex-wrap:wrap">
        <button class=blue onclick=newCampaign()>+ New campaign</button>
        <button onclick=loadCampaign_pick()>Load campaign</button>
        <button onclick=chooseRoot()>Choose root folder</button>
        <button onclick="openFolder('')">Open campaign folder</button>
      </div>
      <div class=sub id=rootpath style="margin-top:6px;font-size:11px;word-break:break-all"></div>
    </div>
    <div id=runs></div>
    <div id=camptotal></div>
  </div>
  <div id=sanity>
    <h3>sanity <span id=srun></span>
      <span class=sp>
        <button class=sbtn onclick=copyLogs() id=cpbtn>copy full logs</button>
        <button class=sbtn onclick=clearLogs()>clear</button>
      </span>
    </h3>
    <div class=sbtns>
      <button class=sbtn onclick="sanity('arm')" data-excl data-tip="Connect, clear errors, print firmware/joints/pose. Read-only." title="Connect, clear errors, print firmware/joints/pose. Read-only.">check arm</button>
      <button class="sbtn red" onclick=confirmFrame() data-excl title="MOVES THE ARM: jogs 20mm along each world axis to prove the 45-degree mount maths.">verify 45&deg; frame</button>
      <button class=sbtn onclick=nameCameras() data-excl title="Assign labels to cameras by USB port.">name cameras</button>
      <button class=sbtn onclick="sanity('lag')" title="Sweep lag on the latest run; a U-shaped minimum means the action is time-aligned.">cmd&rarr;measured lag</button>
      <button class="sbtn red" onclick=confirmCamLatency() data-excl title="MOVES THE GRIPPER: measures action-to-pixel latency.">camera latency</button>
      <button class=sbtn onclick="sanity('fkcheck')" data-excl data-tip="Compare FK(joints) with the controller's own TCP pose -- verifies the chain behind the 3D view." title="Compare FK(joints) with the controller's own TCP pose -- verifies the chain behind the 3D view.">FK check</button>
      <button class=sbtn onclick="sanity('reach')" data-excl data-tip="Are all 8 corners of the safety box reachable?" title="Are all 8 corners of the safety box reachable?">workspace reach</button>
      <button class=sbtn onclick="sanity('validate')" title="Check every run for the known failure modes.">validate runs</button>
      <button class=sbtn onclick="sanity('preview')" id=pvbtn title="Feed the dashboard synthetic data (no hardware).">preview mode</button>
    </div>
    <div id=term></div>
  </div>
</div>
<div id=right>
  <div id=ticker class=blue><span class=t id=tickertext></span></div>
  <div class=mode>
    <span id=modetxt class=sub></span>
    <button id=backlive style="margin-left:auto;display:none" onclick=goLive()>Back to live</button>
  </div>
  <div id=imgblock>
    <div id=cams></div>
    <div id=pbbar class=off>
      <button id=pbtn onclick=togglePlay()>Play</button>
      <button id=pstop onclick=stopPlay()>Stop</button>
      <div id=pbslider></div>
      <span id=ptime class=sub style="font-variant-numeric:tabular-nums">0.0 / 0.0 s</span>
    </div>
  </div>
  <div id=panelblock>
  <div class=panels>
    <div class="card" id=cardctl><h2>controller</h2>
      <div class=ctlstack>
        <canvas id=pad width=224 height=120></canvas>
        <div class=padrow>
          <div class=abxy>
            <button class=y onclick="press('rehome')" title="re-home the arm">Y</button>
            <button class=x onclick="press('clear_errors')"
                    title="clear errors + warnings, re-enter servo mode">X</button>
            <button class=b onclick="press('stop_episode')" title="stop recording">B</button>
            <button class=a onclick="press('start_episode')" title="start recording">A</button>
          </div>
          <dl class=btnkey>
            <dt class=ka>A</dt><dd>start run</dd>
            <dt class=kb>B</dt><dd>stop run</dd>
            <dt class=kx>X</dt><dd>clear errors</dd>
            <dt class=ky>Y</dt><dd>re-home</dd>
          </dl>
        </div>
        <div id=trig></div>
      </div>
    </div>
    <div class="card" id=cardprop><h2>proprioception</h2><div id=prop></div></div>
    <div class="card" id=cardscene><h2>3D scene</h2>
      <div class=scene>
        <div id=arm3d></div>
        <div id=wsfields></div>
      </div>
    </div>
  </div>
</div>
</div>
<div id=modal><div class=mbox id=mbox></div></div>
<style>__NOUI_CSS__</style>
<script>__SPLIT_JS__</script>
<script>__NOUI_JS__</script>
<script>
let CAMS=[], SEL=null, REP=null, T0=0, PLAYING=false, LIVE={}, ARM=null, PREVIEW=false;
function setTicker(text, cls){
  if(tickertext.dataset.k===text+cls) return;
  tickertext.dataset.k=text+cls;
  // Build ONE unit, measure it, then repeat until the strip is at least twice the
  // container width. The animation shifts by exactly -50%, so the second half is
  // always in frame as the first leaves -- with too few repeats the strip is narrower
  // than the viewport and the tail scrolls off into blank space.
  const unit=text+' \u00b7 ';
  tickertext.textContent=unit;
  const uw=Math.max(1,tickertext.offsetWidth);
  const need=Math.max(4,Math.ceil((ticker.offsetWidth*2)/uw)+2);
  const half=unit.repeat(need);
  tickertext.textContent=half+half;      // exactly 2 halves <-> the -50% keyframe
  ticker.className=cls;
}
addEventListener('resize',()=>{ const k=tickertext.dataset.k; tickertext.dataset.k='';
  if(k) setTicker(k.replace(/(blue|green|red)$/,''), k.match(/(blue|green|red)$/)?.[0]||'blue'); });
function fmt(v,n=1){return (v==null||isNaN(v))?'--':Number(v).toFixed(n)}
// HH:MM:SS. Rounds to whole seconds, so a campaign total is the sum of what is
// displayed rather than drifting from it.
function hms(sec){
  if(sec==null||isNaN(sec)) return '--:--:--';
  const t=Math.max(0,Math.round(Number(sec)));
  const h=Math.floor(t/3600), m=Math.floor((t%3600)/60), s=t%60;
  return String(h).padStart(2,'0')+':'+String(m).padStart(2,'0')+':'+String(s).padStart(2,'0');
}

// resizable panes -- Split.js, vendored so the page needs no network
Split(['#left','#right'],{sizes:[34,66],minSize:[260,420],gutterSize:5,cursor:'col-resize'});
Split(['#campaignpane','#sanity'],{direction:'vertical',sizes:[58,42],
  minSize:[110,150],gutterSize:5,cursor:'row-resize'});
// images (+ their playback bar) above, the controller/proprioception/3D panels below
Split(['#imgblock','#panelblock'],{direction:'vertical',sizes:[50,50],
  minSize:[140,180],gutterSize:5,cursor:'row-resize'});
// controller | proprioception | 3D scene, each boundary draggable. The 3D view takes
// the largest default share and arm3d.js already has a ResizeObserver, so it re-renders
// as the gutters move.
Split(['#cardctl','#cardprop','#cardscene'],{sizes:[20,20,60],
  minSize:[120,110,140],gutterSize:5,cursor:'col-resize'});

async function loadCampaign(){
  const c=await (await fetch('/api/campaign')).json();
  cname.textContent=c.name;
  csub.textContent=`${c.runs.length} run(s) - ${c.complete} complete`;
  // Sum only runs that finished processing -- the same ones showing a duration, so the
  // total always equals the sum of what is on screen. Rounded per-run first, for the
  // same reason.
  const done=c.runs.filter(r=>r.status==='ready');
  const totalS=done.reduce((a,r)=>a+Math.max(0,Math.round(r.duration_s||0)),0);
  const pending=c.runs.length-done.length;
  camptotal.innerHTML=`<span class=lb>campaign total</span>`
    +`<span><span class=tv>${hms(totalS)}</span>`
    +`<span class=sub2>${done.length} run(s)`
    +`${pending?` &middot; ${pending} not counted yet`:''}</span></span>`;
  runs.innerHTML='';
  c.runs.slice().reverse().forEach(r=>{
    const d=document.createElement('div');
    d.className='run'+(SEL===r.name?' sel':''); d.id='run_'+r.name;
    const tag=r.status==='ready'?'':
      `<span class="tag ${r.status}">${r.status==='recording'?'in progress':'processing'}</span>`;
    d.innerHTML=`<span class="dot${r.complete?'':' bad'}"></span>
      <div class=meta><div><span class=n>${String(r.index).padStart(4,'0')}</span> ${tag}</div>
      <div class=ts>${r.timestamp||''}</div></div>
      <span class=d title="HH:MM:SS -- shown once processing is done">${r.status==='ready'?hms(r.duration_s):'--:--:--'}</span>
      <button class=iconbtn title="open this run's folder">Open run folder</button>
      <button class="iconbtn del" title="delete this run permanently">Delete</button>`;
    d.querySelector('.iconbtn').onclick=e=>{e.stopPropagation();openFolder(r.name)};
    d.querySelector('.del').onclick=e=>{e.stopPropagation();delRun(r.name,r.duration_s)};
    d.onclick=()=>{ if(r.status!=='ready'){wiggle(d);return} openRun(r.name); };
    runs.appendChild(d);
  });
}
function delRun(name,dur){
  // Deleting a run destroys raw data that cannot be re-recorded, so it asks first and
  // names what is going away.
  showModal(`<h4>Delete this run?</h4>
    <p><b>${name}</b> &middot; ${hms(dur)}<br>
    Its raw streams, videos and summary are removed from disk permanently. There is no
    undo, and a recording cannot be reproduced.</p>
    <div class=mact><button onclick=closeModal()>Cancel</button>
    <button class=red onclick="doDelRun('${name}')">Delete permanently</button></div>`);
}
async function doDelRun(name){
  const r=await (await fetch('/api/run/delete',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify({run:name})})).json();
  closeModal();
  if(!r.ok){ toast(r.error||'could not delete'); return; }
  if(SEL===name){ SEL=null; goLive(); }
  loadCampaign();
}
async function loadCampaign_pick(){
  const r=await (await fetch('/api/campaign/list')).json();
  if(!r.campaigns || !r.campaigns.length){ toast('no campaigns yet - create one first'); return; }
  showModal(`<h4>Load campaign</h4>
    <p>Switches where new runs are written. A run in progress must be stopped first.</p>
    <div class=mrow><label>campaign</label>
      <select id=campin style="flex:2">${r.campaigns.map(c=>
        `<option value="${c.slug}" ${c.active?'selected':''}>${c.name} - ${c.runs} run(s)`
        +`${c.active?'  (current)':''}</option>`).join('')}</select></div>
    <div class=mact><button onclick=closeModal()>Cancel</button>
    <button class=blue onclick=doLoadCampaign()>Load</button></div>`);
}
async function doLoadCampaign(){
  const slug=document.getElementById('campin').value;
  const r=await (await fetch('/api/campaign/load',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify({slug})})).json();
  if(!r.ok){ toast(r.error||'could not load'); return; }
  closeModal(); SEL=null; REP=null; goLive(); loadCampaign();
}
function wiggle(el){ el.classList.remove('wiggle'); void el.offsetWidth;
  el.classList.add('wiggle'); setTimeout(()=>el.classList.remove('wiggle'),500); }
async function openFolder(run){
  const r=await (await fetch('/api/open',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({run:run||''})})).json();
  if(r.ok) return;
  if(r.containerised){
    // Give them something they can actually use: the host path, selected ready to copy.
    showModal(`<h4>Folder on this machine</h4>
      <p>${r.error}</p>
      <div class=mrow><input id=fpath readonly value="${r.path}" style="flex:2"></div>
      <div class=mact><button onclick=closeModal()>Close</button>
      <button class=blue onclick="copyPath()">Copy path</button></div>`);
    const i=document.getElementById('fpath'); i.focus(); i.select();
    return;
  }
  toast('could not open folder: '+(r.error||'')); }
async function copyPath(){
  const v=document.getElementById('fpath').value;
  try{ await navigator.clipboard.writeText(v); toast('path copied'); }
  catch(e){ toast('select the text and copy it manually'); } }
async function press(button){
  const r=await (await fetch('/api/button',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({button})})).json();
  if(!r.ok) toast(r.error||'button failed'); }

// Where each feed sits in the 2x2, read row-major: TL, TR, BL, BR.
// Scene pair down the left, right-gripper pair down the right, top above bottom --
// so the grid mirrors the rig instead of following expected_labels' order.
const CAM_SLOTS=['scene_left','gripper_right_top','scene_right','gripper_right_bottom'];
function camOrder(labels){
  return labels.slice().sort((a,b)=>{
    const ia=CAM_SLOTS.indexOf(a), ib=CAM_SLOTS.indexOf(b);
    return (ia<0?99:ia)-(ib<0?99:ib);
  });
}
let POLL=[];        // one self-pacing snapshot loop per live camera
function stopPolling(){ POLL.forEach(p=>{p.stop=true}); POLL=[]; }
function camGrid(labels,playback){
  CAMS=labels; cams.innerHTML=''; stopPolling();
  labels=camOrder(labels);
  labels.forEach(l=>{ const d=document.createElement('div'); d.className='cam';
    d.innerHTML = playback ? `<video id="v_${l}" muted preload=auto></video><span class=lb>${l}</span>`
                           : `<img id="s_${l}" alt="${l}"><span class=lb>${l}</span>`;
    cams.appendChild(d); });
  if(playback) return;
  // Ask for the next frame only once the previous one has arrived. That self-paces to
  // whatever the machine can actually deliver and, crucially, never holds a connection
  // open -- four persistent MJPEG streams starved the browser's per-origin budget and
  // left the last two panes frozen.
  labels.forEach(l=>{
    const img=document.getElementById('s_'+l); if(!img) return;
    const p={stop:false}; POLL.push(p);
    const next=()=>{ if(p.stop) return;
      setTimeout(()=>{ if(!p.stop) img.src='/snapshot/'+encodeURIComponent(l)+'?n='+Date.now(); },40); };
    img.onload=next;
    img.onerror=next;      // 503 before the first frame lands is normal; keep asking
    img.src='/snapshot/'+encodeURIComponent(l)+'?n='+Date.now();
  });
}
const es=new EventSource('/api/events');
es.onmessage=e=>{ LIVE=JSON.parse(e.data);
  if(!REP){ draw(LIVE.controller||{}); proprio(LIVE.proprio||{});
    const st=LIVE.status||{};
    if(st.cameras && st.cameras.join()!==CAMS.join()) camGrid(st.cameras,false);
    modetxt.textContent=st.recording?`${st.run_name||''} - ${fmt(st.elapsed)}s`:(st.hint||'');
    PREVIEW=!!st.preview;
    // Mirror the server-side lockout: these take over the arm and cameras, so they
    // must not look available during a run.
    document.querySelectorAll('#sanity .sbtn[data-excl]').forEach(b=>{
      b.disabled=!!st.recording;
      if(st.recording){ if(!b.dataset.tip0) b.dataset.tip0=b.title;
        b.title='unavailable during a run - press B to stop first'; }
      else if(b.dataset.tip0){ b.title=b.dataset.tip0; }
    });
    if(st.recording) setTicker('RECORDING','red');
    else if(PREVIEW)  setTicker('SIMULATED','blue');
    else              setTicker('LIVE','green');
    if(ARM){ ARM.setJoints(LIVE.proprio?.joints||[], 1-((LIVE.controller?.gripper)||0));
             ARM.setTarget(LIVE.proprio?.target||null, LIVE.proprio?.target_yaw||0); }
  }};

function stick(ctx,cx,cy,r,x,y,label){
  ctx.strokeStyle='#c9cfd6'; ctx.lineWidth=1;
  ctx.beginPath(); ctx.arc(cx,cy,r,0,7); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(cx-r,cy); ctx.lineTo(cx+r,cy);
  ctx.moveTo(cx,cy-r); ctx.lineTo(cx,cy+r); ctx.stroke();
  ctx.fillStyle='#000';
  ctx.beginPath(); ctx.arc(cx+x*r*0.86, cy-y*r*0.86, 6,0,7); ctx.fill();
  ctx.fillStyle='#6b7280'; ctx.font='10px system-ui'; ctx.textAlign='center';
  ctx.fillText(label,cx,cy+r+12);
}
function draw(c){
  const ctx=pad.getContext('2d'); ctx.clearRect(0,0,224,120);
  stick(ctx,54,50,40, c.move_y??c.my??0, c.move_x??c.mx??0, 'move in surface plane');
  stick(ctx,164,50,40, c.yaw??0, c.height??c.h??0, 'height / yaw');
  const g=c.gripper??c.g??0;
  // Left trigger is unbound but still displayed: a live number is how you tell a
  // dead pad from one nobody is touching.
  const lt=(c.raw&&c.raw.ABS_Z!=null)?c.raw.ABS_Z:null;
  trig.innerHTML=
    `<div class=row><span>left trigger <span class=sub>(unbound)</span></span>
       <span>${lt==null?'--':fmt(lt,2)}</span></div>
     <div class=bar><div style="width:${lt==null?0:(lt*100).toFixed(0)}%"></div></div>
     <div class=row><span>right trigger</span><span>${fmt(g,2)}</span></div>
     <div class=bar><div style="width:${(g*100).toFixed(0)}%"></div></div>`;
}
function proprio(p){
  const j=p.joints||[], w=p.world||[]; let h='';
  // Label on its own line above the value: the xyz triple is the widest thing here and
  // side-by-side it set the whole card's minimum width.
  if(w.length===3) h+=`<div class=stackrow><span class=k>end effector world position (mm)</span>`
    +`<span class=v>x ${fmt(w[0])} &nbsp; y ${fmt(w[1])} &nbsp; z ${fmt(w[2])}</span></div>`;
  if(p.gripper!=null) h+=`<div class=row><span>gripper</span><span>${fmt(p.gripper,0)}</span></div>`;
  j.forEach((v,i)=>{ const f=Math.max(0,Math.min(1,(v+180)/360));
    h+=`<div class=row><span>Joint ${i+1}</span><span>${fmt(v)}&deg;</span></div>
        <div class=bar><div style="width:${(f*100).toFixed(0)}%"></div></div>`;});
  if(p.error) h+=`<div class=row style="color:var(--red)"><span>error</span><span>code ${p.error}</span></div>`;
  prop.innerHTML=h||'<span class=sub>waiting</span>';
}

async function openRun(name){
  SEL=name; loadCampaign();
  const resp=await fetch('/api/run/'+name); const r=await resp.json();
  if(!resp.ok||r.error){ const el=document.getElementById('run_'+name);
    if(el) wiggle(el); else toast(r.error||'could not open run'); return; }
  REP=r; camGrid(r.cameras.map(c=>c.label),true);
  r.cameras.forEach(c=>{ const v=document.getElementById('v_'+c.label); if(v){v.src=c.url;v.load();} });
  modetxt.textContent=`${name} - ${fmt(r.duration_s)}s`
    + (r.trimmed_s>0.05?` (trimmed ${fmt(r.trimmed_s)}s where a stream was missing)`:'');
  setTicker('PLAY BACK '+name,'blue');
  backlive.style.display=''; pbbar.classList.remove('off');
  DUR=Math.max(0.1,r.duration_s); SLIDER.updateOptions({range:{min:0,max:DUR}},true);
  T0=performance.now()/1000; PLAYING=true; pbtn.textContent='Pause';
  // Actually start them. openRun set PLAYING=true but never called play(), so the
  // panes held their first frame while the numbers advanced.
  r.cameras.forEach(c=>{ const v=document.getElementById('v_'+c.label);
    if(v){ v.currentTime=(c.skip||0); const p=v.play(); if(p&&p.catch) p.catch(()=>{}); } });
  requestAnimationFrame(tick);
}
function goLive(){ REP=null; PLAYING=false; backlive.style.display='none';
  pbbar.classList.add('off');
  SEL=null; loadCampaign(); const st=LIVE.status||{}; camGrid(st.cameras||[],false);
  modetxt.textContent=''; }
function togglePlay(){ if(!REP)return; PLAYING=!PLAYING; pbtn.textContent=PLAYING?'Pause':'Play';
  if(PLAYING){T0=performance.now()/1000-parseFloat(SLIDER.get()); requestAnimationFrame(tick);}
  REP.cameras.forEach(c=>{const v=document.getElementById('v_'+c.label); if(v){PLAYING?v.play():v.pause();}}); }
let DUR=1, SLIDER=null, DRAG=false;
noUiSlider.create(pbslider,{start:0,connect:'lower',range:{min:0,max:1},step:0.01,
  behaviour:'drag-tap'});
SLIDER=pbslider.noUiSlider;
SLIDER.on('start',()=>{DRAG=true});
SLIDER.on('slide',(v)=>{ const t=parseFloat(v[0]); T0=performance.now()/1000-t; apply(t); });
SLIDER.on('end',()=>{DRAG=false});
function stopPlay(){ if(!REP)return;
  // Stop means "done with this recording": pause the videos and hand the whole right
  // panel back to the live feeds, rather than parking on frame 0 of a run.
  PLAYING=false; pbtn.textContent='Play';
  REP.cameras.forEach(c=>{const v=document.getElementById('v_'+c.label); if(v)v.pause();});
  SLIDER.set(0); goLive(); }
function at(rows,t){ if(!rows.length)return null; let lo=0,hi=rows.length-1;
  while(lo<hi){const m=(lo+hi)>>1; if(rows[m].t<t)lo=m+1; else hi=m;}
  const a=rows[Math.max(0,lo-1)],b=rows[lo];
  return (Math.abs(a.t-t)<Math.abs(b.t-t))?a:b; }
function apply(t){
  const c=at(REP.controller,t); if(c) draw(c);
  const p=at(REP.proprio,t); if(p) proprio({joints:p.j,world:p.w,gripper:p.gp});
  ptime.textContent=fmt(t)+' / '+fmt(DUR)+' s';
  if(ARM){ ARM.setJoints(p?p.j:[], 1-((c?(c.g??c.gripper):0)||0));
           if(p&&p.w&&p.w.length===3) ARM.setTarget(p.w,0); }
  REP.cameras.forEach(c=>{ const v=document.getElementById('v_'+c.label);
    if(v&&isFinite(v.duration)){
      // t is measured from the start of the COVERED window; each mp4 starts earlier,
      // so skip into it by however much of it precedes that window.
      const want=Math.max(0,t+(c.skip||0));
      if(Math.abs(v.currentTime-want)>0.15) v.currentTime=want; } });
}
function tick(){ if(!REP||!PLAYING)return; let t=performance.now()/1000-T0;
  if(t>DUR){t=0;T0=performance.now()/1000;}
  if(!DRAG) SLIDER.set(t);
  apply(t); requestAnimationFrame(tick); }

let SINCE=0;
function cls(l){ if(/FAIL|error|Error|Traceback|WARNING|refused|not found/.test(l))return 'e';
  if(/PASS|OK:|saved|verified|reached|connected/.test(l))return 'g';
  if(/^\$ /.test(l))return 'c'; if(/WARN|processing/.test(l))return 'w'; return ''; }
async function pollSanity(){
  try{ const r=await (await fetch('/api/sanity/log?since='+SINCE)).json();
    if(r.lines.length){ r.lines.forEach(l=>{ const d=document.createElement('div');
      const c=cls(l); if(c)d.className=c; d.textContent=l; term.appendChild(d); });
      SINCE=r.total; term.scrollTop=term.scrollHeight; }
    srun.innerHTML=r.running?`<span class=spin></span>`:'';
    document.querySelectorAll('.sbtns .sbtn').forEach(b=>b.disabled=!!r.running);
  }catch(e){}
}
setInterval(pollSanity,600); pollSanity();
async function sanity(action){
  const r=await (await fetch('/api/sanity/start',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify({action})})).json();
  if(!r.ok) toast(r.error||'could not start'); }
async function clearLogs(){ await fetch('/api/sanity/clear',{method:'POST',
  headers:{'Content-Type':'application/json'},body:'{}'}); term.innerHTML=''; SINCE=0; }
function copyLogs(){ const txt=[...term.children].map(d=>d.textContent).join('\n');
  const done=()=>{cpbtn.textContent='copied'; setTimeout(()=>cpbtn.textContent='copy full logs',1200);};
  navigator.clipboard.writeText(txt).then(done,()=>{ const ta=document.createElement('textarea');
    ta.value=txt; document.body.appendChild(ta); ta.select(); document.execCommand('copy'); ta.remove(); done(); }); }

function showModal(html){ mbox.innerHTML=html; modal.classList.add('on'); }
function closeModal(){ modal.classList.remove('on'); }
modal.onclick=e=>{ if(e.target===modal) closeModal(); };
document.addEventListener('keydown',e=>{ if(e.key==='Escape') closeModal(); });
function esc(t){return String(t).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
function toast(msg){ showModal(`<h4>Heads up</h4><p>${esc(msg)}</p>
  <div class=mact><button class=blue onclick=closeModal()>OK</button></div>`); }
function confirmCamLatency(){ showModal(`<h4>Measure camera latency</h4>
  <p><b>This closes and reopens the gripper.</b> It watches a wrist camera for the first
  frame that changes, giving action-to-pixel latency. Needs the cameras streaming
  (i.e. run it from a live <code>swoosh-collect</code> session).</p>
  <div class=mact><button onclick=closeModal()>Cancel</button>
  <button class=red onclick="closeModal();sanity('camlatency')">Measure</button></div>`); }
function confirmFrame(){ showModal(`<h4>Verify the 45&deg; frame</h4>
  <p><b>This moves the arm.</b> It jogs 20&nbsp;mm along each world axis (forward, left, up) and
  reports which way the end effector actually went. Clear the workspace first.<br><br>
  This is the one check that catches a sign error in the mount rotation: the matrix is a valid
  rotation either way, so only the robot can settle it.</p>
  <div class=mact><button onclick=closeModal()>Cancel</button>
  <button class=red onclick="closeModal();sanity('frame')">Jog the arm</button></div>`); }
function newCampaign(){ showModal(`<h4>New campaign</h4>
  <p>Creates a folder under <code>campaigns/</code> (gitignored) and switches to it.</p>
  <div class=mrow><label>name</label><input id=cnew placeholder="lego-pick-place" style="flex:2"></div>
  <div class=mact><button onclick=closeModal()>Cancel</button>
  <button class=blue onclick=doNewCampaign()>Create</button></div>`);
  const i=document.getElementById('cnew'); i.focus();
  i.onkeydown=e=>{ if(e.key==='Enter') doNewCampaign(); }; }
async function doNewCampaign(){
  const name=(document.getElementById('cnew').value||'').trim(); if(!name) return;
  const r=await (await fetch('/api/campaign/new',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify({name})})).json();
  if(!r.ok){toast(r.error);return} closeModal(); SEL=null; goLive(); loadCampaign(); }
async function nameCameras(){
  const r=await (await fetch('/api/sanity/cameras')).json();
  if(!r.found.length){ toast('No cameras found on the bus. Plugged in? The name filter is in conf/collect.yaml.'); return; }
  const opts=lbl=>r.expected.map(e=>`<option ${e===lbl?'selected':''}>${e}</option>`).join('');
  showModal(`<h4>Name the cameras</h4>
    <p>Pinned by USB port path, so labels survive reboots and replugs. ${r.found.length} found.</p>
    ${r.found.map(c=>`<div class=mrow>
      <label>${c.device}<br><span class=sub style="font-size:11px">usb ${c.usb_port}</span></label>
      <select data-port="${c.usb_port}"><option value="">- skip -</option>${opts(r.assigned[c.usb_port]||'')}</select>
    </div>`).join('')}
    <div class=mact><button onclick=closeModal()>Cancel</button>
    <button class=blue onclick=saveCameras()>Save</button></div>`); }
async function saveCameras(){
  const mapping={}; mbox.querySelectorAll('select[data-port]').forEach(s=>{ if(s.value) mapping[s.dataset.port]=s.value; });
  const r=await (await fetch('/api/sanity/cameras',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify({mapping})})).json();
  closeModal(); if(!r.ok) toast(r.error||'save failed'); }

(async ()=>{ try{
  const m=await import('/vendor/arm3d.js');
  ARM=await m.createArmView(document.getElementById('arm3d'),{mountTiltDeg:45});
  const ws=await (await fetch('/api/workspace')).json();
  if(ws.box && ws.box.x){ ARM.setWorkspace(ws.box); buildWsFields(ws.box); }
}catch(e){ document.getElementById('arm3d').innerHTML=
  '<div class=sub style="padding:8px">3D view unavailable: '+e+'</div>'; } })();
function buildWsFields(box){
  const rows=[['x',0,'x min'],['x',1,'x max'],['y',0,'y min'],['y',1,'y max'],
              ['z',0,'z min'],['z',1,'z max']];
  wsfields.innerHTML=rows.map(([ax,i,lb])=>
    `<label>${lb}<input type=number step=10 data-ax="${ax}" data-i="${i}"
       value="${box[ax][i]}"></label>`).join('');
  wsfields.querySelectorAll('input').forEach(inp=>{
    inp.onchange=async()=>{
      const b={x:[0,0],y:[0,0],z:[0,0]};
      wsfields.querySelectorAll('input').forEach(x=>{
        b[x.dataset.ax][+x.dataset.i]=parseFloat(x.value); });
      ARM.setWorkspace(b);
      const r=await (await fetch('/api/workspace',{method:'POST',
        headers:{'Content-Type':'application/json'},body:JSON.stringify({box:b})})).json();
      if(!r.ok) toast(r.error||'could not save workspace');
    };
  });
}
setTicker('LIVE','green');
async function loadRoot(){
  try{ const r=await (await fetch('/api/root')).json();
    rootpath.textContent='campaigns are written to  '+r.resolved; }catch(e){}
}
function chooseRoot(){
  fetch('/api/root').then(r=>r.json()).then(cur=>{
    showModal(`<h4>Choose root folder</h4>
      <p>Where campaigns are written. A relative path resolves against the repo; an
      absolute path (an external drive, say) is used as-is. Created if missing.</p>
      <div class=mrow><label>path</label>
        <input id=rootin style="flex:2" value="${cur.configured}"></div>
      <div class=sub style="font-size:11px">currently: ${cur.resolved}</div>
      <div class=mact><button onclick=closeModal()>Cancel</button>
      <button class=blue onclick=saveRoot()>Set</button></div>`);
    const i=document.getElementById('rootin'); i.focus();
    i.onkeydown=e=>{ if(e.key==='Enter') saveRoot(); };
  });
}
async function saveRoot(){
  const path=(document.getElementById('rootin').value||'').trim(); if(!path) return;
  const r=await (await fetch('/api/root',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify({path})})).json();
  if(!r.ok){ toast(r.error); return; }
  closeModal(); loadRoot(); loadCampaign();
}
loadRoot();
loadCampaign(); setInterval(loadCampaign,4000);
</script></body></html>"""

PAGE = (PAGE.replace("__SPLIT_JS__", _vendor("split.min.js"))
            .replace("__NOUI_JS__", _vendor("nouislider.min.js"))
            .replace("__NOUI_CSS__", _vendor("nouislider.min.css")))
