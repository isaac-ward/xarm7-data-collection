"""Teleop + record. The main entry point.

    swoosh-collect --campaign my-campaign

Controls
    left stick        move in the ground-parallel plane   (world XY)
    right stick Y     raise / lower that plane            (world Z)
    right stick X     rotate the end effector about world Z
    right trigger     gripper, proportional
    A                 start an episode
    B                 stop an episode
    Y                 re-home (behind the countdown)
    Start             quit cleanly; the arm holds where it is

No deadman: the arm tracks the sticks whenever this is running. Two things stand in
for one -- the target is clamped to a workspace box, and the sticks are the ONLY thing
that moves it, so a released stick means a stationary arm.

WHY THE TARGET IS INTEGRATED AND NOT SERVOED FROM MEASUREMENT. `target += stick*rate*dt`
keeps the commanded pose independent of tracking error. Re-seeding it from the measured
pose each tick would feed servo lag back into the command and drift downhill under
gravity when the sticks are centred.

The dashboard is a separate localhost server; this loop only ever drops a snapshot into
a shared `LiveState` under a short lock, so rendering can never stall the arm.
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from .arm import RightArm
from .cameras import CameraRig
from .campaign import Campaign, set_run_status
from .config import get, load_config
from .frames import R_BASE_FROM_WORLD, R_WORLD_FROM_BASE, world_to_base
from .preflight import report, run_checks
from .provenance import frame_is_verified, snapshot
from .recorder import RunRecorder
from .server import DashboardServer, LiveState
from .xbox import XboxPad

STOP = threading.Event()


def _process_run_async(run_dir: Path, cfg: dict, inflight: set) -> None:
    """Summarise a finished run in a SEPARATE, nice'd PROCESS.

    Not a thread: encoding a summary video peaks around 8 GB of RAM for a five-minute
    episode, and an OOM in-process would take the collector -- and the live arm
    connection -- down with it. A subprocess can die on its own. `nice` keeps it off
    the control loop's back, and the loop refuses to start a new run while one is
    in flight, so a summariser never competes with an active recording.
    """
    import subprocess
    import sys

    inflight.add(run_dir.name)

    def work() -> None:
        try:
            proc = subprocess.Popen(
                ["nice", "-n", "10", sys.executable, "-m", "swoosh_collect.summarize",
                 "--campaign", run_dir.parent.name, "--run", run_dir.name],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
            )
            _, err = proc.communicate()
            if proc.returncode != 0:
                set_run_status(run_dir, "ready",
                               processing_error=(err or "").strip()[-400:])
            else:
                set_run_status(run_dir, "ready")
        except Exception as exc:
            set_run_status(run_dir, "ready", processing_error=f"{type(exc).__name__}: {exc}")
        finally:
            inflight.discard(run_dir.name)

    threading.Thread(target=work, daemon=True).start()


def _clamp_to_box(xyz: np.ndarray, box: dict) -> tuple[np.ndarray, bool]:
    lo = np.array([box["x"][0], box["y"][0], box["z"][0]], dtype=np.float64)
    hi = np.array([box["x"][1], box["y"][1], box["z"][1]], dtype=np.float64)
    out = np.clip(xyz, lo, hi)
    return out, bool(np.any(out != xyz))


def main() -> int:
    ap = argparse.ArgumentParser(description="Xbox teleop + data collection, right arm.")
    ap.add_argument("--campaign", required=True, help="campaign name (see swoosh-campaign)")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--no-cameras", action="store_true", help="teleop without recording video")
    ap.add_argument("--no-home", action="store_true", help="skip the initial re-home")
    ap.add_argument("--no-browser", action="store_true", help="don't auto-open a browser tab")
    ap.add_argument("--simulate", action="store_true",
                    help="run against SIMULATED hardware -- no arm, pad or cameras needed. "
                         "Exercises the real control loop, recorder, dashboard and export.")
    ap.add_argument("--idle-pad", action="store_true",
                    help="with --simulate, hold the sticks at zero")
    ap.add_argument("--allow-unverified-frame", action="store_true",
                    help="collect even though the 45-degree frame is unverified (NOT advised)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    campaign = Campaign.open(args.campaign, cfg)

    # A wrong frame matrix mislabels every action in the campaign and is unrecoverable
    # afterwards, so refuse by default rather than discover it in training.
    ok_frame, why = frame_is_verified()
    if args.simulate:
        ok_frame, why = True, "simulation - frame verification not applicable"
    if not ok_frame and not args.allow_unverified_frame:
        print(f"[collect] REFUSING TO COLLECT: {why}", file=sys.stderr)
        print("[collect] override with --allow-unverified-frame if you really mean it.",
              file=sys.stderr)
        return 2
    print(f"[collect] frame: {why}", flush=True)

    signal.signal(signal.SIGINT, lambda *_: STOP.set())
    signal.signal(signal.SIGTERM, lambda *_: STOP.set())

    if args.simulate:
        from .simhw import SimulatedArm, SimulatedCameraRig, SimulatedPad
        pad = SimulatedPad(cfg, idle=args.idle_pad)
        pad.start()
        print("[collect] SIMULATED hardware -- no arm, pad or cameras", flush=True)
    else:
        pad = XboxPad(cfg)
    if not args.simulate and not pad.start():
        print("No gamepad found. Plug in the wired Xbox pad and try again.", file=sys.stderr)
        print("  check: ls -l /dev/input/event*", file=sys.stderr)
        return 1
    print(f"[collect] pad: {pad._device.name}", flush=True)

    arm = SimulatedArm(cfg) if args.simulate else RightArm(cfg)
    print(f"[collect] connecting to right arm at {arm.ip} ...", flush=True)
    try:
        arm.connect()
    except Exception as exc:
        print(f"[collect] {exc}", file=sys.stderr)
        pad.stop()
        return 1

    if not args.no_home:
        arm.go_home()
    arm.start_gripper_worker()
    arm.start_streaming()

    # Seed the target from where the arm actually is, so the first tick is a zero move.
    target_world, base_pose = arm.current_pose_world()
    rpy_home = list(base_pose[3:6])
    rpy = list(rpy_home)
    yaw_world = 0.0  # accumulated yaw about world Z, applied on top of the home rpy

    rate = float(get(cfg, "control.rate_hz", 100.0))
    period = 1.0 / max(rate, 1.0)
    v_mm_s = float(get(cfg, "control.translation_rate_mm_s", 120.0))
    yaw_deg_s = float(get(cfg, "control.yaw_rate_deg_s", 60.0))
    box = dict(get(cfg, "control.workspace_box_mm"))
    dash_dt = 1.0 / max(float(get(cfg, "dashboard.refresh_hz", 15.0)), 1.0)
    state_every = max(1, int(rate / 50))  # poll the arm at ~50 Hz; the SDK read isn't free

    from scipy.spatial.transform import Rotation as R

    R_home_world = R.from_matrix(
        R_WORLD_FROM_BASE @ R.from_euler("xyz", rpy_home, degrees=True).as_matrix()
    )

    live = LiveState()
    live.preview = bool(args.simulate)     # the ticker says SIMULATED
    server = DashboardServer(
        cfg, campaign, live, port=int(get(cfg, "dashboard.port", 8770))
    )
    server.arm_ref["arm"] = arm          # lets the sanity buttons talk to the arm
    url = server.start(open_browser=not args.no_browser)
    print(f"[collect] dashboard: {url}", flush=True)
    print("[collect] A start   B stop   Y re-home   Start quit", flush=True)

    rec: RunRecorder | None = None
    rig: CameraRig | None = None
    run_dir: Path | None = None
    run_started: float | None = None
    last_state: Any = None
    last_pub = 0.0
    last_now: float | None = None
    since_ok: float | None = None
    reanchors = 0
    inflight: set[str] = set()
    reanchor_mm = float(get(cfg, "control.reanchor_mm", 30.0))
    reanchor_after_s = float(get(cfg, "control.reanchor_after_s", 0.2))
    hint = ""
    completed = len([r for r in campaign.runs() if r.complete])

    t_loop0 = time.monotonic()
    next_tick = t_loop0
    tick = 0

    try:
        while not STOP.is_set():
            now = time.monotonic()
            if now < next_tick:
                time.sleep(min(0.002, next_tick - now))
                continue
            # MEASURED dt, clamped. A fixed `period` here meant that after any stall
            # the loop replayed the backlog with a full 10 ms of stick each, stepping
            # the target by up to 120 mm in a few milliseconds.
            dt = period if last_now is None else min(now - last_now, 5.0 * period)
            last_now = now
            next_tick = max(next_tick + period, now - period)
            tick += 1
            t = now - t_loop0

            nb = live.take_workspace()      # live edits from the dashboard
            if nb:
                box = nb
                hint = "workspace box updated"

            snap = pad.snapshot(t)

            # Physical pad presses and on-screen dashboard presses are the same
            # thing as far as the loop is concerned.
            events = [b for _t, b in pad.drain_events()] + live.drain_buttons()
            for button in events:
                if button == "quit":
                    hint = "quit pressed"
                    STOP.set()

                elif button == "start_episode" and rec is None and inflight:
                    hint = (f"still processing {sorted(inflight)[0]} - "
                            f"wait for it to finish before starting another run")

                elif button == "start_episode" and rec is None:
                    # Every episode is gated on the same cheap checks -- a one-shot
                    # campaign cannot afford one recorded with three cameras or an
                    # unverified frame. Printed with ticks/crosses; nothing starts if
                    # any check fails.
                    checks = run_checks(cfg, arm, pad,
                                        not args.no_cameras and not args.simulate,
                                        inflight, last_state, simulate=args.simulate)
                    live.set_checks([{"name": c.name, "ok": c.ok, "detail": c.detail}
                                     for c in checks])
                    if not report(checks):
                        hint = ("pre-run checks failed: "
                                + next(c.name for c in checks if not c.ok))
                        next_tick = time.monotonic()
                        continue
                    # run.json is written immediately with status `recording`, so the
                    # dashboard lists the run the instant A is pressed.
                    run_dir = campaign.new_run_dir()
                    rec = RunRecorder(run_dir, t0=t_loop0)
                    if not args.no_cameras:
                        rig = (SimulatedCameraRig if args.simulate else CameraRig)(
                            cfg, run_dir / "video")
                        try:
                            # ONE CLOCK ORIGIN for every stream: t_loop0, not `now`.
                            # Passing the A-press time here gave cameras a different
                            # origin from the JSONL rows and silently misaligned every
                            # episode by the process uptime -- the exact lego_assemblies
                            # failure. Non-blocking: the 0.25 s per-camera USB stagger
                            # used to run inside this loop and stalled the servo for ~1 s.
                            rig.start(t_loop0, blocking=False)
                        except Exception as exc:
                            hint = f"cameras: {exc}"
                            rig = None
                    run_started = now
                    # Blocking work just happened; don't let the loop burst-catch-up.
                    next_tick = time.monotonic()
                    hint = f"recording {run_dir.name}"

                elif button == "stop_episode" and rec is not None:
                    if rig is not None:
                        rig.stop()
                        rig.write_timestamps()
                    meta = rec.close("b_button", {
                        "cameras": rig.status() if rig else [],
                        "provenance": snapshot(cfg, arm),
                    })
                    next_tick = time.monotonic()
                    completed += 1
                    _process_run_async(run_dir, cfg, inflight)  # -> `processing` then `ready`
                    hint = f"saved {run_dir.name} ({meta['duration_s']:.1f}s) - processing"
                    rec, rig, run_dir, run_started = None, None, None, None

                elif button == "rehome":
                    if rec is not None:
                        hint = "stop the run (B) before re-homing"
                    else:
                        hint = "re-homing"
                        arm.go_home()
                        arm.start_streaming()
                        target_world, base_pose = arm.current_pose_world()
                        rpy_home = list(base_pose[3:6])
                        R_home_world = R.from_matrix(
                            R_WORLD_FROM_BASE
                            @ R.from_euler("xyz", rpy_home, degrees=True).as_matrix()
                        )
                        yaw_world = 0.0
                        next_tick = time.monotonic()
                        hint = "re-homed"

            # --- integrate the sticks into the desired pose -------------------
            target_world = target_world + np.array(
                [
                    snap.move_x * v_mm_s * dt,
                    snap.move_y * v_mm_s * dt,
                    snap.height * v_mm_s * dt,
                ]
            )
            target_world, clamped = _clamp_to_box(target_world, box)
            yaw_world += snap.yaw * yaw_deg_s * dt

            # Yaw is about the WORLD z axis, so it composes on the left in the world
            # frame and is then expressed back in the base frame the SDK expects.
            # Composing it into the base euler directly would rotate about the arm's
            # tilted z and be wrong by the 45-degree mount angle.
            R_target_base = R.from_matrix(
                R_BASE_FROM_WORLD
                @ (R.from_euler("z", yaw_world, degrees=True) * R_home_world).as_matrix()
            )
            rpy = R_target_base.as_euler("xyz", degrees=True).tolist()

            pose_base = [*world_to_base(target_world).tolist(), *rpy]
            code = arm.servo_pose(pose_base)
            graw = arm.set_gripper(snap.gripper)
            if code == 1:
                hint = "controller HAS_ERROR - stop and run swoosh-sanity"

            if tick % state_every == 0:
                last_state = arm.read_state(t)
                # If the arm stops following (unreachable target, latched error, wall),
                # the integrated target would keep running away while the arm is
                # frozen -- recording actions that did nothing. Re-anchor instead.
                if last_state.pose_world_xyz:
                    lag_mm = float(np.linalg.norm(
                        target_world - np.array(last_state.pose_world_xyz)))
                    if lag_mm > reanchor_mm:
                        since_ok = since_ok or now
                        if now - since_ok > reanchor_after_s:
                            target_world = np.array(last_state.pose_world_xyz)
                            since_ok = None
                            reanchors += 1
                            hint = f"target re-anchored ({lag_mm:.0f} mm behind)"
                    else:
                        since_ok = None

            if rec is not None:
                rec.tick(t, dt, tick, code, snap.connected)
                rec.controller(snap.as_row())
                rec.commanded(t, target_world, rpy, yaw_world, snap.gripper, clamped)
                rec.xarm_command(t, pose_base, code, graw)
                if last_state is not None and tick % state_every == 0:
                    rec.arm_state(last_state.as_row())

            # Hand the dashboard a snapshot: a dict copy under a short lock. Every
            # encode and socket write happens on server threads.
            if now - last_pub >= dash_dt:
                last_pub = now
                live.publish(
                    controller={
                        "move_x": snap.move_x, "move_y": snap.move_y,
                        "height": snap.height, "yaw": snap.yaw,
                        "gripper": snap.gripper, "connected": snap.connected,
                    },
                    proprio={
                        "joints": (last_state.joints_deg if last_state else []),
                        "world": (last_state.pose_world_xyz if last_state else []),
                        "gripper": (last_state.gripper_pos if last_state else None),
                        "error": (last_state.error_code if last_state else 0),
                        # the DESIRED pose, drawn in the 3D view as a sphere + arrow
                        "target": [float(v) for v in target_world],
                        "target_yaw": float(yaw_world),
                    },
                    status={
                        "recording": rec is not None,
                        "run_index": int(run_dir.name.split("_")[1]) if run_dir else None,
                        "run_name": run_dir.name if run_dir else None,
                        "elapsed": (now - run_started) if run_started else 0.0,
                        "cameras": [c["label"] for c in (rig.status() if rig else [])],
                        "hint": hint,
                        "processing": sorted(inflight),
                        "preview": live.preview,
                    },
                )
                if rig is not None:
                    for cc in rig.caps:
                        with cc.lock:
                            if cc.latest is not None:
                                live.put_frame(cc.label, cc.latest)
    finally:
        if rec is not None:
            if rig is not None:
                rig.stop()
                rig.write_timestamps()
            rec.close("signal", {"cameras": rig.status() if rig else [],
                                 "provenance": snapshot(cfg, arm)})
            if run_dir is not None:
                _process_run_async(run_dir, cfg, inflight)
        pad.stop()
        arm.shutdown()
        # Give a still-running summariser a moment before the process exits.
        time.sleep(0.3)
        server.stop()

    print(f"\n[collect] done. {completed} run(s) in {campaign.path}")
    print(f"[collect] export with:  swoosh-export --campaign {campaign.path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
