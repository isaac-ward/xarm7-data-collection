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


def _clamp_to_box(xyz: np.ndarray, box: dict,
                  prev: np.ndarray | None = None) -> tuple[np.ndarray, bool]:
    """Clamp the target into the box WITHOUT ever moving the arm by itself.

    The obvious np.clip is wrong when the arm starts outside the box: it returns the
    wall, and the servo then drives the whole excursion at once. That is a real jolt --
    re-home left the EE at world y = -414.8 with the box allowing y >= -150, and the
    first tick after streaming resumed commanded a 265 mm step.

    Rate-limiting the approach was the next wrong answer: it still moves the arm
    hundreds of millimetres with nobody touching the sticks.

    So, per axis: never allow the violation to get WORSE, and never overshoot the wall
    coming back. An axis already outside stays where it is until the operator drives it
    in, and can never be pushed further out. Inside the box this is exactly np.clip.
    """
    lo = np.array([box["x"][0], box["y"][0], box["z"][0]], dtype=np.float64)
    hi = np.array([box["x"][1], box["y"][1], box["z"][1]], dtype=np.float64)
    if prev is None:
        return np.clip(xyz, lo, hi), bool(np.any(np.clip(xyz, lo, hi) != xyz))
    out = xyz.astype(np.float64).copy()
    for i in range(3):
        if out[i] < lo[i]:                      # below the floor
            out[i] = min(max(out[i], prev[i]), lo[i])
        elif out[i] > hi[i]:                    # above the ceiling
            out[i] = max(min(out[i], prev[i]), hi[i])
    return out, bool(np.any(out != xyz))


def main() -> int:
    ap = argparse.ArgumentParser(description="Xbox teleop + data collection, right arm.")
    ap.add_argument("--campaign", default=None,
                    help="campaign name; defaults to the most recently created one, "
                         "because starting in whichever campaign was named first is "
                         "how runs end up in the wrong place (see swoosh-campaign)")
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
    if args.campaign:
        campaign = Campaign.open(args.campaign, cfg)
    else:
        campaign = Campaign.latest(cfg)
        if campaign is None:
            print("[collect] no campaigns yet. Create one with:\n"
                  '  swoosh-campaign new "my campaign"', file=sys.stderr)
            return 2
        print(f"[collect] campaign: {campaign.path.name} (most recent; "
              f"pass --campaign to choose another)", flush=True)

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
    # Camera frames were published only inside the 15 Hz status branch, so a preview
    # could never be fresher than 66 ms no matter how fast the browser asked. Publishing
    # is just a reference swap under a short lock, so it can run at the cameras' own
    # rate without touching the SSE rate.
    frame_dt = 1.0 / max(float(get(cfg, "dashboard.preview_fps", 25.0)), 1.0)
    last_frame_pub = 0.0
    state_every = max(1, int(rate / 50))  # poll the arm at ~50 Hz; the SDK read isn't free

    from scipy.spatial.transform import Rotation as R

    mode_warned = False
    last_recover = 0.0
    slew_warned = False
    preview_rig = None      # cameras run in preview whenever no run is recording

    RigCls = SimulatedCameraRig if args.simulate else CameraRig

    def start_preview():
        """Live camera feeds with no recording, so the operator can see all four (and
        check which is which) before pressing A. Cameras used to exist only for the
        duration of a run, which left the dashboard's four panes permanently black."""
        if args.no_cameras:
            return None
        try:
            r = RigCls(cfg, server.campaign.path / ".preview", record=False)
            # preview records no timestamps, so this origin is unused
            r.start(time.monotonic(), blocking=False)
            return r
        except Exception as exc:
            print(f"[collect] camera preview unavailable: {exc}", file=sys.stderr,
                  flush=True)
            return None
    R_home_world = R.from_matrix(
        R_WORLD_FROM_BASE @ R.from_euler("xyz", rpy_home, degrees=True).as_matrix()
    )

    def warn_if_outside_box(where: str, w) -> None:
        """A home pose outside the safety box is a configuration error, not something
        to silently correct: the clamp would drag the arm to the wall as soon as
        streaming starts."""
        names = "xyz"
        bad = []
        for i, ax in enumerate(names):
            lo, hi = float(box[ax][0]), float(box[ax][1])
            v = float(w[i])
            if not (lo <= v <= hi):
                # Enough precision that the message cannot read as nonsense. At 0 dp a
                # pose 0.04 mm over the wall printed as "x=650 outside [350, 650]",
                # which looks like a bug in the check rather than a real excursion.
                over = (v - hi) if v > hi else (lo - v)
                bad.append(f"{ax}={v:.2f} outside [{lo:.0f}, {hi:.0f}] "
                           f"by {over:.2f} mm")
        if bad:
            print(f"[collect] WARNING: {where} is OUTSIDE the workspace box "
                  f"({'; '.join(bad)}). The arm will HOLD, not move itself -- jog back "
                  f"in or fix the box; the box and the home pose disagree.",
                  file=sys.stderr, flush=True)
        return bool(bad)

    # Seed from the real pose. Left as None, the first tick fell through to a plain
    # np.clip and jumped straight to the wall -- which is the exact bug this guards.
    prev_target = target_world.copy()
    warn_if_outside_box("the starting pose", target_world)

    live = LiveState()
    # Everything interesting the loop has to say goes to BOTH stdout and the
    # dashboard's terminal panel. stdout alone is not good enough: the collector is
    # usually started detached or under compose, where nobody is watching its console,
    # and the operator's only window into the run is the browser.
    def LOG(msg: str, err: bool = False) -> None:
        print(msg, file=sys.stderr if err else sys.stdout, flush=True)
        try:
            server.sanity.log(msg)
        except Exception:
            pass          # the dashboard is a convenience; never let it break the loop

    # SIMULATED is now a read-only fact about how the process was started, not a mode
    # anything can switch into. The dashboard used to carry a "preview mode" button
    # that flipped this flag -- which changed the ticker and nothing else, since the
    # arm, pad and cameras are already-constructed real objects by this point. A
    # control that misreports the mode without changing it is worse than no control.
    live.preview = bool(args.simulate)
    assert not (live.preview and not args.simulate), (
        "SIMULATED must only ever be set by --simulate")
    server = DashboardServer(
        cfg, campaign, live, port=int(get(cfg, "dashboard.port", 8770))
    )
    server.arm_ref["arm"] = arm          # lets the sanity buttons talk to the arm
    url = server.start(open_browser=not args.no_browser)
    print(f"[collect] dashboard: {url}", flush=True)
    print("[collect] A start   B stop   X clear errors   Y re-home   Start quit",
          flush=True)

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
    arm.set_clock_origin(t_loop0)     # so the arm's own stamps share this clock
    next_tick = t_loop0
    preview_rig = start_preview()
    LOG("[collect] A start   B stop   X clear errors   Y re-home   Start quit")
    LOG(f"[collect] arm {arm.ip}   campaign {server.campaign.path.name}   "
        f"box x{list(map(int, box['x']))} y{list(map(int, box['y']))} "
        f"z{list(map(int, box['z']))}")
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

            reanchored_tick = False
            snap = pad.snapshot(t)

            # Every physical press is echoed, bound or not: an operator pressing an
            # unbound button used to get complete silence, indistinguishable from a
            # dead pad. This is logging only -- the arm is driven by drain_events().
            for _tp, ecode, logical in pad.drain_presses():
                if logical:
                    LOG(f"[pad] {ecode} -> {logical}")
                else:
                    LOG(f"[pad] {ecode} (no binding)")

            # Physical pad presses and on-screen dashboard presses are the same
            # thing as far as the loop is concerned.
            dash = live.drain_buttons()
            for button in dash:
                LOG(f"[dash] {button}")
            events = [b for _t, b in pad.drain_events()] + dash
            for button in events:
                if button == "quit":
                    hint = "quit pressed"
                    LOG("[collect] Start -> quitting, arm holds position")
                    STOP.set()

                elif button == "start_episode" and rec is None and inflight:
                    hint = (f"still processing {sorted(inflight)[0]} - "
                            f"wait for it to finish before starting another run")
                    LOG(f"[collect] A ignored: {sorted(inflight)[0]} still processing")

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
                    ok_all = report(checks)
                    for c in checks:
                        LOG(f"  [{'ok ' if c.ok else 'FAIL'}] {c.name}"
                            + (f" -- {c.detail}" if c.detail else ""))
                    if not ok_all:
                        hint = ("pre-run checks failed: "
                                + next(c.name for c in checks if not c.ok))
                        LOG("[collect] A REFUSED: pre-run checks failed", err=True)
                        next_tick = time.monotonic()
                        continue
                    # run.json is written immediately with status `recording`, so the
                    # dashboard lists the run the instant A is pressed.
                    # via the server, so a campaign switched in the dashboard
                    # actually takes effect here
                    run_dir = server.campaign.new_run_dir()
                    rec = RunRecorder(run_dir, t0=t_loop0)
                    if not args.no_cameras:
                        # Release the devices from preview first -- a camera cannot be
                        # opened twice, so the recording rig would fail outright.
                        if preview_rig is not None:
                            preview_rig.stop()
                            preview_rig = None
                        rig = RigCls(cfg, run_dir / "video")
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
                    LOG(f"[collect] A -> RECORDING {run_dir.name}")

                elif button == "stop_episode" and rec is not None:
                    # Mark it stopped BEFORE any of the teardown work, so the tag
                    # flips the moment B is pressed rather than after the cameras
                    # have closed and the summariser has spawned.
                    if run_dir is not None:
                        set_run_status(run_dir, "stopped")
                    live.bump()
                    LOG("[collect] B -> stopped, closing streams")
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
                    LOG(f"[collect] B -> saved {run_dir.name}, "
                        f"{meta['duration_s']:.1f}s, now processing")
                    rec, rig, run_dir, run_started = None, None, None, None
                    preview_rig = start_preview()   # feeds go live again

                elif button == "clear_errors" and rec is not None:
                    # Re-home refuses mid-run; this must too. It re-seeds the target
                    # AND resets yaw_world to 0, so pressing it during a recording
                    # makes action.commanded_pose_world's yaw jump to 0 with no
                    # physical change -- an unflagged discontinuity in the action.
                    hint = "stop the run (B) before clearing errors"
                    LOG("[collect] X refused: stop the run (B) first -- clearing "
                        "re-seeds the target mid-episode", err=True)

                elif button == "clear_errors":
                    try:
                        err, warn = arm.clear_errors()
                        target_world, base_pose = arm.current_pose_world()
                        rpy_home = list(base_pose[3:6])
                        R_home_world = R.from_matrix(
                            R_WORLD_FROM_BASE
                            @ R.from_euler("xyz", rpy_home, degrees=True).as_matrix()
                        )
                        yaw_world = 0.0
                        since_ok = None
                        mode_warned = False
                        prev_target = target_world.copy()
                        next_tick = time.monotonic()
                        hint = (f"cleared error {err} / warn {warn}" if (err or warn)
                                else "nothing to clear - arm was healthy")
                        LOG(f"[collect] X: cleared error={err} warn={warn}, servo "
                            f"mode re-entered, target re-anchored")
                    except Exception as exc:
                        hint = f"clear FAILED: {exc}"
                        LOG(f"[collect] clear FAILED: {exc}", err=True)

                elif button == "rehome":
                    if rec is not None:
                        hint = "stop the run (B) before re-homing"
                    else:
                        hint = "re-homing"
                        LOG("[collect] re-homing ...")
                        try:
                            # Y is deliberate and the operator's hand is on the pad,
                            # so the full startup countdown is dead time here. Startup
                            # homing keeps the 3 s banner, where nobody may be watching.
                            arm.go_home(countdown=int(
                                get(cfg, "arm.rehome_countdown_s", 1)))
                            arm.start_streaming()
                            target_world, base_pose = arm.current_pose_world()
                            rpy_home = list(base_pose[3:6])
                            R_home_world = R.from_matrix(
                                R_WORLD_FROM_BASE
                                @ R.from_euler("xyz", rpy_home, degrees=True).as_matrix()
                            )
                            yaw_world = 0.0
                            since_ok = None
                            prev_target = target_world.copy()
                            next_tick = time.monotonic()
                            if warn_if_outside_box("the home pose", target_world):
                                hint = ("re-homed, but home is OUTSIDE the workspace "
                                        "box - easing to the edge")
                            else:
                                hint = "re-homed"
                            LOG(f"[collect] re-homed -> world "
                                f"[{target_world[0]:.0f} {target_world[1]:.0f} "
                                f"{target_world[2]:.0f}] mm, streaming resumed")
                        except Exception as exc:
                            # The enclosing try has no `except`, only `finally`, so a
                            # raise here used to tear down the arm, the recorder and
                            # the dashboard. Report it and keep the session alive.
                            hint = f"re-home FAILED: {exc}"
                            LOG(f"[collect] re-home FAILED: {exc}", err=True)

            # --- integrate the sticks into the desired pose -------------------
            target_world = target_world + np.array(
                [
                    snap.move_x * v_mm_s * dt,
                    snap.move_y * v_mm_s * dt,
                    snap.height * v_mm_s * dt,
                ]
            )
            # `prev_target` is what makes the clamp incapable of moving the arm on its
            # own: an axis outside the box holds until the OPERATOR drives it in.
            target_world, clamped = _clamp_to_box(target_world, box, prev_target)
            outside = float(np.linalg.norm(
                target_world - np.clip(
                    target_world,
                    [box["x"][0], box["y"][0], box["z"][0]],
                    [box["x"][1], box["y"][1], box["z"][1]])))
            if outside > 1.0:
                if not slew_warned:
                    LOG(f"[collect] pose is {outside:.0f} mm OUTSIDE the workspace box. "
                        f"Holding position -- jog back in, or fix the box. The arm will "
                        f"not move itself.", err=True)
                    slew_warned = True
                hint = f"{outside:.0f} mm outside the box - jog back in"
            else:
                slew_warned = False
            prev_target = target_world.copy()
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
            # A servo command sent in mode 0 is ignored but STILL returns 0 (see
            # arm.start_streaming), so `code` cannot detect it. Check the mode itself,
            # cheaply, or the loop happily records actions for an arm that is frozen.
            elif tick % state_every == 0 and not arm.streaming_ok():
                # Something took the arm out of servo mode -- in practice a sanity task,
                # which opens a second SDK session and ends with set_state(4). Teleop is
                # dead until we re-enter mode 1, and set_servo_cartesian will keep
                # returning 0 the whole time.
                if rec is not None:
                    # Mid-episode: do NOT auto-recover. A ~1 s recovery with a live
                    # operator produces a startling jump, and the run is already
                    # compromised -- say so and let them stop it deliberately.
                    hint = "ARM NOT IN SERVO MODE - commands ignored - STOP THIS RUN (B)"
                    if not mode_warned:
                        LOG("[collect] WARNING: arm left servo mode MID-RUN; "
                            "commands ignored -- STOP THIS RUN (B)", err=True)
                        mode_warned = True
                elif now - last_recover > 2.0:
                    # Not recording, so recovery is free. Re-anchor the target to where
                    # the arm actually is first, or it would lurch to a stale target the
                    # instant streaming resumes.
                    last_recover = now
                    try:
                        arm.start_streaming()
                        target_world, base_pose = arm.current_pose_world()
                        rpy_home = list(base_pose[3:6])
                        R_home_world = R.from_matrix(
                            R_WORLD_FROM_BASE
                            @ R.from_euler("xyz", rpy_home, degrees=True).as_matrix()
                        )
                        yaw_world = 0.0
                        since_ok = None
                        prev_target = target_world.copy()
                        next_tick = time.monotonic()
                        hint = "servo mode recovered - teleop live again"
                        LOG("[collect] servo mode recovered after an external "
                            "takeover; target re-anchored")
                    except Exception as exc:
                        hint = f"ARM NOT IN SERVO MODE - recovery failed: {exc}"

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
                            prev_target = target_world.copy()
                            since_ok = None
                            reanchors += 1
                            reanchored_tick = True
                            hint = f"target re-anchored ({lag_mm:.0f} mm behind)"
                    else:
                        since_ok = None

            if rec is not None:
                rec.tick(t, dt, tick, code, snap.connected)
                rec.controller(snap.as_row())
                rec.commanded(t, target_world, rpy, yaw_world, snap.gripper, clamped,
                              reanchored=reanchored_tick)
                rec.xarm_command(t, pose_base, code, graw,
                                 gripper_sent=arm.gripper_command_record())
                if last_state is not None and tick % state_every == 0:
                    rec.arm_state(last_state.as_row())

            # Hand the dashboard a snapshot: a dict copy under a short lock. Every
            # encode and socket write happens on server threads.
            if now - last_frame_pub >= frame_dt:
                last_frame_pub = now
                feed = rig if rig is not None else preview_rig
                if feed is not None:
                    for cc in feed.caps:
                        with cc.lock:
                            if cc.latest is not None:
                                live.put_frame(cc.label, cc.latest)

            if now - last_pub >= dash_dt:
                last_pub = now
                live.publish(
                    controller={
                        "move_x": snap.move_x, "move_y": snap.move_y,
                        "height": snap.height, "yaw": snap.yaw,
                        "gripper": snap.gripper, "connected": snap.connected,
                        # Unbound axes are still worth SEEING -- the left trigger is
                        # how you tell a dead pad from a pad you are not pressing.
                        # snap.raw carries every axis pre-shaping.
                        "raw": {k: round(float(v), 4)
                                for k, v in (snap.raw or {}).items()},
                    },
                    proprio={
                        # MEASURED joints, not the controller's plan. The 3D view
                        # renders these, and the two differ by up to ~10 deg while
                        # moving -- which is why the rendered arm did not line up with
                        # the real one. The plan is published alongside for comparison.
                        "joints": ((last_state.joints_real_deg
                                    or last_state.joints_deg) if last_state else []),
                        "joints_planned": (last_state.joints_deg if last_state else []),
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
                        "cameras": [c["label"] for c in
                                    ((rig or preview_rig).status()
                                     if (rig or preview_rig) else [])],
                        # Full per-camera health, not just labels. A camera that failed
                        # to open or died had no way of telling the operator: its pane
                        # simply stayed black, which looks identical to "still warming
                        # up". The dashboard now says which one and why.
                        "camera_health": ((rig or preview_rig).status()
                                          if (rig or preview_rig) else []),
                        "hint": hint,
                        "processing": sorted(inflight),
                        "preview": live.preview,
                    },
                )

    finally:
        if preview_rig is not None:
            preview_rig.stop()
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

    print(f"\n[collect] done. {completed} run(s) in {server.campaign.path}")
    print(f"[collect] export with:  swoosh-export --campaign {server.campaign.path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
