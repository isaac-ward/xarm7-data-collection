"""Raw streams -> LeRobot v2.1 at a fixed rate. The dataset a world model trains on.

WHAT THE ACTION IS. `action` is the XBOX CONTROLLER INPUT -- what the operator did --
because that is the causal signal and the only one that exists before the robot
responds. The commanded pose and the literal SDK arguments are ALSO exported, as
`action.commanded_*` and `action.xarm_*`, but kept as separate columns so a consumer
chooses deliberately rather than inheriting whatever we happened to merge.

This is a direct lesson from `swoosh-data/lego_assemblies`, where the published `action`
turned out to be raw VR controller pose in a per-session room frame that nothing else
referenced. Recovering the real command took a full re-derivation from raw logs. Here
the raw logs are kept, all three action families are published side by side, and each
is labelled with its frame and units.

RESAMPLING IS NEAREST-IN-TIME, NEVER INTERPOLATED. Interpolating a command invents one
that was never issued. Every stream carries its own timestamps on one monotonic clock,
so each output row takes the nearest sample of each stream and records how far away it
was in `sync_error_s` -- if that number is ever large, the dataset says so out loud
instead of hiding it.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .campaign import Campaign
from .config import get, load_config
from .summarize import _read_jsonl

CHUNK = 1000


def _series(rows: list[dict], key: str, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """(times, values) for one vector field, dropping rows that lack it."""
    ts, vs = [], []
    for r in rows:
        v = r.get(key)
        if v is None:
            continue
        arr = np.asarray(v, dtype=np.float64).ravel()
        if arr.size != dim:
            continue
        ts.append(float(r["t"]))
        vs.append(arr)
    if not ts:
        return np.zeros(0), np.zeros((0, dim))
    return np.asarray(ts), np.stack(vs)


def _pick(times: np.ndarray, values: np.ndarray, grid: np.ndarray, dim: int):
    """Nearest-in-time resample. Returns (values_on_grid, abs time error per row)."""
    # Mismatched lengths mean `times` came from a different row subset than `values`,
    # which resamples one stream onto another's clock and shifts it silently. That bug
    # shipped once; make it loud.
    if len(times) != len(values):
        raise ValueError(
            f"_pick got {len(times)} timestamps for {len(values)} values -- these come "
            f"from different row subsets and the result would be time-shifted"
        )
    if len(times) == 0:
        return np.full((len(grid), dim), np.nan), np.full(len(grid), np.nan)
    idx = np.searchsorted(times, grid)
    idx = np.clip(idx, 0, len(times) - 1)
    prev = np.clip(idx - 1, 0, len(times) - 1)
    take_prev = np.abs(times[prev] - grid) < np.abs(times[idx] - grid)
    sel = np.where(take_prev, prev, idx)
    return values[sel], np.abs(times[sel] - grid)


def _reencode_on_grid(src: Path, dst: Path, idx: np.ndarray, fps: float,
                      quality: int) -> int:
    """Write one output frame per export row, picking source frame idx[i]."""
    import imageio.v2 as imageio

    frames = []
    with imageio.get_reader(str(src)) as rd:
        for f in rd:
            frames.append(f)
    if not frames:
        return 0
    w = imageio.get_writer(str(dst), fps=fps, codec="libx264", quality=quality,
                           macro_block_size=1)
    n = 0
    try:
        for j in idx:
            w.append_data(frames[int(np.clip(j, 0, len(frames) - 1))])
            n += 1
    finally:
        w.close()
    return n


def export_run(run_dir: Path, cfg: dict) -> dict[str, Any] | None:
    """One run -> arrays on the export grid, plus per-camera frame indices."""
    ctl = _read_jsonl(run_dir / "raw" / "controller.jsonl")
    cmd = _read_jsonl(run_dir / "raw" / "commanded.jsonl")
    xac = _read_jsonl(run_dir / "raw" / "xarm_command.jsonl")
    stt = _read_jsonl(run_dir / "raw" / "arm_state.jsonl")
    if not (ctl and stt):
        return None

    fps = float(get(cfg, "recording.export_rate_hz", 30.0))
    cams = []
    unusable = []
    for m in sorted((run_dir / "video").glob("*_frame_times.json")):
        d = json.loads(m.read_text())
        if d.get("t") and (run_dir / "video" / f"{d['label']}.mp4").is_file():
            cams.append({"label": d["label"], "t": np.asarray(d["t"], dtype=np.float64),
                         "stopped_early": bool(d.get("stopped_early")),
                         "stopped_at_s": float(d.get("stopped_at_s", -1.0))})
        else:
            why = "no frame timestamps" if not d.get("t") else "mp4 missing"
            unusable.append(f"{d.get('label', m.name)} ({why})")

    # A camera that never produced a usable stream used to be dropped here in silence,
    # leaving a 3-camera dataset that validate still called complete. The expected set
    # is what we promised the consumer, so a shortfall is a refusal.
    want = list(get(cfg, "cameras.expected_labels", []) or [])
    mapped = set((get(cfg, "cameras.by_usb_path", {}) or {}).values())
    expected = [w for w in want if w in mapped]
    have = {c["label"] for c in cams}
    missing = [w for w in expected if w not in have]
    if missing or unusable:
        print(f"  [SKIP] {run_dir.name}: cameras missing {missing or ''} "
              f"{('unusable: ' + ', '.join(unusable)) if unusable else ''}".strip())
        return None

    # A camera that died part-way capped `hi` below, silently truncating the episode to
    # the death time while still reporting success. Refuse instead.
    died = [c["label"] for c in cams if c["stopped_early"]]
    if died:
        print(f"  [SKIP] {run_dir.name}: camera(s) stopped early mid-run: "
              f"{', '.join(died)} -- episode would be truncated to the failure point")
        return None

    # The grid covers only the span where EVERY stream (cameras included) has data, so
    # no row is padded with a value that never existed.
    lo = max([ctl[0]["t"], stt[0]["t"]] + [float(c["t"][0]) for c in cams])
    hi = min([ctl[-1]["t"], stt[-1]["t"]] + [float(c["t"][-1]) for c in cams])
    # A shared clock origin means every stream starts within a second or so of the
    # others. A large gap means the origins differ -- refuse rather than emit a
    # confidently-wrong dataset, which is how lego_assemblies happened.
    starts = [ctl[0]["t"], stt[0]["t"]] + [float(c["t"][0]) for c in cams]
    if max(starts) - min(starts) > 5.0:
        print(f"  [SKIP] stream starts differ by {max(starts) - min(starts):.1f}s "
              f"-- clock origins disagree, refusing to export")
        return None

    # How much of the recorded run survives the intersection above. The previous
    # dataset threw away 46% of its frames and nobody noticed, because no coverage
    # number was ever printed. Print it on EVERY export, pass or fail.
    raw_span = max(ctl[-1]["t"], stt[-1]["t"]) - min(ctl[0]["t"], stt[0]["t"])
    coverage = ((hi - lo) / raw_span) if raw_span > 0 else 0.0
    lost_s = raw_span - (hi - lo)
    floor = float(get(cfg, "recording.min_coverage", 0.9))
    # The rig opens on a worker with a per-camera USB stagger plus warmup frames, so the
    # video streams legitimately start a second or two after the control loop. That head
    # loss is known and BOUNDED, and on a short run it is a large fraction while on a
    # real episode it is negligible -- so a flat percentage floor is the wrong shape.
    # Budget the expected startup cost, then police loss beyond it.
    allowance_s = float(get(cfg, "recording.coverage_allowance_s", 2.5))
    budget_s = max(allowance_s, (1.0 - floor) * raw_span)
    print(f"  [{run_dir.name}] retained {coverage * 100:.1f}% of {raw_span:.1f}s "
          f"(lost {lost_s:.2f}s, budget {budget_s:.2f}s)")
    if lost_s > budget_s:
        print(f"  [SKIP] {run_dir.name}: {lost_s:.1f}s of a {raw_span:.1f}s run is not "
              f"covered by every stream (budget {budget_s:.1f}s) -- refusing rather "
              f"than emitting a silently truncated episode")
        return None
    if hi - lo < 1.0 / fps:
        return None
    n = int((hi - lo) * fps)
    grid = lo + np.arange(n) / fps

    errs = []
    # --- action: what the OPERATOR did. STAYS the Xbox input, because that is the
    # causal signal -- the thing a person actually did. Now 6 dims: both triggers are
    # carried, the right one as `gripper` (it drives the jaws) and the left one raw.
    # The left trigger is unbound, so it reads zero unless someone binds it -- but
    # recording it means the column exists the day someone does, and an unbound axis
    # reading zero is distinguishable from an axis nobody recorded at all.
    a_t = np.asarray([r["t"] for r in ctl])
    _lt = str(get(cfg, "controller.axes.left_trigger", "ABS_Z"))
    a_v = np.stack([
        np.asarray(
            [r.get(k, 0.0) for k in ("move_x", "move_y", "height", "yaw", "gripper")]
            + [float((r.get("raw") or {}).get(_lt, 0.0) or 0.0)],
            dtype=np.float64)
        for r in ctl
    ])
    action, e = _pick(a_t, a_v, grid, 6); errs.append(e)

    # --- action.commanded: what WE asked the arm for ------------------------
    c_t, c_xyz = _series(cmd, "target_world_xyz_mm", 3)
    _, c_rpy = _series(cmd, "target_rpy_deg", 3)          # BASE-frame rpy, see below
    cg_t = np.asarray([r["t"] for r in cmd]) if cmd else np.zeros(0)
    cg_v = (np.asarray([[r.get("gripper_closure", np.nan),
                         r.get("target_yaw_world_deg", np.nan),
                         1.0 if r.get("clamped_by_workspace") else 0.0] for r in cmd])
            if cmd else np.zeros((0, 3)))
    cmd_xyz, e = _pick(c_t, c_xyz, grid, 3); errs.append(e)
    cmd_rpy, _ = _pick(c_t, c_rpy, grid, 3)
    cmd_aux, _ = _pick(cg_t, cg_v, grid, 3)   # [gripper_closure, yaw_world_deg, clamped]

    # --- action.xarm: the literal SDK arguments -----------------------------
    x_t, x_pose = _series(xac, "set_servo_cartesian_pose_base", 6)
    xarm_pose, e = _pick(x_t, x_pose, grid, 6); errs.append(e)

    # --- observation.state: what the ROBOT reports --------------------------
    # Each read is caught independently in arm.py, so a row can carry a valid pose
    # with joints_deg: []. These three therefore come from DIFFERENT row subsets, and
    # indexing pose_base/pose_world with the joints' time array silently shifted every
    # pose row for the rest of the episode after a single dropped read. Take the times
    # each series actually returns -- which is what the gripper below already did.
    # MEASURED joints, not planned. Confirmed on the first real run: the SDK default
    # (is_real=False, what joints_deg holds) is the CONTROLLER'S PLAN and differs from
    # the servo feedback by up to 9.84 deg (p95 3.59) while moving. An observation that
    # follows the command through a stall is not an observation, so observation.state
    # takes joints_real_deg and the plan ships alongside it as its own feature.
    sr_t, joints_real = _series(stt, "joints_real_deg", 7)
    sp_t, joints_plan = _series(stt, "joints_deg", 7)
    if len(joints_real):
        s_t, joints = sr_t, joints_real
    else:
        print(f"  [{run_dir.name}] no joints_real_deg (run predates it) -- "
              f"observation.state falls back to the PLANNED joints")
        s_t, joints = sp_t, joints_plan
    _unused_s_t, _unused = _series(stt, "joints_deg", 7)
    pb_t, pose_base = _series(stt, "pose_base_mm_deg", 6)
    pw_t, pose_world = _series(stt, "pose_world_xyz_mm", 3)

    # The controller's reported TCP is the PLAN, not a measurement -- the first real
    # run's lag sweep agreed with the command to 0.01 mm at 0 ms lag, which no physical
    # arm does. So derive the pose the honest way: run the MEASURED joints through the
    # MJCF chain (verified against the controller to 2.5 mm). This is the FLANGE, so it
    # differs from the controller's TCP by the constant tool offset, which is recorded
    # in provenance. Both ship; observation.state carries the derived one.
    if len(joints_real):
        from scipy.spatial.transform import Rotation as _R

        from .frames import base_to_world
        from .kinematics import fk_chain
        dv = []
        for row in joints_real:
            T = fk_chain(row)[-1]
            xyz = T[:3, 3] * 1000.0
            # ORIENTATION from the measured joints too. Taking position from the
            # measured joints while leaving rpy from the controller's report would mix
            # a measurement and a plan inside one 6-vector, which is worse than either.
            rpy = _R.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)
            dv.append([*xyz, *rpy, *base_to_world(xyz)])
        derived = np.asarray(dv, dtype=np.float64)     # 3 xyz + 3 rpy + 3 world
        d_base6, d_world = derived[:, :6], derived[:, 6:]
        d_t = sr_t
    else:
        d_base6 = d_world = None
    # The gripper is polled at <=10 Hz and its cached value copied into every 50 Hz
    # row, so resampling it on the ROW time makes a 10 Hz signal masquerade as 50 Hz --
    # exactly lego failure #3. Use each reading's own measurement time and keep one
    # sample per distinct measurement, so the 30 Hz grid interpolates from the truth.
    g_rows = [r for r in stt if r.get("gripper_pos") is not None]
    stamped = [r for r in g_rows
               if r.get("gripper_pos_t") is not None
               and not np.isnan(float(r.get("gripper_pos_t", np.nan)))]
    if stamped:
        seen_t, sg_t_l, sg_v_l = set(), [], []
        for r in stamped:                      # dedupe: one row per real measurement
            gt = float(r["gripper_pos_t"])
            if gt in seen_t:
                continue
            seen_t.add(gt)
            sg_t_l.append(gt)
            sg_v_l.append([float(r["gripper_pos"])])
        order = np.argsort(np.asarray(sg_t_l))
        sg_t = np.asarray(sg_t_l)[order]
        sg_v = np.asarray(sg_v_l)[order]
    else:
        # Runs recorded before gripper_pos_t existed: fall back to the row clock and
        # say so, rather than silently pretending the rate is real.
        print(f"  [{run_dir.name}] gripper has no measurement timestamps; "
              f"falling back to the state-row clock (rate is NOT 50 Hz)")
        sg_t = np.asarray([r["t"] for r in g_rows])
        sg_v = np.asarray([[r.get("gripper_pos", np.nan)] for r in g_rows])
    j, e = _pick(s_t, joints, grid, 7); errs.append(e)
    pb, e = _pick(pb_t, pose_base, grid, 6); errs.append(e)
    pw, e = _pick(pw_t, pose_world, grid, 3); errs.append(e)
    # 6D ROTATION, alongside the euler angles rather than instead of them.
    # WHY: euler angles jump 360 degrees when they wrap, and a quaternion can flip
    # sign while describing the identical rotation. Either way a model sees an
    # enormous change where nothing physically moved -- on lego_assemblies rpy wrapped
    # 702/802 times per arm and quaternions flipped 29 times. The 6D form is just the
    # first two COLUMNS of the rotation matrix: six numbers that vary smoothly, with
    # no wrap and no sign ambiguity. The third column is their cross product, so
    # nothing is lost. Euler stays for anyone who wants to read a number off it.
    rot6 = None
    if len(joints_real):
        from .kinematics import fk_chain as _fk
        R6 = []
        for row in joints_real:
            M = _fk(row)[-1][:3, :3]
            R6.append([*M[:, 0], *M[:, 1]])          # first two columns
        rot6_series = np.asarray(R6, dtype=np.float64)

    if d_base6 is not None:
        # observation.state is now ENTIRELY from the measured joints -- position and
        # orientation, base and world. The controller's own report ships untouched as
        # observation.tcp_reported_base for anyone who wants the plan.
        pb, e = _pick(d_t, d_base6, grid, 6); errs.append(e)
        pw, _ = _pick(d_t, d_world, grid, 3)
        rot6, _ = _pick(d_t, rot6_series, grid, 6)
    gp, _ = _pick(sg_t, sg_v, grid, 1)
    if gp.size and np.all(np.isnan(gp)):
        # _grip_pos starts as NaN and stays NaN if the gripper never reported, and that
        # NaN went straight into observation.state[16] unchecked.
        print(f"  [SKIP] {run_dir.name}: gripper position never reported -- "
              f"observation.state would carry an all-NaN column")
        return None
    state = np.concatenate([j, pb, pw, gp], axis=1)          # 7 + 6 + 3 + 1 = 17

    cam_idx = {}
    for c in cams:
        order = np.arange(len(c["t"]), dtype=np.float64)[:, None]
        picked, e = _pick(c["t"], order, grid, 1)
        cam_idx[c["label"]] = picked[:, 0].astype(np.int64)
        errs.append(e)

    return {
        "n": n,
        "timestamp": (grid - lo).astype(np.float32),
        "action": action.astype(np.float32),
        # WORLD column holds world quantities ONLY. It previously carried base-frame
        # rpy under names roll/pitch/yaw, which is a 45-degree lie hidden in a column
        # name -- exactly the class of mistake that made lego_assemblies unusable.
        "action.commanded_pose_world": np.concatenate(
            [cmd_xyz, cmd_aux[:, 1:2], cmd_aux[:, 0:1]], axis=1).astype(np.float32),   # 5
        # Delta of the commanded world pose per grid step. Row 0 is zero by
        # construction (there is no previous row to difference against) rather than
        # NaN, so a consumer summing the column reconstructs the trajectory exactly.
        "action.delta_commanded_world": np.vstack([
            np.zeros((1, 4), dtype=np.float64),
            np.diff(np.concatenate(
                [cmd_xyz, cmd_aux[:, 1:2]], axis=1), axis=0),
        ]).astype(np.float32),                                                         # 4
        "action.commanded_rpy_base": cmd_rpy.astype(np.float32),                       # 3
        "flags": np.concatenate([
            cmd_aux[:, 2:3],                                   # clamped by workspace
            _pick(np.asarray([r["t"] for r in xac]) if xac else np.zeros(0),
                  np.asarray([[r.get("servo_code") if r.get("servo_code") is not None else -1]
                              for r in xac]) if xac else np.zeros((0, 1)), grid, 1)[0],
            _pick(np.asarray([r["t"] for r in stt]),
                  np.asarray([[r.get("error_code", 0)] for r in stt]), grid, 1)[0],
            _pick(np.asarray([r["t"] for r in ctl]),
                  np.asarray([[1.0 if r.get("connected", True) else 0.0] for r in ctl]),
                  grid, 1)[0],
        ], axis=1).astype(np.float32),                                                 # 4
        # The controller's PLANNED joints, kept beside the measured ones: the gap
        # between them is how hard the arm was failing to follow, which is exactly the
        # signal a world model needs during contact.
        # exactly what the controller reported, untouched, for anyone who wants it
        "observation.flange_rot6_base": (
            rot6.astype(np.float32) if rot6 is not None
            else np.full((n, 6), np.nan, np.float32)),
        "observation.tcp_reported_base": _pick(pb_t, pose_base, grid, 6)[0].astype(
            np.float32),
        "observation.joints_planned_deg": _pick(sp_t, joints_plan, grid, 7)[0].astype(
            np.float32) if len(joints_plan) else np.full((n, 7), np.nan, np.float32),
        "action.xarm_servo_cartesian_base": xarm_pose.astype(np.float32),    # 6
        "observation.state": state.astype(np.float32),                       # 17
        "sync_error_s": np.max(np.stack(errs), axis=0).astype(np.float32),
        "cam_idx": cam_idx,
        "cams": [c["label"] for c in cams],
    }


FEATURES = {
    "action": (6, ["move_x", "move_y", "height", "yaw",
                   "gripper_right_trigger", "left_trigger_unbound"]),
    # Delta of the commanded WORLD pose per grid step. `action` above is stick
    # deflection, which is a RATE -- its meaning depends on control.rate_hz,
    # translation_rate_mm_s, the deadzone and the expo, so the same number means
    # different motion under a different config. This delta is invariant to all of
    # that and is what actually moved the arm. Both ship; choose at training time.
    "action.delta_commanded_world": (4, ["dx_mm", "dy_mm", "dz_mm", "dyaw_deg"]),
    "action.commanded_pose_world": (
        5, ["x_mm", "y_mm", "z_mm", "yaw_world_deg", "gripper_closure"]),
    "action.commanded_rpy_base": (3, ["roll_deg", "pitch_deg", "yaw_deg"]),
    # Without these a world model happily trains on stretches where the action did
    # nothing: clamped at a wall, refused by a latched controller error, or issued
    # while the pad was unplugged.
    "flags": (4, ["clamped_by_workspace", "servo_code", "arm_error_code", "pad_connected"]),
    "action.xarm_servo_cartesian_base": (
        6, ["x_mm", "y_mm", "z_mm", "roll_deg", "pitch_deg", "yaw_deg"]),
    # wrap-free, sign-unambiguous orientation: the first two columns of the flange's
    # rotation matrix in the arm base frame (the third is their cross product)
    "observation.flange_rot6_base": (
        6, ["r11", "r21", "r31", "r12", "r22", "r32"]),
    "observation.tcp_reported_base": (
        6, ["x_mm", "y_mm", "z_mm", "roll_deg", "pitch_deg", "yaw_deg"]),
    "observation.joints_planned_deg": (
        7, [f"joint_{i+1}_planned_deg" for i in range(7)]),
    "observation.state": (17, (
        [f"joint_{i+1}_measured_deg" for i in range(7)]
        + ["flange_base_x_mm", "flange_base_y_mm", "flange_base_z_mm",
           "flange_base_roll_deg", "flange_base_pitch_deg", "flange_base_yaw_deg"]
        + ["flange_world_x_mm", "flange_world_y_mm", "flange_world_z_mm",
           "gripper_pos"])),
    "sync_error_s": (1, ["sync_error_s"]),
}


def main() -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    ap = argparse.ArgumentParser(description="Export a campaign to LeRobot v2.1.")
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--out", default=None, help="output dir (default campaigns/<c>/lerobot)")
    ap.add_argument("--task", default="Pick and place with the right arm.")
    ap.add_argument("--include-failed", action="store_true",
                    help="export runs that failed validation, or that never finished "
                         "recording. NOT advised -- validate stores its verdict so "
                         "this exporter can refuse, and by default it does.")
    args = ap.parse_args()

    cfg = load_config()
    campaign = Campaign.open(args.campaign, cfg)
    out = Path(args.out) if args.out else campaign.path / "lerobot"
    fps = float(get(cfg, "recording.export_rate_hz", 30.0))

    runs = [r for r in campaign.runs() if (r.path / "raw" / "controller.jsonl").is_file()]
    if not runs:
        print(f"no runs with raw data in {campaign.path}")
        return 1

    # validate.py has always written its verdict into run.json "so the exporter can
    # refuse a bad run" -- and the exporter never read it. A run left at
    # status: recording by a crash, or one with failing checks, exported like any
    # other. Refuse by default and say why for each, so a silent omission can never be
    # mistaken for a clean export.
    if not args.include_failed:
        keep, refused = [], []
        for r in runs:
            v = (r.meta or {}).get("validation") or {}
            if r.status != "ready":
                refused.append((r, f"status is {r.status!r}, not 'ready'"))
            elif not r.complete:
                refused.append((r, f"never finished cleanly (stopped_by="
                                   f"{r.stopped_by!r})"))
            elif not v:
                refused.append((r, "never validated -- run: swoosh-validate "
                                   f"--campaign {campaign.path.name}"))
            elif not v.get("ok"):
                bad = [c["check"] for c in v.get("checks", [])
                       if c.get("level") == "fail"]
                refused.append((r, f"failed validation: {', '.join(bad) or 'see run.json'}"))
            else:
                keep.append(r)
        for r, why in refused:
            print(f"  [REFUSED] {r.path.name}: {why}")
        if refused:
            print(f"  {len(refused)} run(s) refused; --include-failed overrides")
        runs = keep
        if not runs:
            print("\nno runs passed. Nothing exported.")
            return 1

    (out / "meta").mkdir(parents=True, exist_ok=True)
    (out / f"data/chunk-{0:03d}").mkdir(parents=True, exist_ok=True)

    episodes, ep_stats, total = [], [], 0
    cam_labels: list[str] = []
    prov_tcp: dict | None = None
    kept = 0
    for r in runs:
        ex = export_run(r.path, cfg)
        if ex is None:
            print(f"  [skip] {r.path.name}: not enough overlapping data")
            continue
        idx = kept
        if prov_tcp is None:
            try:
                prov_tcp = (json.loads((r.path / "run.json").read_text())
                            .get("provenance") or {})
            except Exception:
                prov_tcp = {}
        # Was `cam_labels or ex["cams"]`, which froze the dataset's camera set from
        # whichever episode exported first; a later episode with a different set then
        # wrote videos info.json does not list, or promised ones that do not exist.
        if not cam_labels:
            cam_labels = ex["cams"]
        elif list(ex["cams"]) != list(cam_labels):
            print(f"  [skip] {r.path.name}: cameras {ex['cams']} differ from the "
                  f"dataset's {cam_labels}")
            continue
        n = ex["n"]
        cols = {
            "observation.state": [v.tolist() for v in ex["observation.state"]],
            "observation.joints_planned_deg":
                [v.tolist() for v in ex["observation.joints_planned_deg"]],
            "observation.tcp_reported_base":
                [v.tolist() for v in ex["observation.tcp_reported_base"]],
            "observation.flange_rot6_base":
                [v.tolist() for v in ex["observation.flange_rot6_base"]],
            "action": [v.tolist() for v in ex["action"]],
            "action.commanded_pose_world": [v.tolist() for v in ex["action.commanded_pose_world"]],
            # These two were computed, declared in FEATURES, and written into
            # info.json and episodes_stats -- but never added here, so info.json
            # advertised columns the parquet did not contain. `flags` is the whole
            # lego-#4 remedy (let the consumer decide instead of masking frames away),
            # so its absence silently removed the guard.
            "action.commanded_rpy_base": [v.tolist() for v in ex["action.commanded_rpy_base"]],
            "action.delta_commanded_world":
                [v.tolist() for v in ex["action.delta_commanded_world"]],
            "flags": [v.tolist() for v in ex["flags"]],
            "action.xarm_servo_cartesian_base":
                [v.tolist() for v in ex["action.xarm_servo_cartesian_base"]],
            "sync_error_s": ex["sync_error_s"].tolist(),
            "timestamp": ex["timestamp"].tolist(),
            "frame_index": list(range(n)),
            "episode_index": [idx] * n,
            "index": list(range(total, total + n)),
            "task_index": [0] * n,
        }
        table = pa.table({
            k: (pa.array(v, pa.list_(pa.float32(), FEATURES[k][0]))
                if k in FEATURES and FEATURES[k][0] > 1 else pa.array(v))
            for k, v in cols.items()
        })
        # Assert the contract rather than trusting it. Two declared columns were
        # missing from every parquet this exporter ever wrote; this makes that class
        # of bug impossible instead of waiting for someone to notice downstream.
        promised = set(FEATURES)
        written = set(table.column_names)
        if promised - written:
            raise RuntimeError(
                f"declared in FEATURES but absent from the parquet: "
                f"{sorted(promised - written)} -- info.json would promise columns the "
                f"data files do not contain"
            )
        for k, (width, _names) in FEATURES.items():
            if width > 1 and len(table.column(k)[0]) != width:
                raise RuntimeError(
                    f"column {k} is {len(table.column(k)[0])} wide, FEATURES says {width}"
                )
        pq.write_table(table, out / f"data/chunk-{0:03d}/episode_{idx:06d}.parquet")

        for label in ex["cams"]:
            d = out / f"videos/chunk-{0:03d}/observation.images.{label}"
            d.mkdir(parents=True, exist_ok=True)
            # RE-ENCODE ONTO THE GRID. Copying the raw mp4 verbatim leaves frame i of
            # the video meaning a different instant from row i of the parquet: the
            # recording starts at that camera's own first frame (staggered per camera)
            # and its true rate is not exactly `fps`, so a loader that seeks by
            # timestamp -- which is what LeRobot does -- drifts across the episode.
            # After this, mp4 frame i IS parquet row i, by construction.
            n_written = _reencode_on_grid(
                r.path / "video" / f"{label}.mp4",
                d / f"episode_{idx:06d}.mp4",
                ex["cam_idx"][label], fps, int(get(cfg, "cameras.mp4_quality", 8)),
            )
            if n_written != n:
                print(f"  [warn] {label}: wrote {n_written} frames for {n} rows")

        worst = float(np.nanmax(ex["sync_error_s"])) if n else 0.0
        episodes.append({"episode_index": idx, "tasks": [args.task], "length": n})
        # Per-episode stats for EVERY numeric feature. LeRobot normalises inputs from
        # these; with only sync_error present a loader either errors or silently
        # normalises with defaults.
        st_all: dict[str, Any] = {}
        for key in FEATURES:
            arr = np.asarray(ex[key], dtype=np.float64)
            if arr.ndim == 1:
                arr = arr[:, None]
            with np.errstate(all="ignore"):
                st_all[key] = {
                    "min": np.nanmin(arr, axis=0).tolist(),
                    "max": np.nanmax(arr, axis=0).tolist(),
                    "mean": np.nanmean(arr, axis=0).tolist(),
                    "std": np.nanstd(arr, axis=0).tolist(),
                    "count": [n],
                }
        ep_stats.append({"episode_index": idx, "stats": st_all})
        total += n
        kept += 1
        print(f"  [ok]   {r.path.name}: {n} rows, worst sync {worst*1000:.1f} ms")

    if kept == 0:
        print("nothing exported")
        return 1

    features: dict[str, Any] = {
        k: {"dtype": "float32", "shape": [dim], "names": names}
        for k, (dim, names) in FEATURES.items()
    }
    for k in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        features[k] = {"dtype": "int64" if k != "timestamp" else "float32",
                       "shape": [1], "names": None}
    for label in cam_labels:
        features[f"observation.images.{label}"] = {
            "dtype": "video", "shape": [int(get(cfg, "cameras.height", 480)),
                                        int(get(cfg, "cameras.width", 640)), 3],
            "names": ["height", "width", "channel"],
            "info": {"video.fps": fps},
        }

    (out / "meta" / "info.json").write_text(json.dumps({
        "codebase_version": "v2.1",
        "robot_type": "xarm7_right",
        "total_episodes": kept,
        "total_frames": total,
        "total_tasks": 1,
        "total_videos": kept * len(cam_labels),
        "total_chunks": 1,
        "chunks_size": CHUNK,
        "fps": fps,
        "splits": {"train": f"0:{kept}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }, indent=4))
    (out / "meta" / "episodes.jsonl").write_text(
        "\n".join(json.dumps(e) for e in episodes) + "\n")
    (out / "meta" / "episodes_stats.jsonl").write_text(
        "\n".join(json.dumps(e) for e in ep_stats) + "\n")
    (out / "meta" / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": args.task}) + "\n")

    print(f"\n[export] {kept} episode(s), {total} frames @ {fps:g} Hz -> {out}")
    print("[export] action = XBOX CONTROLLER INPUT (both triggers); the commanded")
    print("         pose, its per-step delta and the literal xArm arguments are")
    print("         separate columns, and raw/ is still authoritative.")
    _tcp = (prov_tcp or {}).get("tcp_offset")
    print(f"[export] END-EFFECTOR POSE IS THE FLANGE, not the fingertips.")
    print(f"         tcp_offset = {_tcp!r}. With no tool offset configured, every")
    print(f"         observation.state pose is the plate the gripper bolts to --")
    print(f"         roughly 17 cm short of where the fingers actually meet.")
    print(f"         Do not read it as a grasp point without adding that offset.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
