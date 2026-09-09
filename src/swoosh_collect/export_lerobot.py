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
    for m in sorted((run_dir / "video").glob("*_frame_times.json")):
        d = json.loads(m.read_text())
        if d.get("t") and (run_dir / "video" / f"{d['label']}.mp4").is_file():
            cams.append({"label": d["label"], "t": np.asarray(d["t"], dtype=np.float64)})

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
    if hi - lo < 1.0 / fps:
        return None
    n = int((hi - lo) * fps)
    grid = lo + np.arange(n) / fps

    errs = []
    # --- action: what the OPERATOR did (5 dims) -----------------------------
    a_t = np.asarray([r["t"] for r in ctl])
    a_v = np.stack([
        np.asarray([r.get(k, 0.0) for k in ("move_x", "move_y", "height", "yaw", "gripper")],
                   dtype=np.float64)
        for r in ctl
    ])
    action, e = _pick(a_t, a_v, grid, 5); errs.append(e)

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
    s_t, joints = _series(stt, "joints_deg", 7)
    _, pose_base = _series(stt, "pose_base_mm_deg", 6)
    _, pose_world = _series(stt, "pose_world_xyz_mm", 3)
    sg_t = np.asarray([r["t"] for r in stt])
    sg_v = np.asarray([[r.get("gripper_pos", np.nan)] for r in stt])
    j, e = _pick(s_t, joints, grid, 7); errs.append(e)
    pb, _ = _pick(s_t, pose_base, grid, 6)
    pw, _ = _pick(s_t, pose_world, grid, 3)
    gp, _ = _pick(sg_t, sg_v, grid, 1)
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
        "action.commanded_rpy_base": cmd_rpy.astype(np.float32),                       # 3
        "flags": np.concatenate([
            cmd_aux[:, 2:3],                                   # clamped by workspace
            _pick(np.asarray([r["t"] for r in xac]) if xac else np.zeros(0),
                  np.asarray([[r.get("servo_code") if r.get("servo_code") is not None else -1]
                              for r in xac]) if xac else np.zeros((0, 1)), grid, 1)[0],
            _pick(s_t, np.asarray([[r.get("error_code", 0)] for r in stt]), grid, 1)[0],
            _pick(np.asarray([r["t"] for r in ctl]),
                  np.asarray([[1.0 if r.get("connected", True) else 0.0] for r in ctl]),
                  grid, 1)[0],
        ], axis=1).astype(np.float32),                                                 # 4
        "action.xarm_servo_cartesian_base": xarm_pose.astype(np.float32),    # 6
        "observation.state": state.astype(np.float32),                       # 17
        "sync_error_s": np.max(np.stack(errs), axis=0).astype(np.float32),
        "cam_idx": cam_idx,
        "cams": [c["label"] for c in cams],
    }


FEATURES = {
    "action": (5, ["move_x", "move_y", "height", "yaw", "gripper"]),
    "action.commanded_pose_world": (
        5, ["x_mm", "y_mm", "z_mm", "yaw_world_deg", "gripper_closure"]),
    "action.commanded_rpy_base": (3, ["roll_deg", "pitch_deg", "yaw_deg"]),
    # Without these a world model happily trains on stretches where the action did
    # nothing: clamped at a wall, refused by a latched controller error, or issued
    # while the pad was unplugged.
    "flags": (4, ["clamped_by_workspace", "servo_code", "arm_error_code", "pad_connected"]),
    "action.xarm_servo_cartesian_base": (
        6, ["x_mm", "y_mm", "z_mm", "roll_deg", "pitch_deg", "yaw_deg"]),
    "observation.state": (17, (
        [f"joint_{i+1}_deg" for i in range(7)]
        + ["base_x_mm", "base_y_mm", "base_z_mm",
           "base_roll_deg", "base_pitch_deg", "base_yaw_deg"]
        + ["world_x_mm", "world_y_mm", "world_z_mm", "gripper_pos"])),
    "sync_error_s": (1, ["sync_error_s"]),
}


def main() -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    ap = argparse.ArgumentParser(description="Export a campaign to LeRobot v2.1.")
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--out", default=None, help="output dir (default campaigns/<c>/lerobot)")
    ap.add_argument("--task", default="Pick and place with the right arm.")
    args = ap.parse_args()

    cfg = load_config()
    campaign = Campaign.open(args.campaign, cfg)
    out = Path(args.out) if args.out else campaign.path / "lerobot"
    fps = float(get(cfg, "recording.export_rate_hz", 30.0))

    runs = [r for r in campaign.runs() if (r.path / "raw" / "controller.jsonl").is_file()]
    if not runs:
        print(f"no runs with raw data in {campaign.path}")
        return 1

    (out / "meta").mkdir(parents=True, exist_ok=True)
    (out / f"data/chunk-{0:03d}").mkdir(parents=True, exist_ok=True)

    episodes, ep_stats, total = [], [], 0
    cam_labels: list[str] = []
    kept = 0
    for r in runs:
        ex = export_run(r.path, cfg)
        if ex is None:
            print(f"  [skip] {r.path.name}: not enough overlapping data")
            continue
        idx = kept
        cam_labels = cam_labels or ex["cams"]
        n = ex["n"]
        cols = {
            "observation.state": [v.tolist() for v in ex["observation.state"]],
            "action": [v.tolist() for v in ex["action"]],
            "action.commanded_pose_world": [v.tolist() for v in ex["action.commanded_pose_world"]],
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
    print("[export] action = XBOX CONTROLLER INPUT; commanded pose and the literal")
    print("         xArm arguments are separate columns, and raw/ is still authoritative.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
