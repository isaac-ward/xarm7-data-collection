"""Per-run review artefacts, plus the campaign index page.

    summary.mp4      2x2 grid of the four cameras, TIME-SYNCED on the shared clock
    inputs.png       controller axes + episode markers over time
    kinematics.png   joint angles, world EE pose, gripper over time
    index.html       one page per campaign listing every run

TIME-SYNCED MATTERS. The four cameras free-run at slightly different rates and start
0.25 s apart because of the USB stagger, so concatenating frame N of each gives a grid
where the panels drift apart. Instead each output frame picks, per camera, the frame
whose recorded timestamp is nearest that output time -- the same nearest-in-time rule
the LeRobot export uses, so the video shows what the dataset will contain.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .campaign import Campaign
from .config import get, load_config


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    for line in path.open():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def _nearest_index(times: np.ndarray, t: float) -> int:
    if len(times) == 0:
        return -1
    i = int(np.searchsorted(times, t))
    if i <= 0:
        return 0
    if i >= len(times):
        return len(times) - 1
    return i if abs(times[i] - t) < abs(times[i - 1] - t) else i - 1


# -- summary mp4 -------------------------------------------------------------
def build_summary_mp4(run_dir: Path, cfg: dict, fps: float | None = None) -> Path | None:
    import cv2
    import imageio.v2 as imageio

    video_dir = run_dir / "video"
    metas = sorted(video_dir.glob("*_frame_times.json"))
    if not metas:
        return None
    order = list(get(cfg, "cameras.expected_labels", []) or [])
    cams = []
    for m in metas:
        d = json.loads(m.read_text())
        mp4 = video_dir / f"{d['label']}.mp4"
        if mp4.is_file() and d.get("t"):
            cams.append({"label": d["label"], "t": np.asarray(d["t"], dtype=np.float64),
                         "mp4": mp4})
    if not cams:
        return None
    cams.sort(key=lambda c: order.index(c["label"]) if c["label"] in order else 99)

    fps = float(fps or get(cfg, "recording.export_rate_hz", 30.0))
    # Only the span every camera covers, so no panel is frozen at an end.
    t_start = max(float(c["t"][0]) for c in cams)
    t_end = min(float(c["t"][-1]) for c in cams)
    if t_end <= t_start:
        return None

    # STREAM, don't accumulate. `[f for f in reader]` on four 640x480 cameras is about
    # 8 GB for a five-minute episode. Each camera is read once, forward only, keeping a
    # single current frame -- the output grid is monotonic in time, so the source index
    # per camera is monotonic too.
    readers = [imageio.get_reader(str(c["mp4"])) for c in cams]
    cur: list[Any] = [None] * len(cams)
    at: list[int] = [-1] * len(cams)

    def advance_to(k: int, want: int):
        """Forward-only seek: pull frames until we reach `want`."""
        while at[k] < want:
            try:
                cur[k] = readers[k].get_next_data()
                at[k] += 1
            except (StopIteration, IndexError):
                break
        return cur[k]

    h = int(get(cfg, "cameras.height", 480))
    w = int(get(cfg, "cameras.width", 640))
    ph, pw = h // 2, w // 2          # half-size panels keep the file small
    out_path = run_dir / "summary.mp4"
    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264",
                                quality=int(get(cfg, "cameras.mp4_quality", 8)),
                                macro_block_size=1)
    try:
        n = int((t_end - t_start) * fps)
        for k in range(max(n, 1)):
            t = t_start + k / fps
            panels = []
            for ci, cam in enumerate(cams):     # `ci`, not `k`: `k` is the OUTPUT frame
                idx = _nearest_index(cam["t"], t)
                frame = advance_to(ci, idx)
                if frame is not None:
                    img = cv2.resize(frame, (pw, ph))
                else:
                    img = np.zeros((ph, pw, 3), dtype=np.uint8)
                img = img.copy()
                cv2.putText(img, cam["label"], (6, 16), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (255, 255, 255), 1, cv2.LINE_AA)
                panels.append(img)
            while len(panels) < 4:
                panels.append(np.zeros((ph, pw, 3), dtype=np.uint8))
            grid = np.concatenate(
                [np.concatenate(panels[:2], axis=1),
                 np.concatenate(panels[2:4], axis=1)], axis=0
            )
            cv2.putText(grid, f"t={t - t_start:6.2f}s", (6, grid.shape[0] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            writer.append_data(grid)
    finally:
        writer.close()
        for r in readers:
            try:
                r.close()
            except Exception:
                pass
    return out_path


# -- plots -------------------------------------------------------------------
def plot_inputs(run_dir: Path) -> Path | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _read_jsonl(run_dir / "raw" / "controller.jsonl")
    if not rows:
        return None
    t = np.array([r["t"] for r in rows])
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    for key, lbl in (("move_x", "left stick  fwd/back"),
                     ("move_y", "left stick  left/right")):
        axes[0].plot(t, [r.get(key, 0.0) for r in rows], lw=1.0, label=lbl)
    axes[0].set_ylabel("stick"); axes[0].set_ylim(-1.05, 1.05)
    for key, lbl in (("height", "right stick  up/down"), ("yaw", "right stick  yaw")):
        axes[1].plot(t, [r.get(key, 0.0) for r in rows], lw=1.0, label=lbl)
    axes[1].set_ylabel("stick"); axes[1].set_ylim(-1.05, 1.05)
    axes[2].plot(t, [r.get("gripper", 0.0) for r in rows], lw=1.0,
                 color="tab:red", label="right trigger  gripper")
    axes[2].set_ylabel("closure"); axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_xlabel("time since run start (s)")
    for a in axes:
        a.grid(alpha=0.3); a.legend(loc="upper right", fontsize=8)
    fig.suptitle(f"controller inputs -- {run_dir.name}  (the world-model action signal)")
    fig.tight_layout()
    out = run_dir / "inputs.png"
    fig.savefig(out, dpi=110); plt.close(fig)
    return out


def plot_kinematics(run_dir: Path) -> Path | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    state = _read_jsonl(run_dir / "raw" / "arm_state.jsonl")
    cmd = _read_jsonl(run_dir / "raw" / "commanded.jsonl")
    if not state:
        return None
    ts = np.array([r["t"] for r in state])
    joints = np.array([r.get("joints_deg") or [np.nan] * 7 for r in state], dtype=float)
    world = np.array([r.get("pose_world_xyz_mm") or [np.nan] * 3 for r in state], dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for j in range(joints.shape[1]):
        axes[0].plot(ts, joints[:, j], lw=0.9, label=f"J{j+1}")
    axes[0].set_ylabel("joint angle (deg)")
    axes[0].legend(ncol=7, fontsize=7, loc="upper right")

    for i, (lbl, col) in enumerate((("x fwd", "tab:blue"), ("y left", "tab:orange"),
                                    ("z up", "tab:green"))):
        axes[1].plot(ts, world[:, i], lw=1.1, color=col, label=f"measured {lbl}")
    if cmd:
        tc = np.array([r["t"] for r in cmd])
        tw = np.array([r["target_world_xyz_mm"] for r in cmd], dtype=float)
        for i, col in enumerate(("tab:blue", "tab:orange", "tab:green")):
            axes[1].plot(tc, tw[:, i], lw=0.8, ls="--", color=col, alpha=0.7)
    axes[1].set_ylabel("EE position, world (mm)")
    axes[1].legend(fontsize=8, loc="upper right",
                   title="dashed = commanded", title_fontsize=7)

    g = np.array([r.get("gripper_pos", np.nan) for r in state], dtype=float)
    axes[2].plot(ts, g, lw=1.1, color="tab:red", label="measured gripper position")
    axes[2].set_ylabel("gripper"); axes[2].set_xlabel("time since run start (s)")
    axes[2].legend(fontsize=8, loc="upper right")
    for a in axes:
        a.grid(alpha=0.3)
    fig.suptitle(f"kinematics -- {run_dir.name}")
    fig.tight_layout()
    out = run_dir / "kinematics.png"
    fig.savefig(out, dpi=110); plt.close(fig)
    return out


# -- campaign index ----------------------------------------------------------
def build_index(campaign: Campaign) -> Path:
    runs = campaign.runs()
    total = sum(r.duration_s for r in runs)
    rows = []
    for r in runs:
        has = lambda n: (r.path / n).is_file()  # noqa: E731
        media = []
        if has("summary.mp4"):
            media.append(
                f'<video src="{r.path.name}/summary.mp4" controls muted '
                f'preload="metadata"></video>'
            )
        for png in ("inputs.png", "kinematics.png"):
            if has(png):
                media.append(f'<a href="{r.path.name}/{png}">'
                             f'<img src="{r.path.name}/{png}"></a>')
        rows.append(
            f"<tr><td class=n>{r.index:04d}</td>"
            f"<td>{r.duration_s:.1f}s</td>"
            f"<td>{'ok' if r.complete else r.stopped_by}</td>"
            f"<td class=m>{''.join(media) or '<i>not summarized</i>'}</td></tr>"
        )
    html = f"""<!doctype html><meta charset=utf-8>
<title>{campaign.name} -- swoosh collection</title>
<style>
 body{{font:14px/1.5 system-ui,sans-serif;margin:2rem;background:#fbfbfa;color:#1a1a1a}}
 h1{{margin:0 0 .2rem}} .sub{{color:#666;margin-bottom:1.4rem}}
 table{{border-collapse:collapse;width:100%}}
 th,td{{text-align:left;padding:.55rem .7rem;border-bottom:1px solid #e5e5e5;vertical-align:top}}
 th{{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:#666}}
 .n{{font-variant-numeric:tabular-nums;font-weight:600}}
 .m video{{width:320px;border-radius:6px;background:#000}}
 .m img{{width:260px;border-radius:6px;border:1px solid #e5e5e5;margin-left:.5rem}}
 @media(prefers-color-scheme:dark){{body{{background:#151515;color:#eee}}
   th,td{{border-color:#333}} .m img{{border-color:#333}} .sub{{color:#999}}}}
</style>
<h1>{campaign.name}</h1>
<div class=sub>{len(runs)} run(s) &middot; {sum(r.complete for r in runs)} complete
 &middot; {total/60:.1f} min total &middot; created {campaign.meta.get('created_iso','?')}</div>
<table><tr><th>run</th><th>duration</th><th>ended</th><th>media</th></tr>
{''.join(rows) or '<tr><td colspan=4><i>no runs yet</i></td></tr>'}
</table>"""
    out = campaign.path / "index.html"
    out.write_text(html, encoding="utf-8")
    return out


def summarize_run(run_dir: Path, cfg: dict, force: bool = False) -> dict[str, Any]:
    made = {}
    if force or not (run_dir / "inputs.png").is_file():
        made["inputs"] = plot_inputs(run_dir)
    if force or not (run_dir / "kinematics.png").is_file():
        made["kinematics"] = plot_kinematics(run_dir)
    if force or not (run_dir / "summary.mp4").is_file():
        made["summary"] = build_summary_mp4(run_dir, cfg)
    return made


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build per-run summary MP4s and plots, and the campaign index."
    )
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--run", default=None, help="one run folder name; default all")
    ap.add_argument("--force", action="store_true", help="rebuild artefacts that exist")
    args = ap.parse_args()

    cfg = load_config()
    campaign = Campaign.open(args.campaign, cfg)
    runs = [r for r in campaign.runs() if args.run in (None, r.path.name)]
    if not runs:
        print(f"no matching runs in {campaign.path}")
        return 1
    for r in runs:
        print(f"[summarize] {r.path.name} ...", flush=True)
        made = summarize_run(r.path, cfg, force=args.force)
        for k, v in made.items():
            print(f"    {k:10s} {'-> ' + Path(v).name if v else 'skipped (no data)'}")
    from .campaign import set_run_status

    for r in runs:
        # A run whose summariser was killed would otherwise sit at `processing` forever.
        set_run_status(r.path, "ready")
    idx = build_index(campaign)
    print(f"\n[summarize] index: {idx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
