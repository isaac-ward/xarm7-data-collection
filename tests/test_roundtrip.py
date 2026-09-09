"""collect -> export round trip against FAKE hardware.

This is the test whose absence let two fatal bugs through: the streams and the
cameras had different clock origins, and the exported video was not on the export
grid. Neither is visible in a unit test of either half alone.

A known square-wave is injected on the controller; the arm follows it with a known
lag; the cameras encode a frame counter. After export we assert the parquet action
matches what was injected and the video frame at row i is the frame the timestamps
say it should be.
"""
import json, sys, pathlib, shutil
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from swoosh_collect.config import load_config
from swoosh_collect.campaign import Campaign
from swoosh_collect.recorder import RunRecorder
from swoosh_collect import export_lerobot as EX

FPS, DUR, LAG = 30.0, 6.0, 0.05


def build(tmp: pathlib.Path, cfg, uptime: float):
    """Simulate a session where the process has been up `uptime` before A is pressed.

    Every stream is stamped against t_loop0 (process start) -- the SHARED origin.
    If cameras used the A-press instead, the export must refuse.
    """
    import imageio.v2 as imageio
    run = tmp / "run_0001_20260909_000000"
    (run / "raw").mkdir(parents=True); (run / "video").mkdir(parents=True)
    rec = RunRecorder(run, t0=0.0)
    t0 = uptime                       # A pressed at this point on the shared clock
    n = int(DUR * 100)
    for k in range(n):
        t = t0 + k / 100.0
        mx = 1.0 if (int((t - t0) * 2) % 2 == 0) else -1.0     # 0.5 s square wave
        pos = np.array([330.0 + 50 * (t - t0), 0.0, 620.0])
        rec.controller({"t": t, "move_x": mx, "move_y": 0.0, "height": 0.0,
                        "yaw": 0.0, "gripper": 0.5, "connected": True, "raw": {}})
        rec.commanded(t, pos, [180.0, 0.0, 0.0], 12.0, 0.5, False)
        rec.xarm_command(t, [*pos, 180.0, 0.0, 0.0], 0, 425.0)
        if k % 2 == 0:
            meas = pos - np.array([50 * LAG, 0, 0])            # arm trails by LAG
            rec.arm_state({"t": t, "joints_deg": [float(j) for j in range(7)],
                           "pose_base_mm_deg": [*meas, 180.0, 0.0, 0.0],
                           "pose_world_xyz_mm": list(meas), "gripper_pos": 425.0,
                           "state": 0, "mode": 1, "error_code": 0, "warn_code": 0})
    for i, lbl in enumerate(cfg["cameras"]["expected_labels"]):
        cfps = [30.0, 28.5, 30.0, 25.0][i]
        ts = list(np.arange(t0 + i * 0.25, t0 + DUR, 1.0 / cfps))
        w = imageio.get_writer(str(run / "video" / f"{lbl}.mp4"), fps=cfps,
                               codec="libx264", quality=7, macro_block_size=1)
        for kk in range(len(ts)):
            # Whole-frame grey encodes the frame index. A single-pixel counter does
            # NOT survive h264 -- the first version of this test used one and failed
            # for that reason, not because the alignment was wrong.
            f = np.full((64, 64, 3), (kk * 7) % 251, np.uint8)
            w.append_data(f)
        w.close()
        (run / "video" / f"{lbl}_frame_times.json").write_text(json.dumps(
            {"label": lbl, "device": f"/dev/video{i}", "frames": len(ts),
             "measured_fps": cfps, "error": "", "t": [round(float(x), 6) for x in ts]}))
    rec.close("b_button")
    return run


def test_shared_origin_exports():
    cfg = load_config(); tmp = ROOT / "campaigns" / "_rt"; shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    run = build(tmp, cfg, uptime=137.0)          # long uptime: the old bug's trigger
    ex = EX.export_run(run, cfg)
    assert ex is not None, "export refused a correctly-stamped run"
    assert ex["n"] > int((DUR - 1.5) * FPS), f"only {ex['n']} rows"
    print(f"    exported {ex['n']} rows after 137 s uptime")


def test_mismatched_origin_is_refused():
    """The exact old bug: cameras stamped from the A press, streams from process start."""
    cfg = load_config(); tmp = ROOT / "campaigns" / "_rt2"; shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    run = build(tmp, cfg, uptime=137.0)
    for m in (run / "video").glob("*_frame_times.json"):        # rebase cameras to 0
        d = json.loads(m.read_text()); d["t"] = [t - 137.0 for t in d["t"]]
        m.write_text(json.dumps(d))
    assert EX.export_run(run, cfg) is None, "MISALIGNED RUN WAS EXPORTED"
    print("    mismatched origins correctly refused")


def test_action_survives_export():
    cfg = load_config(); tmp = ROOT / "campaigns" / "_rt3"; shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    run = build(tmp, cfg, uptime=12.0)
    ex = EX.export_run(run, cfg)
    a = ex["action"][:, 0]
    assert set(np.unique(np.round(a, 3))) <= {-1.0, 1.0}, np.unique(a)
    flips = int((np.diff(np.sign(a)) != 0).sum())
    expect = int(ex["n"] / FPS / 0.5) - 1
    assert abs(flips - expect) <= 2, f"{flips} flips, expected ~{expect}"
    print(f"    square wave survived: {flips} flips, expected ~{expect}")


def test_video_frame_matches_row():
    """mp4 frame i must BE parquet row i after the on-grid re-encode."""
    import imageio.v2 as imageio
    cfg = load_config(); tmp = ROOT / "campaigns" / "_rt4"; shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    run = build(tmp, cfg, uptime=9.0)
    ex = EX.export_run(run, cfg)
    lbl = cfg["cameras"]["expected_labels"][1]          # the 28.5 fps one, staggered
    dst = tmp / "out.mp4"
    written = EX._reencode_on_grid(run / "video" / f"{lbl}.mp4", dst,
                                   ex["cam_idx"][lbl], FPS, 8)
    assert written == ex["n"], f"{written} frames for {ex['n']} rows"
    got = [f for f in imageio.get_reader(str(dst))]
    bad = []
    for i in range(0, ex["n"], max(1, ex["n"] // 12)):
        want = (int(ex["cam_idx"][lbl][i]) * 7) % 251
        have = float(np.mean(got[i]))
        if abs(have - want) > 6:
            bad.append((i, want, round(have, 1)))
    assert not bad, f"re-encoded video not aligned to rows at {bad}"
    print(f"    {written} video frames, frame i == row i at every probe")


if __name__ == "__main__":
    import traceback
    fs = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for f in fs:
        try:
            print(f"  {f.__name__}"); f()
        except Exception:
            bad += 1; print("    FAILED"); traceback.print_exc()
    for d in ("_rt", "_rt2", "_rt3", "_rt4"):
        shutil.rmtree(ROOT / "campaigns" / d, ignore_errors=True)
    print(f"\n{len(fs)-bad}/{len(fs)} passed")
    raise SystemExit(1 if bad else 0)
