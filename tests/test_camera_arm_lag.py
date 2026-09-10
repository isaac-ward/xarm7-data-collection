"""The camera-vs-arm lag estimator must recover an offset we planted.

A latency number nobody can check is worse than none, so this builds a synthetic run
where the true camera delay is KNOWN and asserts the estimator finds it.

Three mistakes were made writing this test, all worth keeping written down:

 1. Brightness was varied with arm SPEED. Frame-to-frame difference is then the
    DERIVATIVE of speed, not speed, and the phase shift biased the answer by ~35 ms.
    On a real camera the image changes in proportion to how fast things MOVE, so the
    synthetic has to move something.

 2. The moving bar was driven by an integrated (always-positive) speed, so it swept
    once to the edge and pinned there. Most of the clip had no motion at all and the
    peak correlation collapsed to 0.21.

 3. Frames were 64x64. The estimator decimates by [::4] (sensible for 640x480), which
    left a 2-pixel bar and destroyed the signal. The synthetic must be a realistic
    frame size.
"""
import json
import pathlib
import shutil
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from swoosh_collect.config import load_config          # noqa: E402
from swoosh_collect.validate import camera_arm_lag     # noqa: E402

DUR, ARM_HZ, CAM_FPS = 20.0, 50.0, 30.0
H, W = 240, 320
_rng = np.random.default_rng(0)
_BUMPS = np.sort(_rng.uniform(1.5, DUR - 1.5, 16))
_SIGNS = _rng.choice([-1.0, 1.0], size=len(_BUMPS))


def pos_mm(t: float) -> float:
    """Position as a sum of in-and-out bumps, so the arm keeps moving all run."""
    return 400.0 + 40.0 * float(
        np.sum(_SIGNS * np.exp(-((t - _BUMPS) ** 2) / (2 * 0.20 ** 2)))
    )


def build(tmp: pathlib.Path, true_lag_s: float) -> None:
    import imageio.v2 as imageio

    (tmp / "raw").mkdir(parents=True, exist_ok=True)
    (tmp / "video").mkdir(parents=True, exist_ok=True)

    with (tmp / "raw" / "arm_state.jsonl").open("w") as fh:
        for k in range(int(DUR * ARM_HZ)):
            t = k / ARM_HZ
            x = pos_mm(t)
            fh.write(json.dumps({
                "t": t, "joints_deg": [0.0] * 7,
                "pose_base_mm_deg": [x, 0.0, 500.0, 180.0, 0.0, 0.0],
                "pose_world_xyz_mm": [x, 0.0, 500.0], "gripper_pos": 400.0,
                "state": 0, "mode": 1, "error_code": 0, "warn_code": 0}) + "\n")

    lbl = "gripper_right_top"
    ts = list(np.arange(0.0, DUR, 1.0 / CAM_FPS))
    w = imageio.get_writer(str(tmp / "video" / f"{lbl}.mp4"), fps=CAM_FPS,
                           codec="libx264", quality=7, macro_block_size=16)
    for t in ts:
        # A wide bar tracking arm position, DELAYED -- so image change is proportional
        # to arm speed, delayed, which is the relationship on a real camera.
        c = int(np.clip(W / 2 + (pos_mm(t - true_lag_s) - 400.0) * 2.2, 30, W - 30))
        f = np.zeros((H, W, 3), np.uint8)
        f[:, c - 25:c + 25] = 235
        w.append_data(f)
    w.close()
    (tmp / "video" / f"{lbl}_frame_times.json").write_text(json.dumps(
        {"label": lbl, "frames": len(ts), "measured_fps": CAM_FPS, "error": "",
         "t": [round(float(x), 6) for x in ts]}))


def _recover(true_lag: float) -> dict:
    tmp = ROOT / "campaigns" / f"_lagtest_{int(true_lag * 1000)}"
    shutil.rmtree(tmp, ignore_errors=True)
    build(tmp, true_lag)
    r = camera_arm_lag(tmp, load_config())
    shutil.rmtree(tmp, ignore_errors=True)
    assert r["ok"], r
    return r


def test_recovers_planted_lags():
    worst = 0.0
    for true_lag in (0.0, 0.05, 0.10, 0.20):
        r = _recover(true_lag)
        err = abs(r["lag_s"] - true_lag)
        worst = max(worst, err)
        print(f"    planted {true_lag * 1000:5.0f} ms -> measured "
              f"{r['lag_s'] * 1000:7.1f} ms   err {err * 1000:4.1f} ms   "
              f"r={r['peak_corr']:.2f}")
    assert worst < 0.020, f"worst error {worst * 1000:.1f} ms"


def test_peak_is_sharp():
    """A flat correlation curve would make the number meaningless."""
    r = _recover(0.10)
    c = np.array([v for _, v in r["curve"]])
    print(f"    peak r={c.max():.2f}, median r={np.median(c):.2f}")
    assert c.max() > 0.5, f"peak correlation only {c.max():.2f}"
    assert c.max() - np.median(c) > 0.25, "correlation curve too flat to trust"


if __name__ == "__main__":
    import traceback

    fs = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for f in fs:
        try:
            print(f"  {f.__name__}")
            f()
        except Exception:
            bad += 1
            print("    FAILED")
            traceback.print_exc()
    print(f"\n{len(fs) - bad}/{len(fs)} passed")
    raise SystemExit(1 if bad else 0)
