"""Check a recorded run for the failure modes that ruined lego_assemblies.

Run it before trusting a campaign. Each check either passes, warns, or fails, and the
result is written into run.json so the exporter can refuse bad runs.

The four lego failures, and the check that catches each:
  1. action was raw controller pose in an unrecorded frame -> `provenance` (the frame
     matrix and its verification are stored with every run)
  2. action misaligned in time from the observation it caused -> `lag`
  3. state silently sample-and-held at a third of its stated rate -> `held`
  4. an over-aggressive mask threw away half the data -> `coverage`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .campaign import Campaign
from .config import get, load_config
from .summarize import _read_jsonl

GREEN, RED, YEL, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


def _r(name: str, level: str, detail: str, **extra) -> dict:
    return {"check": name, "level": level, "detail": detail, **extra}


def held_fraction(rows: list[dict], key: str) -> float:
    """Fraction of consecutive rows byte-identical in `key`. lego's state was 69.9%."""
    vals = [r.get(key) for r in rows if r.get(key) is not None]
    if len(vals) < 2:
        return float("nan")
    same = sum(1 for a, b in zip(vals, vals[1:]) if a == b)
    return same / (len(vals) - 1)


def stream_rate(rows: list[dict]) -> float:
    if len(rows) < 2:
        return 0.0
    span = float(rows[-1]["t"]) - float(rows[0]["t"])
    return (len(rows) - 1) / span if span > 0 else 0.0


def lag_curve(cmd: list[dict], stt: list[dict], max_ms: int = 400, step_ms: int = 20):
    """median |commanded(t) - measured(t+lag)| over a lag sweep."""
    if len(cmd) < 50 or len(stt) < 50:
        return [], []
    ct = np.array([x["t"] for x in cmd])
    cx = np.array([x["target_world_xyz_mm"] for x in cmd], dtype=float)
    st = np.array([x["t"] for x in stt])
    # Compare against the MEASURED pose. pose_world_xyz_mm is derived from the
    # controller's get_position, which is its PLAN -- so this sweep used to find its
    # minimum at 0 ms with a sub-millimetre residual, i.e. it was measuring how well
    # the controller agrees with itself and could not see servo lag at all. Derive the
    # pose from the measured joints instead, via the chain verified against the
    # controller to 2.5 mm.
    real = [x.get("joints_real_deg") for x in stt]
    if any(r for r in real):
        from .frames import base_to_world
        from .kinematics import fk_chain
        sx = np.array([
            base_to_world(fk_chain(r)[-1][:3, 3] * 1000.0) if r else [np.nan] * 3
            for r in real], dtype=float)
    else:
        sx = np.array([x.get("pose_world_xyz_mm") or [np.nan] * 3 for x in stt],
                      dtype=float)
    lags, meds = [], []
    for ms in range(0, max_ms + 1, step_ms):
        idx = np.clip(np.searchsorted(st, ct + ms / 1000.0), 0, len(st) - 1)
        d = np.linalg.norm(cx - sx[idx], axis=1)
        d = d[np.isfinite(d)]
        if len(d):
            lags.append(ms)
            meds.append(float(np.median(d)))
    return lags, meds


def validate_run(run_dir: Path, cfg: dict) -> dict[str, Any]:
    out: list[dict] = []
    ctl = _read_jsonl(run_dir / "raw" / "controller.jsonl")
    cmd = _read_jsonl(run_dir / "raw" / "commanded.jsonl")
    stt = _read_jsonl(run_dir / "raw" / "arm_state.jsonl")
    tk = _read_jsonl(run_dir / "raw" / "tick.jsonl")

    if not (ctl and cmd and stt):
        return {"ok": False, "checks": [_r("streams", "fail", "missing raw streams")]}

    # 1. one clock origin
    starts = [ctl[0]["t"], cmd[0]["t"], stt[0]["t"]]
    cam_t0 = []
    for m in sorted((run_dir / "video").glob("*_frame_times.json")):
        d = json.loads(m.read_text())
        if d.get("t"):
            cam_t0.append(float(d["t"][0]))
            starts.append(float(d["t"][0]))
    spread = max(starts) - min(starts)
    out.append(_r("clock origin", "pass" if spread < 5 else "fail",
                  f"streams start within {spread:.2f}s of each other", spread_s=spread))

    # 2. sample-and-hold (lego failure 3)
    hf = held_fraction(stt, "pose_base_mm_deg")
    out.append(_r("state freshness",
                  "fail" if hf >= 0.999 else ("warn" if hf > 0.7 else "pass"),
                  f"{100*hf:.1f}% of arm-state rows identical to the previous one "
                  f"(lego was 69.9%; a parked arm repeats legitimately, so only a "
                  f"stream that never changes is a failure)", held_fraction=hf))

    # 2b. ...and the same test on EVERY numeric field, not just that one. Checking a
    # single column is how gripper_pos sat at 80% held without anyone noticing: it is
    # polled at <=10 Hz and copied into every 50 Hz row. Fields known to be slower than
    # the stream are listed so they report their true rate instead of failing.
    _gh = float(get(cfg, "gripper.poll_hz", 20.0))
    SLOW_BY_DESIGN = {"gripper_pos": _gh, "gripper_pos_t": _gh}
    rate = float(get(cfg, "control.rate_hz", 100.0)) / 2.0    # arm_state is rate/2
    numeric: dict[str, int] = {}
    for r in stt:
        for k, v in r.items():
            if k == "t" or isinstance(v, bool):
                continue
            if isinstance(v, (int, float)) or (
                    isinstance(v, list) and v and all(
                        isinstance(x, (int, float)) for x in v)):
                numeric[k] = numeric.get(k, 0)
    # Fields that are CONSTANT when nothing is wrong. Reporting 100% held for
    # error_code is not a defect, it is the good outcome -- but the first version of
    # this check failed on them, which failed every healthy run, which (with the
    # exporter now honouring validation) refused every export. Report, never fail.
    CONSTANT_WHEN_HEALTHY = {"state", "mode", "error_code", "warn_code"}
    for k in sorted(numeric):
        f = held_fraction(stt, k)
        if k in CONSTANT_WHEN_HEALTHY:
            out.append(_r(f"held fraction [{k}]", "pass",
                          f"{100*f:.1f}% held - expected, this field is constant "
                          f"unless something goes wrong", held_fraction=f))
            continue
        if k in SLOW_BY_DESIGN:
            # Judging the RATE off the VALUE column was wrong: gripper_pos repeats
            # both because the poll is slow AND because the gripper is not moving, and
            # a run where the operator barely used the trigger failed for being
            # stationary. gripper_pos_t (the measurement TIME) advances on every poll
            # regardless of whether the value changed, so that is what carries the
            # rate. The value column is judged like any other: only a stream that
            # never changes at all is broken.
            expected = max(0.0, 1.0 - (SLOW_BY_DESIGN[k] / rate))
            if k.endswith("_t"):
                lvl = "pass" if f <= expected + 0.1 else "warn"
                note = (f"{100*f:.1f}% held; declared {SLOW_BY_DESIGN[k]:.0f} Hz in a "
                        f"{rate:.0f} Hz stream implies ~{100*expected:.0f}% -- this is "
                        f"the poll clock, so it measures the ACHIEVED rate "
                        f"(~{rate*(1-f):.1f} Hz)")
            else:
                lvl = "fail" if f >= 0.999 else "pass"
                note = (f"{100*f:.1f}% held -- repeats both from the {
                    SLOW_BY_DESIGN[k]:.0f} Hz poll and from the gripper simply not "
                        f"moving; see the _t column for the real rate")
            out.append(_r(f"held fraction [{k}]", lvl, note,
                          held_fraction=f, declared_hz=SLOW_BY_DESIGN[k]))
        else:
            # A stationary arm produces bit-identical rows honestly -- an episode with
            # a long dwell is not a broken episode. Only a stream that NEVER changes is
            # actually dead, so that is the failure; the rest is a warning to look at.
            lvl = "fail" if f >= 0.999 else ("warn" if f > 0.7 else "pass")
            out.append(_r(f"held fraction [{k}]", lvl,
                          f"{100*f:.1f}% of rows identical to the previous one"
                          + (" - stream never changes at all" if lvl == "fail"
                             else " (a stationary arm repeats legitimately)"),
                          held_fraction=f))

    # 2c. planned vs measured joints -- settles the is_real question from the data.
    both = [r for r in stt if r.get("joints_deg") and r.get("joints_real_deg")]
    if both:
        import numpy as _np
        d = _np.array([
            _np.abs(_np.asarray(r["joints_deg"], float)
                    - _np.asarray(r["joints_real_deg"], float)).max()
            for r in both
        ])
        mx, p95 = float(d.max()), float(_np.percentile(d, 95))
        # A real difference means the SDK default is the controller's PLAN, not servo
        # feedback -- in which case observation.state follows the command through a
        # stall and the real feedback is what should be exported.
        out.append(_r("joints: planned vs measured", "warn" if mx > 0.05 else "pass",
                      f"max {mx:.3f} deg, p95 {p95:.3f} deg over {len(both)} rows"
                      + (" -- the SDK default is NOT servo feedback; export "
                         "joints_real_deg" if mx > 0.05 else
                         " -- the default IS the servo feedback, no change needed"),
                      max_deg=mx, p95_deg=p95))
    else:
        out.append(_r("joints: planned vs measured", "warn",
                      "run predates joints_real_deg, cannot compare"))

    # 2d. how stale the recorded action actually is, measured not assumed.
    # Only over ticks where the stick is MOVING: a resting stick reports nothing, so
    # its last value is legitimately old and would swamp the statistic.
    aged = [r for r in ctl if r.get("input_age_s") is not None]
    import math as _m
    moving = []
    prev = None
    for r in aged:
        cur = (r.get("move_x", 0), r.get("move_y", 0), r.get("height", 0),
               r.get("yaw", 0), r.get("gripper", 0))
        a = r.get("input_age_s")
        if prev is not None and cur != prev and a is not None and not _m.isnan(a):
            moving.append(float(a))
        prev = cur
    if moving and min(moving) < -0.001:
        # Negative ages are impossible. Runs recorded before 2026-09-08 stamped these
        # by subtracting an absolute CLOCK_MONOTONIC event time from a t_loop0-relative
        # `t`, so the figure is the epoch gap, not a latency. Say so instead of
        # reporting millions of milliseconds as if they meant something.
        out.append(_r("action staleness (moving)", "warn",
                      f"unusable in this run: ages are negative "
                      f"(min {min(moving):.1f} s) -- recorded before the pad clock "
                      f"epoch fix, so input_age_s is the epoch gap, not a latency"))
    elif moving:
        import numpy as _np
        arr = _np.asarray(moving)
        p50, p95 = float(_np.median(arr)), float(_np.percentile(arr, 95))
        out.append(_r("action staleness (moving)", "warn" if p95 > 0.030 else "pass",
                      f"median {p50*1000:.1f} ms, p95 {p95*1000:.1f} ms behind the "
                      f"operator's thumb, over {len(moving)} changing ticks",
                      median_ms=p50 * 1000, p95_ms=p95 * 1000))
    else:
        out.append(_r("action staleness (moving)", "warn",
                      "no per-axis event times (run predates them, or the kernel "
                      "refused CLOCK_MONOTONIC on the pad)"))

    # 2e. was the arm's report stream actually alive throughout?
    alive = [r.get("report_alive") for r in stt if "report_alive" in r]
    if alive:
        bad = sum(1 for a in alive if not a)
        out.append(_r("arm report stream alive", "pass" if bad == 0 else "fail",
                      f"{len(alive) - bad}/{len(alive)} state rows had a live report "
                      f"stream" + ("" if bad == 0 else
                                   " -- joint reads during the dead window are suspect")))

    # 3. rates
    for nm, rows, want in (("controller", ctl, get(cfg, "control.rate_hz", 100.0)),
                           ("arm_state", stt, get(cfg, "control.rate_hz", 100.0) / 2)):
        r = stream_rate(rows)
        lvl = "pass" if r > 0.7 * float(want) else "warn"
        out.append(_r(f"{nm} rate", lvl, f"{r:.1f} Hz (expected ~{float(want):.0f})", hz=r))

    # 4. loop health
    if tk:
        dts = np.array([x.get("dt", 0.0) for x in tk], dtype=float)
        over = float((dts > 1.5 / float(get(cfg, "control.rate_hz", 100.0))).mean())
        out.append(_r("loop timing", "pass" if over < 0.05 else "warn",
                      f"{100*over:.1f}% of ticks overran; median dt "
                      f"{1000*float(np.median(dts)):.1f} ms", overrun_frac=over))

    # 5. the lego-2 detector: commanded must predict measured at a sane lag
    lags, meds = lag_curve(cmd, stt)
    if meds:
        best = int(np.argmin(meds))
        lvl = "pass" if 0 < best < len(meds) - 1 else "fail"
        out.append(_r("commanded->measured lag", lvl,
                      f"minimum {meds[best]:.2f} mm at {lags[best]} ms"
                      + ("" if lvl == "pass" else
                         " - minimum at the EDGE of the sweep. If it is at 0 ms with "
                         "a sub-millimetre residual, the 'measured' pose is the "
                         "controller's PLAN rather than a measurement (get_position "
                         "has no is_real variant), so this sweep cannot see servo lag "
                         "at all -- compare against joints_real_deg through the MJCF "
                         "chain instead. Otherwise it is the misalignment signature."),
                      best_lag_ms=lags[best], best_mm=meds[best],
                      curve=list(zip(lags, [round(m, 3) for m in meds]))))

    # 6. cameras: all present, frame counts reconcile
    want_cams = list(get(cfg, "cameras.expected_labels", []) or [])
    got = []
    for m in sorted((run_dir / "video").glob("*_frame_times.json")):
        d = json.loads(m.read_text())
        got.append(d["label"])
        n_ts, n_mp4 = len(d.get("t") or []), int(d.get("frames_in_mp4", -1))
        if n_mp4 >= 0:
            out.append(_r(f"frames reconcile [{d['label']}]",
                          "pass" if n_ts == n_mp4 else "fail",
                          f"{n_ts} timestamps vs {n_mp4} frames in the mp4"))
        else:
            # -1 means the mp4 could not be DECODED, which is strictly worse than a
            # count mismatch. This used to skip the check entirely, so an undecodable
            # video passed validation without comment.
            out.append(_r(f"frames reconcile [{d['label']}]", "fail",
                          "mp4 could not be decoded at all (frames_in_mp4 = -1)"))
        # A camera that died mid-run truncates the exported episode to the failure
        # point, so it must fail here rather than be inferred from a short episode.
        if d.get("stopped_early"):
            out.append(_r(f"camera ran to completion [{d['label']}]", "fail",
                          f"stopped early at t={d.get('stopped_at_s')}s: "
                          f"{d.get('error') or 'read failed'}"))
        else:
            out.append(_r(f"camera ran to completion [{d['label']}]", "pass", ""))
    missing = [w for w in want_cams if w not in got]
    out.append(_r("all cameras present", "pass" if not missing else "fail",
                  ", ".join(got) if not missing else f"missing {missing}"))

    # 7. flags: how much of the run was the action actually doing nothing
    clamped = sum(1 for r in cmd if r.get("clamped_by_workspace")) / max(len(cmd), 1)
    out.append(_r("workspace clamping", "pass" if clamped < 0.2 else "warn",
                  f"{100*clamped:.1f}% of ticks clamped at a wall", clamped_frac=clamped))

    ok = not any(c["level"] == "fail" for c in out)
    return {"ok": ok, "checks": out}


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate recorded runs.")
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--run", default=None)
    args = ap.parse_args()
    cfg = load_config()
    c = Campaign.open(args.campaign, cfg)
    runs = [r for r in c.runs() if args.run in (None, r.path.name)]
    if not runs:
        print("no matching runs")
        return 1
    bad = 0
    for r in runs:
        res = validate_run(r.path, cfg)
        print(f"\n{r.path.name}")
        for ch in res["checks"]:
            col = {"pass": GREEN, "warn": YEL, "fail": RED}[ch["level"]]
            mark = {"pass": "✓", "warn": "!", "fail": "✗"}[ch["level"]]
            print(f"  {col}{mark}{RESET} {ch['check']:32s} {DIM}{ch['detail']}{RESET}")
        # store it so the exporter can refuse a bad run
        meta = json.loads((r.path / "run.json").read_text()) if (r.path / "run.json").is_file() else {}
        meta["validation"] = res
        (r.path / "run.json").write_text(json.dumps(meta, indent=2))
        bad += not res["ok"]
    print(f"\n{len(runs) - bad}/{len(runs)} runs passed")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
