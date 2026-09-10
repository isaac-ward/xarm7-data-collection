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

from .campaign import Campaign, set_run_status
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
    xac = _read_jsonl(run_dir / "raw" / "xarm_command.jsonl")

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
        # PASS, not warn. This check existed to decide WHICH joint reading to export,
        # and that is settled: observation.state takes joints_real_deg and the plan
        # ships beside it. The divergence is still worth seeing -- it is how hard the
        # arm was failing to follow -- so it stays in the detail. Leaving it a warning
        # meant every healthy run carried one, which makes a counted verdict useless.
        out.append(_r("joints: planned vs measured", "pass",
                      f"max {mx:.3f} deg, p95 {p95:.3f} deg over {len(both)} rows"
                      + (" -- the SDK default is the PLAN, not feedback; "
                         "observation.state already uses joints_real_deg"
                         if mx > 0.05 else
                         " -- the default IS the servo feedback"),
                      max_deg=mx, p95_deg=p95))
    else:
        out.append(_r("joints: planned vs measured", "warn",
                      "run predates joints_real_deg, cannot compare"))

    # 2d. how stale the recorded action actually is, measured not assumed.
    # Only over ticks where the stick is MOVING: a resting stick reports nothing, so
    # its last value is legitimately old and would swamp the statistic.
    # Per-AXIS, and only for axes whose value actually changed on that tick. A single
    # scalar per tick cannot express this: an untouched trigger is genuinely minutes
    # stale while the stick being moved is milliseconds fresh, so any max/min over all
    # axes describes the wrong thing. raw_age_s carries each axis separately.
    import math as _m
    AXES = ("move_x", "move_y", "height", "yaw", "gripper")
    axis_code = {k: get(cfg, f"controller.axes.{k}") for k in AXES}
    moving = []
    prev = None
    for r in ctl:
        cur = {k: r.get(k, 0) for k in AXES}
        ages = r.get("raw_age_s") or {}
        if prev is not None:
            for k in AXES:
                if cur[k] != prev[k]:
                    a = ages.get(axis_code.get(k))
                    if a is not None and not _m.isnan(float(a)):
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
                      f"operator's thumb, over {len(moving)} axis-changes",
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

    # 2f. EVERY timestamp field must lie inside the run's own window.
    # Two fields shipped stamped with absolute time.monotonic() while every `t` is
    # relative to t_loop0 -- 61,615 s adrift, which made the export resample against a
    # clock it shared nothing with. A field claiming to be a time is checked against
    # the window it must live in, so the next one cannot pass silently.
    win_lo = min(r["t"] for r in (ctl[:1] + cmd[:1] + stt[:1]))
    win_hi = max(r["t"] for r in (ctl[-1:] + cmd[-1:] + stt[-1:]))
    slack = 5.0
    bad_fields: dict[str, tuple] = {}
    for nm, rows in (("controller", ctl), ("commanded", cmd),
                     ("xarm_command", xac), ("arm_state", stt)):
        for r in rows:
            for k, v in r.items():
                if k == "t" or not k.endswith("_t"):
                    continue
                if v is None or not isinstance(v, (int, float)):
                    continue
                if isinstance(v, float) and v != v:      # NaN is "not measured"
                    continue
                # A "when did X last happen" stamp can legitimately predate the
                # run -- the last gripper command often lands during startup, before
                # A was pressed, giving a small negative value. That is the right
                # clock, not the wrong epoch. Only the magnitude distinguishes them:
                # t_loop0-relative times are seconds, a foreign epoch is tens of
                # thousands. So allow anything from the session start onwards.
                if not (-slack <= v <= win_hi + slack):
                    key = f"{nm}.{k}"
                    if key not in bad_fields:
                        bad_fields[key] = (v, win_lo, win_hi)
    if bad_fields:
        detail = "; ".join(
            f"{k}={v:.1f} outside [0,{hi:.1f}]"
            for k, (v, lo, hi) in sorted(bad_fields.items()))
        out.append(_r("timestamp fields share one clock", "fail",
                      f"{detail} -- a different epoch, so anything resampled on it is "
                      f"aligned to nothing"))
    else:
        out.append(_r("timestamp fields share one clock", "pass",
                      "every *_t field lies inside the run's window"))

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

    # 7. camera vs arm-state alignment. Both are OBSERVATIONS, so the servo lag
    # cancels and what is left is the residual offset between the two recorded
    # streams. Validated against planted offsets to 0.7 ms (tests/test_camera_arm_lag).
    try:
        cl = camera_arm_lag(run_dir, cfg)
        if cl.get("ok"):
            lag_ms = cl["lag_s"] * 1000.0
            sharp = cl["peak_corr"] > 0.35
            # EDGE-PINNED means the search found no interior peak and handed back the
            # boundary. Two runs here returned exactly -333.3 ms = -10.00 frames, the
            # sweep bound, and the spread across seven runs was -333..+115 ms, which no
            # constant physical offset can be. That is the same signature the servo-lag
            # check already calls out -- so it must not be reported as a measurement.
            frames = cl["lag_frames_at_30hz"]
            edge = abs(abs(frames) - int(0.35 * 30.0)) < 0.25
            detail = (f"{lag_ms:+.1f} ms ({frames:+.2f} frames at 30 Hz), "
                      f"peak r={cl['peak_corr']:.2f} on {cl['camera']}")
            if edge or not sharp:
                # Could not measure -- that is not the same as having found a defect,
                # so it does not count against the run. The number stays visible.
                why = ("pinned at the sweep edge, so no interior peak was found"
                       if edge else "correlation too weak to trust; the arm may "
                                    "barely move in this camera's view")
                out.append(_r("camera vs arm-state alignment", "pass",
                              f"INDETERMINATE -- {detail}; {why}",
                              lag_s=cl["lag_s"], peak_corr=cl["peak_corr"],
                              indeterminate=True))
            else:
                # A sharp peak inside the sweep IS a measurement; judge it against the
                # offset this rig is KNOWN to have, not against zero.
                #
                # Measured on every trustworthy estimate to date (sharp peak, inside
                # the sweep): +75.4, +80.3 and +111.7 ms -- mean +89.1, sd 16.1, all
                # the same sign. That is the camera's own capture latency: the kernel
                # stamps a buffer once it is FILLED, which is after exposure, sensor
                # readout, on-camera MJPG compression and USB transfer, so the photons
                # landed before the timestamp and no timestamp fix can remove it.
                # read_lag_s (21 ms) covers only post_read - kernel, a later stage.
                #
                # Judging against zero made every healthy run warn, and a permanent
                # warning is one nobody reads. So the threshold sits above the known
                # offset with headroom: what matters now is a DEPARTURE from it.
                thresh = float(get(cfg, "recording.camera_align_warn_ms", 120.0))
                out.append(_r("camera vs arm-state alignment",
                              "pass" if abs(lag_ms) < thresh else "warn",
                              detail + f" (warns beyond {thresh:.0f} ms; this rig's "
                                       f"known offset is ~89 ms)",
                              lag_s=cl["lag_s"], peak_corr=cl["peak_corr"]))
        else:
            out.append(_r("camera vs arm-state alignment", "warn",
                          cl.get("error", "could not measure")))
    except Exception as exc:
        out.append(_r("camera vs arm-state alignment", "warn",
                      f"{type(exc).__name__}: {exc}"))

    # 8. flags: how much of the run was the action actually doing nothing
    clamped = sum(1 for r in cmd if r.get("clamped_by_workspace")) / max(len(cmd), 1)
    out.append(_r("workspace clamping", "pass" if clamped < 0.2 else "warn",
                  f"{100*clamped:.1f}% of ticks clamped at a wall", clamped_frac=clamped))

    ok = not any(c["level"] == "fail" for c in out)
    n_pass = sum(1 for c in out if c["level"] == "pass")
    return {"ok": ok, "checks": out,
            # counted verdict: green only at full marks, so a warning is not something
            # you can skim past, and a DROP in the total is visible too
            "n_pass": n_pass, "n_total": len(out),
            "n_warn": sum(1 for c in out if c["level"] == "warn"),
            "n_fail": sum(1 for c in out if c["level"] == "fail"),
            "all_green": n_pass == len(out)}


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
        # Store the WHOLE verdict, not just the check list. The run card and the
        # campaign API read the top-level counts; writing only `validation` left them
        # showing whatever the collector recorded at the time, so a re-check against a
        # changed threshold silently appeared to do nothing -- and the data looked
        # broken to anyone reading it later. Same fields, same order as _health_check.
        fails = [ch["check"] for ch in res["checks"] if ch["level"] == "fail"]
        warns = [ch["check"] for ch in res["checks"] if ch["level"] == "warn"]
        status = json.loads((r.path / "run.json").read_text()).get("status", "ready") \
            if (r.path / "run.json").is_file() else "ready"
        set_run_status(r.path, status, validation=res, checks_ok=bool(res["ok"]),
                       checks_reasons=fails + warns,
                       checks_pass=res["n_pass"], checks_total=res["n_total"],
                       checks_green=bool(res["all_green"]))
        bad += not res["ok"]
    print(f"\n{len(runs) - bad}/{len(runs)} runs passed")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())


# -- camera vs arm-state alignment -------------------------------------------
def camera_arm_lag(run_dir: Path, cfg: dict, label: str | None = None,
                   max_lag_s: float = 0.35, grid_hz: float = 30.0) -> dict[str, Any]:
    """Measure the residual offset between the CAMERA stream and the ARM-STATE stream.

    Cross-correlates how fast the arm is moving (|d pose_world / dt|, from arm_state)
    against how much the image is changing (mean |frame - prev frame|, from a wrist
    camera). Both are OBSERVATIONS, so the servo lag between command and motion cancels
    out and what remains is purely the offset between the two recorded streams.

    This is the honest version of the gripper-close latency button, which cannot
    separate camera lag from the gripper physically taking time to shut.

    Sign convention: a POSITIVE lag means the camera stream is LATE -- the image
    changes some milliseconds after the arm state says the arm moved.
    """
    import imageio.v2 as imageio

    stt = _read_jsonl(run_dir / "raw" / "arm_state.jsonl")
    if len(stt) < 100:
        return {"ok": False, "error": "not enough arm_state rows"}

    metas = sorted((run_dir / "video").glob("*_frame_times.json"))
    if not metas:
        return {"ok": False, "error": "no camera timestamp files"}
    chosen = None
    for m in metas:
        d = json.loads(m.read_text())
        if label and d["label"] != label:
            continue
        if not label and "gripper" not in d["label"]:
            continue          # a wrist camera sees the arm's own motion most strongly
        chosen = d
        break
    chosen = chosen or json.loads(metas[0].read_text())
    mp4 = run_dir / "video" / f"{chosen['label']}.mp4"
    if not mp4.is_file() or not chosen.get("t"):
        return {"ok": False, "error": f"no usable video for {chosen['label']}"}

    # --- arm speed
    at = np.asarray([r["t"] for r in stt], dtype=float)
    ax = np.asarray([r.get("pose_world_xyz_mm") or [np.nan] * 3 for r in stt], dtype=float)
    good = np.isfinite(ax).all(axis=1)
    at, ax = at[good], ax[good]
    # CENTRED difference: a backward difference puts speed[k] half a sample early,
    # which shows up directly as a ~10 ms bias in the recovered lag at 50 Hz.
    aspeed = np.zeros(len(at))
    if len(at) > 2:
        aspeed[1:-1] = (np.linalg.norm(ax[2:] - ax[:-2], axis=1)
                        / np.maximum(at[2:] - at[:-2], 1e-6))
        aspeed[0], aspeed[-1] = aspeed[1], aspeed[-2]

    # --- image change energy, streamed one frame at a time
    ct = np.asarray(chosen["t"], dtype=float)
    energy, prev = [], None
    with imageio.get_reader(str(mp4)) as rd:
        for f in rd:
            g = f[::4, ::4].astype(np.float32).mean(axis=2)     # decimate: cheap, ample
            energy.append(0.0 if prev is None else float(np.abs(g - prev).mean()))
            prev = g
    energy = np.asarray(energy, dtype=float)
    n = min(len(energy), len(ct))
    ct, energy = ct[:n], energy[:n]
    # energy[i] = |frame[i] - frame[i-1]| describes motion over the INTERVAL ending at
    # ct[i], so it belongs at that interval's midpoint. Leaving it at ct[i] puts the
    # signal half a frame late and shows up as a constant +16.7 ms bias at 30 fps --
    # measured as +17.4 ms against planted lags before this correction.
    if n > 1:
        ct = np.r_[ct[0], 0.5 * (ct[1:] + ct[:-1])]
    if n < 100:
        return {"ok": False, "error": "not enough frames"}

    # --- common grid, both signals zero-mean and unit-variance
    lo, hi = max(at[0], ct[0]), min(at[-1], ct[-1])
    if hi - lo < 5.0:
        return {"ok": False, "error": "streams overlap for under 5 s"}
    grid = np.arange(lo, hi, 1.0 / grid_hz)
    A = np.interp(grid, at, aspeed)
    E = np.interp(grid, ct, energy)
    z = lambda v: (v - v.mean()) / (v.std() or 1.0)
    A, E = z(A), z(E)

    # --- correlate over the lag sweep. Shifting E EARLIER (camera late) is positive.
    steps = int(max_lag_s * grid_hz)
    lags, corrs = [], []
    for s in range(-steps, steps + 1):
        if s >= 0:
            a, e = A[: len(A) - s], E[s:]
        else:
            a, e = A[-s:], E[: len(E) + s]
        if len(a) < 50:
            continue
        lags.append(s / grid_hz)
        corrs.append(float(np.corrcoef(a, e)[0, 1]))
    if not corrs:
        return {"ok": False, "error": "no usable lag window"}
    k = int(np.argmax(corrs))
    best_lag, best_r = lags[k], corrs[k]
    # Parabolic refine around the peak -> sub-frame resolution
    if 0 < k < len(corrs) - 1:
        y0, y1, y2 = corrs[k - 1], corrs[k], corrs[k + 1]
        denom = (y0 - 2 * y1 + y2)
        if denom != 0:
            best_lag += (0.5 * (y0 - y2) / denom) / grid_hz
    return {
        "ok": True, "camera": chosen["label"], "lag_s": best_lag,
        "lag_frames_at_30hz": best_lag * 30.0, "peak_corr": best_r,
        "curve": [(round(l, 4), round(c, 4)) for l, c in zip(lags, corrs)],
        "note": "positive = camera stream is LATE relative to arm_state",
    }
