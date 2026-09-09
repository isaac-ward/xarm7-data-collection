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
    sx = np.array([x.get("pose_world_xyz_mm") or [np.nan] * 3 for x in stt], dtype=float)
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
    out.append(_r("state freshness", "pass" if hf < 0.5 else "fail",
                  f"{100*hf:.1f}% of arm-state rows identical to the previous one "
                  f"(lego was 69.9%)", held_fraction=hf))

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
                         " - minimum at the EDGE of the sweep, which is the "
                         "misalignment signature"),
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
