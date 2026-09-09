"""The camera-failure guards, which the adversarial review found absent.

#4  a camera that dies mid-run must not yield a silently truncated episode
#5  a configured camera with no usable stream must not be quietly dropped

Both were verified to export "[ok]" before these guards existed.
"""
import json, pathlib, shutil, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from test_roundtrip import build
from swoosh_collect.config import load_config
import swoosh_collect.export_lerobot as EX

base = pathlib.Path("/tmp/camguard"); shutil.rmtree(base, ignore_errors=True)

def fresh(tag):
    cfg = load_config()
    d = base / tag; d.mkdir(parents=True)
    return build(d, cfg, uptime=12.0), cfg

def vid(run, label):
    return run / "video" / f"{label}_frame_times.json"

res = []
def check(name, run, cfg, want_refused):
    got = EX.export_run(run, cfg)
    res.append((name, (got is None) == want_refused))

run, cfg = fresh("ok")
labels = list(cfg["cameras"]["expected_labels"])
victim = labels[-1]
check("baseline: untouched run exports", run, cfg, False)

run, cfg = fresh("nomp4"); (run / "video" / f"{victim}.mp4").unlink()
check("#5 mp4 missing -> refused", run, cfg, True)

run, cfg = fresh("nots")
d = json.loads(vid(run, victim).read_text()); d["t"] = []
vid(run, victim).write_text(json.dumps(d))
check("#5 no timestamps -> refused", run, cfg, True)

run, cfg = fresh("absent")
vid(run, victim).unlink(); (run / "video" / f"{victim}.mp4").unlink()
check("#5 camera absent entirely -> refused", run, cfg, True)

run, cfg = fresh("died")
d = json.loads(vid(run, victim).read_text())
d["stopped_early"] = True; d["stopped_at_s"] = float(d["t"][len(d["t"]) // 3])
vid(run, victim).write_text(json.dumps(d))
check("#4 stopped_early flag -> refused", run, cfg, True)

run, cfg = fresh("trunc")
d = json.loads(vid(run, victim).read_text())
d["t"] = d["t"][:len(d["t"]) // 3]        # died at 1/3 and the flag was lost too
vid(run, victim).write_text(json.dumps(d))
check("#4 truncated, flag lost -> refused by coverage", run, cfg, True)

print("\n" + "=" * 60)
bad = sum(not ok for _, ok in res)
for name, ok in res:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print("=" * 60)
print(f"  {len(res) - bad}/{len(res)} guards behave as intended")
sys.exit(1 if bad else 0)
