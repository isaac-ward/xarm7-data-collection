"""FK sanity, and the Python table must equal the JavaScript one.

The 3D view uses vendor/armfk.js and the simulator uses kinematics.py. If the two
tables ever drift apart, the 3D scene stops showing what the simulator thinks the arm
is doing -- so compare them literally.
"""
import re, sys, pathlib
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from swoosh_collect.kinematics import DH, fk_chain, fk_pose_mm, ik_position


def test_js_and_python_tables_match():
    js = (ROOT / "src/swoosh_collect/vendor/armfk.js").read_text()
    body = js[js.index("export const DH"): js.index("];", js.index("export const DH"))]
    rows = re.findall(r"\[\s*([^,\]]+),\s*([^,\]]+),\s*([^,\]]+)\s*\]", body)
    assert len(rows) == 7, f"parsed {len(rows)} rows from armfk.js"
    ev = {"Math.PI": np.pi, "-Math.PI": -np.pi}
    def num(t):
        t = t.strip()
        if "Math.PI" in t:
            sign = -1.0 if t.startswith("-") else 1.0
            return sign * np.pi / float(t.split("/")[1]) if "/" in t else sign * np.pi
        return float(t)
    js_dh = np.array([[num(a), num(b), num(c)] for a, b, c in rows])
    assert np.abs(js_dh - DH).max() < 1e-12, f"tables differ:\n{js_dh}\nvs\n{DH}"


def test_zero_pose_folds_down_near_the_base():
    """All joints zero must NOT be an extended pose.

    I first asserted "straight up" here, from my own assumption, and it failed: FK puts
    the flange at about (206, 0, 120) mm -- folded down beside the base. That matches
    what manipulation-mono documents about this exact arm:

        "All-zeros is NOT safe -- J4=0 puts the elbow at full extension, which causes
         the arm to self-collide near the base."   (conf/hardware/swoosh_sanity.yaml)

    So the folded result is evidence FOR the table, not against it, and the test now
    encodes the sourced fact instead of my guess. Note this is still only consistency
    with a comment -- the FK-check button on the real robot is what verifies the table.
    """
    p = fk_pose_mm([0] * 7)
    reach = (0.267 + 0.293 + 0.3425 + 0.097) * 1000.0     # 999.5 mm if fully extended
    assert p[2] < 0.35 * reach, f"zero pose is extended, not folded: {p}"
    assert np.linalg.norm(p) < 0.45 * reach, f"zero pose is too far from the base: {p}"


def test_reach_is_physically_sensible():
    """A stretched configuration must reach roughly the arm's published span."""
    best = max(np.linalg.norm(fk_pose_mm([0, a, 0, b, 0, 0, 0]))
               for a in range(-90, 91, 10) for b in range(-90, 91, 10))
    assert 600 < best < 1100, f"max reach {best:.0f} mm is not xArm7-like"


def test_chain_transforms_are_rigid():
    for T in fk_chain([10, -20, 30, 40, -50, 60, 70]):
        R = T[:3, :3]
        assert abs(np.linalg.det(R) - 1.0) < 1e-9
        assert np.abs(R @ R.T - np.eye(3)).max() < 1e-9


def test_ik_round_trip():
    """IK(FK(q)) must land back on the same POSITION (not necessarily the same q)."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(6):
        q = rng.uniform(-60, 60, 7)
        target = fk_pose_mm(q)
        sol = ik_position(target, seed_deg=np.zeros(7))
        worst = max(worst, float(np.linalg.norm(fk_pose_mm(sol) - target)))
    assert worst < 1.0, f"worst IK position error {worst:.3f} mm"


def test_home_pose_is_plausible():
    """manipulation-mono says the home pose puts the TCP near (330, 0, 621) mm.

    Loose tolerance on purpose: their number includes a TCP offset we do not model.
    This is a smell test for a badly wrong table, not a verification -- only the robot
    can do that.
    """
    p = fk_pose_mm([4, -37, -3, 49, -7, 16, -40])
    assert 150 < p[0] < 550, p
    assert abs(p[1]) < 250, p
    assert 400 < p[2] < 800, p


if __name__ == "__main__":
    import traceback
    fs = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for f in fs:
        try:
            f(); print(f"  PASS {f.__name__}")
        except Exception:
            bad += 1; print(f"  FAIL {f.__name__}"); traceback.print_exc()
    print(f"\n{len(fs)-bad}/{len(fs)} passed")
    raise SystemExit(1 if bad else 0)
