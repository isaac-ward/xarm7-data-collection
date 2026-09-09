"""FK sanity, and the Python chain must equal the JavaScript one.

The 3D view uses vendor/armfk.js and the simulator uses kinematics.py. If the two ever
drift apart, the 3D scene stops showing what the simulator thinks the arm is doing -- so
compare them literally. This test earned its keep: it caught the JS side being ported to
the MJCF chain while Python still held the old (28 mm wrong) DH table.
"""
import re, sys, pathlib
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from swoosh_collect.kinematics import CHAIN, fk_chain, fk_pose_mm, ik_position


def test_js_and_python_chains_match():
    """Parse the CHAIN literal out of armfk.js and compare it to the Python one.

    JS carries the twist in DEGREES for readability; Python in radians. Compare in
    degrees so the units are explicit rather than assumed.
    """
    js = (ROOT / "src/swoosh_collect/vendor/armfk.js").read_text()
    start = js.index("export const CHAIN")
    body = js[start: js.index("];", start)]
    rows = re.findall(
        r"pos:\s*\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]\s*,"
        r"\s*twist:\s*([-+\d.]+)", body)
    assert len(rows) == 7, f"parsed {len(rows)} rows from armfk.js CHAIN"
    js_chain = np.array([[float(a), float(b), float(c), float(t)] for a, b, c, t in rows])
    py_chain = np.column_stack([CHAIN[:, :3], np.degrees(CHAIN[:, 3])])
    assert np.abs(js_chain - py_chain).max() < 1e-9, (
        f"chains differ:\nJS:\n{js_chain}\nPython:\n{py_chain}")


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
