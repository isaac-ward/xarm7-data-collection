"""Right-stick-X must rotate the EE about the WORLD z axis (perpendicular to the
ground), not about the arm's tilted base z. With a 45-degree mount those differ by
45 degrees, so getting it wrong is very visible on the real robot."""
import sys, pathlib
import numpy as np
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from swoosh_collect.frames import R_BASE_FROM_WORLD, R_WORLD_FROM_BASE


def compose(rpy_home_base, yaw_world_deg):
    """Exactly the composition collect.py performs each tick."""
    R_home_base = R.from_euler("xyz", rpy_home_base, degrees=True)
    R_home_world = R.from_matrix(R_WORLD_FROM_BASE @ R_home_base.as_matrix())
    R_yaw_world = R.from_euler("z", yaw_world_deg, degrees=True)
    return R.from_matrix(R_BASE_FROM_WORLD @ (R_yaw_world * R_home_world).as_matrix())


HOME = [180.0, 0.0, 0.0]   # a representative xArm wrist-down orientation


def test_zero_yaw_is_identity():
    got = compose(HOME, 0.0).as_euler("xyz", degrees=True)
    assert np.abs(np.array(got) - np.array(HOME)).max() < 1e-9, got


def test_yaw_is_about_world_z():
    """The delta between home and yawed, expressed in WORLD, must be a pure rotation
    about +Z of exactly the commanded angle."""
    for ang in (5.0, 30.0, -45.0, 90.0):
        home_w = R_WORLD_FROM_BASE @ R.from_euler("xyz", HOME, degrees=True).as_matrix()
        got_w = R_WORLD_FROM_BASE @ compose(HOME, ang).as_matrix()
        delta = R.from_matrix(got_w @ home_w.T)
        rotvec = delta.as_rotvec(degrees=True)
        assert abs(rotvec[0]) < 1e-8 and abs(rotvec[1]) < 1e-8, (ang, rotvec)
        assert abs(rotvec[2] - ang) < 1e-8, (ang, rotvec)


def test_not_about_base_z():
    """Guard against the naive implementation (just adding to the base yaw euler),
    which would rotate about the tilted base z and be wrong by the mount angle."""
    ang = 30.0
    correct = compose(HOME, ang).as_euler("xyz", degrees=True)
    naive = [HOME[0], HOME[1], HOME[2] + ang]
    assert np.abs(np.array(correct) - np.array(naive)).max() > 1.0, (correct, naive)


def test_accumulates():
    a = compose(HOME, 20.0)
    b = compose(HOME, 50.0)
    home_w = R_WORLD_FROM_BASE @ R.from_euler("xyz", HOME, degrees=True).as_matrix()
    d = R.from_matrix((R_WORLD_FROM_BASE @ b.as_matrix()) @ (R_WORLD_FROM_BASE @ a.as_matrix()).T)
    assert abs(d.as_rotvec(degrees=True)[2] - 30.0) < 1e-8


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
