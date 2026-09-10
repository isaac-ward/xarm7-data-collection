"""SimulatedArm must expose everything RightArm does.

--simulate is only worth having if it exercises the real code path, which means the
stand-in has to keep up with the driver. It already fell behind once: the lab added
gripper_command_record to RightArm and the next --simulate run died on
AttributeError mid-episode, after the recording had started.
"""
import inspect, sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from swoosh_collect.arm import RightArm
from swoosh_collect.simhw import SimulatedArm, SimulatedCameraRig, SimulatedPad
from swoosh_collect.cameras import CameraRig
from swoosh_collect.xbox import XboxPad


def _public(cls):
    return {n for n, _ in inspect.getmembers(cls, callable) if not n.startswith("_")}


def test_arm_surface_complete():
    missing = _public(RightArm) - _public(SimulatedArm)
    assert not missing, f"SimulatedArm is missing: {sorted(missing)}"


def test_pad_surface_complete():
    missing = _public(XboxPad) - _public(SimulatedPad)
    assert not missing, f"SimulatedPad is missing: {sorted(missing)}"


def test_camera_rig_surface_complete():
    missing = _public(CameraRig) - _public(SimulatedCameraRig)
    assert not missing, f"SimulatedCameraRig is missing: {sorted(missing)}"


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
