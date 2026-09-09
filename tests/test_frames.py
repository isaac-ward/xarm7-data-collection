"""The frame matrix must be a real rotation AND mean what the docstring says."""
import numpy as np
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from swoosh_collect.frames import (
    R_WORLD_FROM_BASE, base_to_world, check_matrix, world_to_base,
)


def test_is_a_rotation():
    c = check_matrix()
    assert abs(c["det"] - 1.0) < 1e-12, c
    assert c["orthonormality_error"] < 1e-12, c
    assert c["cross_error"] < 1e-12, c


def test_base_z_is_45_deg_off_vertical():
    """The whole point of the mount: base +Z sits 45 deg from world up."""
    assert abs(check_matrix()["base_z_tilt_from_vertical_deg"] - 45.0) < 1e-9


def test_round_trip():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(50, 3))
    assert np.abs(base_to_world(world_to_base(v)) - v).max() < 1e-12


def test_lengths_preserved():
    rng = np.random.default_rng(1)
    v = rng.normal(size=(50, 3))
    assert np.abs(np.linalg.norm(world_to_base(v), axis=1) - np.linalg.norm(v, axis=1)).max() < 1e-12


def test_base_x_is_world_forward():
    """Column 0 must be world +X: the mount tilts about the forward axis only."""
    assert np.abs(R_WORLD_FROM_BASE[:, 0] - np.array([1.0, 0.0, 0.0])).max() < 1e-12


def test_ground_plane_motion_needs_both_base_y_and_z():
    """A purely horizontal world move must use BOTH tilted base axes -- this is the
    bug the whole module exists to prevent. If someone 'simplifies' the matrix to
    identity this test fails."""
    horizontal = world_to_base(np.array([0.0, 1.0, 0.0]))   # world +Y (left)
    assert abs(horizontal[1]) > 0.5 and abs(horizontal[2]) > 0.5, horizontal
    assert abs(horizontal[0]) < 1e-12


def test_vertical_motion_is_split_across_two_base_axes():
    up = world_to_base(np.array([0.0, 0.0, 1.0]))
    assert abs(abs(up[1]) - np.sqrt(0.5)) < 1e-12
    assert abs(abs(up[2]) - np.sqrt(0.5)) < 1e-12


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
