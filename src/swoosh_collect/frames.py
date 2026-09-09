"""World <-> arm-base geometry for Swoosh's RIGHT arm.

THE RIGHT ARM IS MOUNTED 45 DEGREES CLOCKWISE FROM VERTICAL, so the xArm's own base
frame -- which is what `get_position` reports and `set_servo_cartesian` consumes -- is
NOT aligned with the room. "Move parallel to the ground" is therefore not "hold base z
constant"; it needs a rotation.

WORLD FRAME (right-handed, the one the operator thinks in):
    +X forward, away from the operator
    +Y left
    +Z up, against gravity

WHERE THE MATRIX COMES FROM. manipulation-mono fits a per-arm Quest->base transform,
`position_axes_right`, whose rows are the arm-base axes expressed in Quest/Unity
coordinates:

    base_X = [ 0.0,     0.0,    1.0   ]
    base_Y = [-0.7071,  0.7071, 0.0   ]
    base_Z = [ 0.7071,  0.7071, 0.0   ]

Unity is left-handed with +X right, +Y up, +Z forward, so world_forward = unity_Z,
world_left = -unity_X, world_up = unity_Y. Rewriting the three rows in the world frame
above gives the columns of R_WORLD_FROM_BASE below. It checks out as a rotation:
orthonormal, det = +1, and base_Y x base_Z = base_X (verified in test_frames.py).

Note manipulation-mono's own comment that `position_axes_right` has det = -1 -- that is
because it is a Unity(left-handed)->base(right-handed) map, so it must include a
handedness flip. Once the Unity axes are re-expressed in a right-handed world frame the
flip is accounted for and what remains is a pure rotation. That subtlety is exactly why
`verify_against_arm()` exists: DO NOT trust this matrix until it has been checked
against the real arm once.
"""

from __future__ import annotations

import numpy as np

SQRT1_2 = float(np.sqrt(0.5))

# Columns are the arm-base axes expressed in world coordinates.
#   column 0 = base +X in world = world +X (forward)
#   column 1 = base +Y in world = (0, +0.7071, +0.7071)
#   column 2 = base +Z in world = (0, -0.7071, +0.7071)  <- 45 deg off vertical, tilted right
R_WORLD_FROM_BASE = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, SQRT1_2, -SQRT1_2],
        [0.0, SQRT1_2, SQRT1_2],
    ],
    dtype=np.float64,
)

R_BASE_FROM_WORLD = R_WORLD_FROM_BASE.T  # a rotation, so the inverse is the transpose


def world_to_base(v_world: np.ndarray) -> np.ndarray:
    """Rotate a world-frame vector (or Nx3 stack) into the arm base frame."""
    v = np.asarray(v_world, dtype=np.float64)
    return v @ R_BASE_FROM_WORLD.T if v.ndim > 1 else R_BASE_FROM_WORLD @ v


def base_to_world(v_base: np.ndarray) -> np.ndarray:
    """Rotate an arm-base vector (or Nx3 stack) into the world frame."""
    v = np.asarray(v_base, dtype=np.float64)
    return v @ R_WORLD_FROM_BASE.T if v.ndim > 1 else R_WORLD_FROM_BASE @ v


def check_matrix() -> dict[str, float]:
    """Self-consistency of R_WORLD_FROM_BASE. Pure maths, no hardware."""
    R = R_WORLD_FROM_BASE
    bx, by, bz = R[:, 0], R[:, 1], R[:, 2]
    return {
        "det": float(np.linalg.det(R)),
        "orthonormality_error": float(np.abs(R @ R.T - np.eye(3)).max()),
        "cross_error": float(np.abs(np.cross(bx, by) - bz).max()),
        "base_z_tilt_from_vertical_deg": float(
            np.degrees(np.arccos(np.clip(np.dot(bz, [0.0, 0.0, 1.0]), -1.0, 1.0)))
        ),
    }


def verify_against_arm(arm, jog_mm: float = 20.0, settle_s: float = 1.5) -> dict:
    """Jog the REAL arm along each world axis and report what actually moved.

    This is the only thing that can catch a sign or handedness error in
    R_WORLD_FROM_BASE, because every such error is still a perfectly well-formed
    rotation and passes `check_matrix()`. Commands +X, +Y, +Z world jogs in turn,
    returning to the start pose between each, and reports the measured world-frame
    displacement.

    A CORRECT matrix gives, for each axis, a displacement whose largest component is
    that same axis with a positive sign:
        +X world -> EE moves FORWARD, away from the operator
        +Y world -> EE moves LEFT
        +Z world -> EE moves UP

    Anything else means the matrix is wrong -- fix it before collecting data, because a
    wrong frame silently corrupts every action label in the dataset.
    """
    import time

    results = {}
    code, start = arm.get_position(is_radian=False)
    if code != 0 or start is None:
        raise RuntimeError(f"get_position failed with code {code}")
    start = [float(v) for v in start]
    start_world = base_to_world(np.array(start[:3]))

    for name, axis in (("+X_forward", 0), ("+Y_left", 1), ("+Z_up", 2)):
        delta_world = np.zeros(3)
        delta_world[axis] = jog_mm
        target = list(start)
        target[:3] = (np.array(start[:3]) + world_to_base(delta_world)).tolist()
        arm.set_position(*target, speed=30, wait=True, is_radian=False)
        time.sleep(settle_s)
        code, reached = arm.get_position(is_radian=False)
        moved_world = base_to_world(np.array([float(v) for v in reached[:3]])) - start_world
        dominant = int(np.argmax(np.abs(moved_world)))
        results[name] = {
            "commanded_world_mm": delta_world.tolist(),
            "measured_world_mm": [round(float(v), 2) for v in moved_world],
            "dominant_axis": "XYZ"[dominant],
            "correct": bool(dominant == axis and moved_world[axis] > 0),
        }
        arm.set_position(*start, speed=30, wait=True, is_radian=False)
        time.sleep(settle_s)

    results["all_correct"] = all(
        v["correct"] for k, v in results.items() if isinstance(v, dict)
    )
    return results
