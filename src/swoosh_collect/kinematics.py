"""xArm7 forward kinematics and a small numerical IK.

The chain here is the SAME chain as `vendor/armfk.js`; the two are cross-checked against
each other in tests/test_kinematics.py so the 3D view and the simulator cannot drift
apart.

WHAT THIS IS AND IS NOT FOR. The real robot's IK lives in the xArm controller --
`set_servo_cartesian` hands it a Cartesian pose and the controller solves it. Nothing
here is used to command the real arm. This exists so that:
  * the simulator can produce plausible joint angles for a commanded pose, and
  * the dashboard's FK-check button can compare FK(joints) against the pose the
    controller itself reports, which is the only way to confirm the table is right.

CHECKED ON THE ROBOT 2026-09-08: this chain's flange sits 2.5 mm from the pose the
controller itself reports, at a pose far from home. The modified-DH table this replaced
was 28.0 mm out -- it had the link twists of joints 6 and 7 swapped, which left the
wrist mis-rotated in the 3D view.
"""

from __future__ import annotations

import numpy as np

# Straight out of mujoco_menagerie `ufactory_xarm7` (xarm7_nohand.xml) -- the same model
# the vendored STL meshes come from, so geometry and rendering cannot disagree. Each row
# is a body: offset by `pos`, apply the body's fixed +/-90 deg twist about X, then rotate
# about the joint's own axis (MuJoCo's default, +Z).
#
#   T_i = T_{i-1} @ Translate(pos) @ Rx(twist) @ Rz(theta_i)
#
# The MJCF's 0.12 m pedestal under link_base is deliberately excluded: the controller's
# base frame is the mounting flange, and get_position reports in that frame.
# pos_x, pos_y, pos_z (metres), twist about X (radians)
CHAIN = np.array([
    [0.0,     0.0,     0.267,  0.0],          # link1
    [0.0,     0.0,     0.0,   -np.pi / 2],    # link2
    [0.0,    -0.293,   0.0,    np.pi / 2],    # link3
    [0.0525,  0.0,     0.0,    np.pi / 2],    # link4
    [0.0775, -0.3425,  0.0,    np.pi / 2],    # link5
    [0.0,     0.0,     0.0,    np.pi / 2],    # link6
    [0.076,   0.097,   0.0,   -np.pi / 2],    # link7 / flange
], dtype=np.float64)


def _link(pos, twist: float, theta: float) -> np.ndarray:
    ca, sa = np.cos(twist), np.sin(twist)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([
        [ct,      -st,      0.0, pos[0]],
        [ca * st,  ca * ct, -sa, pos[1]],
        [sa * st,  sa * ct,  ca, pos[2]],
        [0.0,      0.0,      0.0, 1.0],
    ])


def fk_chain(joints_deg) -> list[np.ndarray]:
    """-> 8 cumulative 4x4 transforms: base, then one per joint (last = flange)."""
    q = np.radians(np.asarray(joints_deg, dtype=np.float64).ravel()[:7])
    out = [np.eye(4)]
    T = out[0]
    for i in range(7):
        T = T @ _link(CHAIN[i, :3], CHAIN[i, 3], q[i])
        out.append(T)
    return out


def fk_pose_mm(joints_deg) -> np.ndarray:
    """Flange position in MILLIMETRES, in the arm base frame (matches get_position)."""
    return fk_chain(joints_deg)[-1][:3, 3] * 1000.0


def ik_position(target_mm, seed_deg, iters: int = 60, damping: float = 0.06):
    """Damped least squares on POSITION only. -> joint angles in degrees.

    Position-only is deliberate: it is enough for a believable simulated arm and for
    the 3D view, and it cannot fail the way a full 6-DoF solve does near singularities.
    The real robot never uses this.
    """
    q = np.radians(np.asarray(seed_deg, dtype=np.float64).ravel()[:7].copy())
    target = np.asarray(target_mm, dtype=np.float64) / 1000.0
    for _ in range(iters):
        chain = fk_chain(np.degrees(q))
        p = chain[-1][:3, 3]
        err = target - p
        if np.linalg.norm(err) < 1e-4:
            break
        # numerical Jacobian: 7 columns, one per joint
        J = np.zeros((3, 7))
        eps = 1e-6
        for i in range(7):
            dq = q.copy()
            dq[i] += eps
            J[:, i] = (fk_chain(np.degrees(dq))[-1][:3, 3] - p) / eps
        JT = J.T
        q += JT @ np.linalg.solve(J @ JT + (damping ** 2) * np.eye(3), err)
    return np.degrees(q)
