"""xArm7 forward kinematics and a small numerical IK.

The FK table here is the SAME modified-DH table as `vendor/armfk.js`; the two are
cross-checked against each other in tests/test_kinematics.py so the 3D view and the
simulator cannot drift apart.

WHAT THIS IS AND IS NOT FOR. The real robot's IK lives in the xArm controller --
`set_servo_cartesian` hands it a Cartesian pose and the controller solves it. Nothing
here is used to command the real arm. This exists so that:
  * the simulator can produce plausible joint angles for a commanded pose, and
  * the dashboard's FK-check button can compare FK(joints) against the pose the
    controller itself reports, which is the only way to confirm the table is right.

UNTIL THAT CHECK PASSES ON THE ROBOT, TREAT THE TABLE AS UNVERIFIED.
"""

from __future__ import annotations

import numpy as np

# alpha_{i-1}, a_{i-1}, d_i   (radians, metres) -- UFACTORY's published xArm7 table
DH = np.array([
    [0.0,          0.0,     0.267],
    [-np.pi / 2,   0.0,     0.0],
    [np.pi / 2,    0.0,     0.293],
    [np.pi / 2,    0.0525,  0.0],
    [np.pi / 2,    0.0775,  0.3425],
    [-np.pi / 2,   0.0,     0.0],
    [np.pi / 2,    0.076,   0.097],
], dtype=np.float64)


def _link(alpha: float, a: float, d: float, theta: float) -> np.ndarray:
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([
        [ct,      -st,      0.0,  a],
        [st * ca,  ct * ca, -sa, -d * sa],
        [st * sa,  ct * sa,  ca,  d * ca],
        [0.0,      0.0,      0.0, 1.0],
    ])


def fk_chain(joints_deg) -> list[np.ndarray]:
    """-> 8 cumulative 4x4 transforms: base, then one per joint (last = flange)."""
    q = np.radians(np.asarray(joints_deg, dtype=np.float64).ravel()[:7])
    out = [np.eye(4)]
    T = out[0]
    for i in range(7):
        T = T @ _link(DH[i, 0], DH[i, 1], DH[i, 2], q[i])
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
