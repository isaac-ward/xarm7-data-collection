"""Solve `arm.home_joints` from `arm.home_intent` and the workspace box.

    swoosh-home                # show what the current home is, and whether it still fits
    swoosh-home --solve        # re-solve from the intent and write it to the config
    swoosh-home --solve --move # ...and drive the arm there (MOVES THE ARM)

WHY THIS EXISTS. `home_joints` is seven numbers, and the workspace box is six more, and
nothing tied them together. Editing the box left the home outside it, and the first
clamp after re-homing then commanded a 265 mm step -- which jolted the arm hard enough
to need an emergency stop. Re-solving is a one-liner; remembering to is not.

The intent is declarative: sit at the box's centre in plan, a set distance below its
ceiling, with the gripper approaching at a given tilt and azimuth. Anything derived
from the box should be derived from the box.
"""

from __future__ import annotations

import argparse

import numpy as np

from .config import get, load_config, set_value
from .frames import R_BASE_FROM_WORLD, R_WORLD_FROM_BASE, base_to_world, world_to_base
from .kinematics import fk_chain


def target_pose(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """(world xyz mm, base rpy deg) the home pose should have, from the intent."""
    from scipy.spatial.transform import Rotation as R

    box = get(cfg, "control.workspace_box_mm")
    intent = get(cfg, "arm.home_intent", {}) or {}
    below = float(intent.get("below_ceiling_mm", 50.0))
    tilt = float(intent.get("approach_below_horizontal_deg", 45.0))
    az = float(intent.get("approach_azimuth_deg", 90.0))

    xyz = np.array([
        (float(box["x"][0]) + float(box["x"][1])) / 2.0,
        (float(box["y"][0]) + float(box["y"][1])) / 2.0,
        float(box["z"][1]) - below,
    ])

    # Approach axis: `tilt` degrees below horizontal, swung `az` degrees from +X
    # toward +Y. The jaw axis is then chosen horizontal, and the third axis follows.
    t, a = np.radians(tilt), np.radians(az)
    z_t = np.array([np.cos(t) * np.cos(a), np.cos(t) * np.sin(a), -np.sin(t)])
    z_t /= np.linalg.norm(z_t)
    y_t = np.cross([0.0, 0.0, 1.0], z_t)
    y_t /= np.linalg.norm(y_t)
    R_world_tool = np.column_stack([np.cross(y_t, z_t), y_t, z_t])
    rpy = R.from_matrix(R_BASE_FROM_WORLD @ R_world_tool).as_euler("xyz", degrees=True)
    return xyz, rpy


def describe(cfg: dict, joints: list[float]) -> dict:
    """Where a joint vector actually puts the flange, and whether it fits the box."""
    T = fk_chain(joints)[-1]
    xyz = T[:3, 3] * 1000.0
    w = base_to_world(xyz)
    z = (R_WORLD_FROM_BASE @ T[:3, :3]) @ np.array([0.0, 0.0, 1.0])
    box = get(cfg, "control.workspace_box_mm")
    inside = all(float(box[ax][0]) <= float(w[i]) <= float(box[ax][1])
                 for i, ax in enumerate("xyz"))
    return {
        "world_mm": [round(float(v), 1) for v in w],
        "approach": [round(float(v), 3) for v in z],
        "below_horizontal_deg": round(
            float(np.degrees(np.arctan2(-z[2], np.linalg.norm(z[:2])))), 1),
        "azimuth_deg": round(float(np.degrees(np.arctan2(z[1], z[0]))), 1),
        "inside_box": inside,
    }


def _park_left(cfg: dict) -> int:
    """Park the left arm at its pre-retracted home, out of the right arm's way.

    This project is right-arm only and never otherwise touches .219. Parking it is a
    courtesy to a right-arm campaign: an arm left mid-workspace is something to collide
    with, and collision_sensitivity is 0 so nothing would catch it.

    The pose is `arm.left_park` -- PO-Assembly-LEGO's HOME_DEG["left"], mirror-symmetric
    with the right arm's original home and corroborated by four copies in
    manipulation-mono. See the comment in conf/collect.yaml.
    """
    import time

    from xarm.wrapper import XArmAPI

    from .utils.safety import emit_motion_warning

    park = get(cfg, "arm.left_park") or {}
    ip = str(park.get("ip", "192.168.1.219"))
    want = [float(v) for v in park.get("joints_deg", [])]
    if len(want) != 7:
        print("arm.left_park.joints_deg must hold 7 angles")
        return 1

    print(f"\nLEFT arm @ {ip}")
    print(f"  park pose {want}")
    a = XArmAPI(ip, is_radian=False)
    time.sleep(0.8)
    if int(a.error_code or 0) == 1:
        print("  EMERGENCY STOP engaged on the left arm -- release it by hand first.")
        a.disconnect()
        return 1
    a.clean_warn()
    a.clean_error()
    try:
        a.set_collision_sensitivity(0)
    except Exception:
        pass
    a.motion_enable(enable=True)
    a.set_mode(0)
    a.set_state(0)

    # Wait for the report stream: api.angles reads all-zeros until it publishes, and
    # all-zeros would make the travel guard below meaningless.
    for _ in range(60):
        if any(abs(float(v)) > 1e-6 for v in (a.angles or [0.0])):
            break
        time.sleep(0.05)
    before = [round(float(v), 1) for v in (a.angles or [])]
    print(f"  currently {before}")
    if len(before) >= 7:
        travel = max(abs(w - b) for w, b in zip(want, before[:7]))
        print(f"  largest single joint change {travel:.0f} deg")
        # Fail closed: a huge delta means a mis-read, wrong units, or not the arm we
        # think it is. PO-Assembly guards the same way.
        if travel > 200.0:
            print("  REFUSING: further than 200 deg on one joint -- check the arm.")
            a.disconnect()
            return 1
        if travel < 0.5:
            print("  already parked; nothing to do.")
            a.disconnect()
            return 0

    emit_motion_warning(robot_name="SWOOSH LEFT", countdown=3)
    code = a.set_servo_angle(angle=want,
                             speed=float(park.get("speed_deg_s", 20.0)),
                             mvacc=float(park.get("acc_deg_s2", 150.0)),
                             wait=True, is_radian=False)
    time.sleep(0.4)
    after = [round(float(v), 1) for v in (a.angles or [])]
    err = max(abs(w - x) for w, x in zip(want, after[:7])) if len(after) >= 7 else 99.0
    print(f"  set_servo_angle -> code {code}")
    print(f"  now at    {after}")
    print(f"RESULT  left arm parked, worst joint error {err:.2f} deg"
          if code == 0 and err < 1.0 else
          f"RESULT  FAILED -- code {code}, worst joint error {err:.2f} deg")
    a.set_state(4)      # leave it stopped, not streaming
    a.disconnect()
    return 0 if (code == 0 and err < 1.0) else 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Solve arm.home_joints from arm.home_intent and the workspace box.")
    ap.add_argument("--solve", action="store_true",
                    help="re-solve and write arm.home_joints")
    ap.add_argument("--move", action="store_true",
                    help="with --solve, drive the arm to the new home (MOVES THE ARM)")
    ap.add_argument("--park-left", action="store_true",
                    help="park the LEFT arm at arm.left_park out of the way of a "
                         "right-arm campaign (MOVES THE LEFT ARM)")
    args = ap.parse_args()

    if args.park_left:
        return _park_left(load_config())

    cfg = load_config()
    box = get(cfg, "control.workspace_box_mm")
    print(f"\nbox            x{[float(v) for v in box['x']]} "
          f"y{[float(v) for v in box['y']]} z{[float(v) for v in box['z']]}")

    cur = list(get(cfg, "arm.home_joints"))
    d = describe(cfg, cur)
    print(f"\ncurrent home   {[round(j, 1) for j in cur]}")
    print(f"  world        {d['world_mm']} mm")
    print(f"  approach     {d['below_horizontal_deg']} deg below horizontal, "
          f"azimuth {d['azimuth_deg']} deg")
    print(f"  inside box   {d['inside_box']}"
          + ("" if d["inside_box"] else "   <-- DISAGREES WITH THE BOX, re-solve"))

    want_xyz, want_rpy = target_pose(cfg)
    print(f"\nintent wants   {[round(float(v), 1) for v in want_xyz]} mm, "
          f"approach {get(cfg, 'arm.home_intent.approach_below_horizontal_deg', 45)} deg "
          f"below horizontal at azimuth "
          f"{get(cfg, 'arm.home_intent.approach_azimuth_deg', 90)} deg")
    if not args.solve:
        print("\n(--solve to re-derive and write it)")
        return 0

    from .arm import RightArm

    arm = RightArm(cfg)
    arm.connect()
    pose = [*world_to_base(want_xyz).tolist(), *want_rpy.tolist()]
    code, js = arm.api.get_inverse_kinematics(
        pose, input_is_radian=False, return_is_radian=False)
    if code != 0 or not js:
        print(f"\nIK failed with code {code} -- is that pose reachable?")
        arm.shutdown()
        return 1
    js = [round(float(v), 1) for v in js[:7]]
    nd = describe(cfg, js)
    err = float(np.linalg.norm(np.asarray(nd["world_mm"]) - want_xyz))
    print(f"\nsolved home    {js}")
    print(f"  world        {nd['world_mm']} mm  (off by {err:.1f} mm)")
    print(f"  approach     {nd['below_horizontal_deg']} deg below horizontal, "
          f"azimuth {nd['azimuth_deg']} deg")
    print(f"  inside box   {nd['inside_box']}")
    if not nd["inside_box"]:
        print("  REFUSING: the solution is outside the box it was solved from")
        arm.shutdown()
        return 1
    set_value("arm.home_joints", js)
    print("\nwritten to arm.home_joints")

    if args.move:
        travel = max(abs(a - b) for a, b in zip(js, arm.joints_now()))
        print(f"\nMOVING -- largest single joint change {travel:.0f} deg")
        arm.cfg = load_config()
        arm.go_home()
        arm.start_streaming()
        print(f"at {[round(a, 1) for a in arm.joints_now()]}")
    else:
        print("press Y on the pad (or --move) to drive there")
    arm.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
