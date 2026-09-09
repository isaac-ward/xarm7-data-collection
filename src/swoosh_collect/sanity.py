"""Pre-flight check for the right arm. Run this on a cold start, and to clear errors.

    swoosh-sanity                 checks only, no motion
    swoosh-sanity --home          also move to the home pose
    swoosh-sanity --verify-frame  also jog +X/+Y/+Z in WORLD coords and report what moved

`--verify-frame` is the important one before your first collection. The 45-degree
mount rotation is a well-formed rotation whether or not its signs are right, so no
amount of unit testing can prove it matches the physical robot -- only jogging the arm
can. Do it once, confirm forward/left/up, and the action labels in every dataset after
that are trustworthy.
"""

from __future__ import annotations

import argparse
import json
import sys

from .arm import RightArm, probe_reachable
from .config import get, load_config
from .frames import check_matrix, verify_against_arm

GREEN, RED, YELLOW, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[0m"


def _ok(msg: str) -> None:
    print(f"  {GREEN}PASS{RESET} {msg}", flush=True)


def _bad(msg: str, hint: str = "") -> None:
    print(f"  {RED}FAIL{RESET} {msg}", flush=True)
    if hint:
        print(f"       {hint}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Right-arm bring-up check.")
    ap.add_argument("--home", action="store_true", help="move to the home pose (MOVES)")
    ap.add_argument("--verify-frame", action="store_true",
                    help="jog each world axis and report what moved (MOVES)")
    ap.add_argument("--gripper", action="store_true", help="cycle the gripper")
    args = ap.parse_args()

    cfg = load_config()
    ip = str(get(cfg, "arm.ip"))
    fails = 0

    print(f"\nright arm @ {ip}\n")

    c = check_matrix()
    if abs(c["det"] - 1.0) < 1e-9 and c["orthonormality_error"] < 1e-9:
        _ok(f"frame matrix is a rotation (base z sits "
            f"{c['base_z_tilt_from_vertical_deg']:.1f} deg off vertical)")
    else:
        fails += 1
        _bad(f"frame matrix is not a rotation: {c}")

    if probe_reachable(ip):
        _ok(f"TCP {ip}:502 reachable")
    else:
        fails += 1
        _bad(f"TCP {ip}:502 unreachable",
             f"controller powered? xArm Studio at http://{ip} should load. "
             f"Host needs {get(cfg, 'arm.host_ip_cidr')} on {get(cfg, 'arm.nic')}.")
        print(f"\n{RED}cannot continue without the arm{RESET}\n")
        return 1

    arm = RightArm(cfg)
    try:
        arm.connect()
        _ok("connected, errors cleared, motion enabled")
    except Exception as exc:
        _bad(f"connect failed: {exc}")
        return 1

    try:
        st = arm.read_state(0.0)
        if st.joints_deg:
            _ok(f"joints (deg): {[round(v, 1) for v in st.joints_deg]}")
        else:
            fails += 1
            _bad("get_servo_angle returned nothing")
        if st.pose_base:
            _ok(f"pose base  (mm/deg): {[round(v, 1) for v in st.pose_base]}")
            _ok(f"pose world (mm):     {[round(v, 1) for v in st.pose_world_xyz]}")
        else:
            fails += 1
            _bad("get_position returned nothing")
        try:
            wo = list(getattr(arm.api, "world_offset", [0] * 6) or [0] * 6)
            if all(abs(float(v)) < 1e-6 for v in wo):
                _ok("controller world offset is zero (we do the 45-deg rotation ourselves)")
            else:
                fails += 1
                _bad(f"controller world offset is {[round(float(v), 3) for v in wo]}",
                     "clear it in xArm Studio. frames.py already rotates base<->world, so a "
                     "controller offset would apply the rotation TWICE.")
        except Exception:
            pass

        if st.error_code:
            fails += 1
            _bad(f"controller error code {st.error_code}",
                 "this run cleared errors; if it persists, check the e-stop")
        else:
            _ok("no latched controller error")
        # Ask the device directly. st.gripper_pos is the value cached by the gripper
        # WORKER thread, and this one-shot check never starts that worker -- so it was
        # always NaN and this always warned "gripper did not report a position", on a
        # rig where the live collector reads 850 perfectly well. A check that cries
        # wolf on every bring-up is worse than no check.
        try:
            gcode, gpos = arm.api.get_gripper_position()
        except Exception as exc:
            gcode, gpos = -1, None
            print(f"  {YELLOW}WARN{RESET} gripper read raised: {exc}")
        if gcode == 0 and gpos is not None:
            _ok(f"gripper position {float(gpos):.0f}")
        else:
            print(f"  {YELLOW}WARN{RESET} gripper did not report a position "
                  f"(code {gcode}) -- expected only if none is attached")

        if args.home:
            arm.go_home()
            _ok("reached home pose")

        if args.gripper:
            op = int(get(cfg, "gripper.open_position", 850))
            cl = int(get(cfg, "gripper.closed_position", 0))
            for pos in (cl, op):
                arm.api.set_gripper_position(pos, wait=True, auto_enable=True)
            _ok("gripper cycled closed -> open")

        if args.verify_frame:
            print(f"\n{YELLOW}jogging each WORLD axis 20 mm -- watch the arm{RESET}")
            from .utils.safety import emit_motion_warning

            emit_motion_warning(robot_name="SWOOSH", countdown=3)
            arm.api.set_mode(0)
            arm.api.set_state(0)
            res = verify_against_arm(arm.api)
            # Persist it: swoosh-collect refuses to start unless a verification exists
            # for THIS matrix hash, so the check has to leave a record.
            from .provenance import save_frame_verification
            save_frame_verification(res)
            for name in ("+X_forward", "+Y_left", "+Z_up"):
                r = res[name]
                (_ok if r["correct"] else _bad)(
                    f"{name}: measured {r['measured_world_mm']} mm, "
                    f"dominant {r['dominant_axis']}"
                )
                if not r["correct"]:
                    fails += 1
            if res["all_correct"]:
                print(f"\n  {GREEN}frame verified against the real arm{RESET} -- "
                      f"world jogs move the EE the way they should")
            else:
                print(f"\n  {RED}FRAME IS WRONG.{RESET} Fix R_WORLD_FROM_BASE in "
                      f"frames.py before collecting: every action label depends on it.")
            print(json.dumps(res, indent=2))
    finally:
        arm.shutdown()

    # RESULT-prefixed so the dashboard's terminal highlights it and it is the last
    # thing on screen -- the panel auto-scrolls, so a summary buried above the
    # bookkeeping reads as no summary at all.
    print(f"\nRESULT  {'all checks passed' if not fails else f'{fails} check(s) FAILED'}\n")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
