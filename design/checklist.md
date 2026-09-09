# xarm7-data-collection — build checklist

Xbox-controller teleop + world-model data collection for **Swoosh's RIGHT ARM ONLY**.
Every line below traces to something Isaac asked for. Ticked as delivered.

Source of truth for hardware facts: `sisl/manipulation-mono` (full read done 2026-09-08).

---

## Decisions locked (asked and answered)

| # | Decision | Answer |
|---|---|---|
| D1 | Stick semantics | Sticks **modify the desired EE position**; releasing holds it. `target += stick * rate * dt`. No spring-back. |
| D2 | Motion frame | **World/ground frame**, compensating the 45° mount tilt. |
| D3 | Output format | **Raw streams (authoritative) + LeRobot v2.1 export.** |
| D4 | ROS | **Dropped entirely.** No ROS 2 / Quest2ROS in the image. |
| D5 | Camera identity | **Pinned by USB port path**, stable across reboots/replugs. |
| D6 | Sync rate | **30 Hz** (matches lego_assemblies, so quickdraw configs carry over). |
| D7 | Gripper | **Proportional** — trigger 0–1 mapped continuously onto gripper travel. |
| D8 | Safety | **Workspace box + stop button.** No deadman. |
| D9 | Xbox link | **Wired USB.** Matched by device name, not a fixed event number. |
| D10 | Between episodes | **Arm stays exactly where it is.** No auto re-home. **Y = re-home on request**, behind the countdown banner. |
| D11 | Machine | **Same lab laptop**, NIC `enp129s0`. |
| D12 | Dashboard | ~~rich TUI~~ **superseded**: a **lightweight localhost web server** (stdlib only) opening a Chrome tab. A TUI cannot show camera feeds. |
| D13 | Dashboard layout | Page split in half. **Left**: campaign name, "new campaign" button, list of runs. **Right**: live controller render, proprio, 4 camera feeds. **Clicking a run replays it, synchronized, on the right**, temporarily replacing live. |

---

## P0 — Repo scaffolding
- [x] Rename the `template` package to the real project package
- [x] `pyproject.toml`: deps + console scripts, `xarm-python-sdk` pinned to **v1.18.4** (master's broken gitlink breaks Docker builds)
- [x] `.gitignore`: ignore `campaigns/` and `logs/`
- [x] `design/checklist.md` (this file), kept ticked as work lands

## P1 — Hardware bring-up (right arm only)
- [x] Right arm at **192.168.1.199**; laptop takes **192.168.1.100/24** on **enp129s0** via `NET_ADMIN`
- [x] Connect sequence: `clean_warn` → `clean_error` → `set_collision_sensitivity` **before** `motion_enable` → `set_mode(0)`/`set_state(0)`
- [x] `collision_sensitivity: 0` default (repo runs 0; gripper mass isn't in the payload model, 1 spuriously trips error 31)
- [x] Gripper init: `set_gripper_mode(0)`, `set_gripper_enable(True)`, `set_gripper_speed`
- [x] Move to right-arm home `[4, -37, -3, 49, -7, 16, -40]` deg @ 20 deg/s before streaming
- [x] Switch to `set_mode(1)`/`set_state(0)` for `set_servo_cartesian` streaming
- [x] Pre-motion `SWOOSH IS ABOUT TO MOVE` countdown banner (lifted from `utils/safety.py`)
- [x] Sanity command: reachability probe, firmware/state, clear errors, home, gripper cycle

## P2 — 45° mount compensation
- [x] Encode base↔world rotation for the right arm (base Z is 45° off vertical, tilted right)
- [ ] **Verify empirically before trusting it**: command +X/+Y/+Z world jogs, confirm the EE moves forward/left/up on the real arm
- [x] Make the matrix a config value, not a literal buried in code

## P3 — Xbox controller input
- [x] Read the pad on Linux (evdev), no ROS. **Wired USB**, matched by device name so the event number can move
- [x] **Left stick** → planar motion in the ground-parallel plane
- [x] **Right stick Y** → raise/lower that plane
- [x] **Right stick X** → rotate EE about the world Z axis
- [x] **Right trigger** → gripper, proportional
- [x] **A** → start episode recording
- [x] **B** → stop episode recording
- [x] **Y** → re-home to the starting pose, on request, behind the `SWOOSH IS ABOUT TO MOVE` countdown
- [x] **Stop button** → clean shutdown (arm holds where it is)
- [x] Deadzone + rate scaling per axis, all configurable
- [x] **No deadman switch** anywhere

## P4 — Control loop
- [x] End-effector positioning control via `set_servo_cartesian` at 100 Hz
- [x] Integrate stick deflection into a running target pose (D1)
- [x] Clamp target to a configurable **workspace box** in world coords (D8)
- [x] Handle `servo_code == 1` (controller HAS_ERROR) without spamming
- [x] Teleop works whether or not an episode is recording

## P5 — Recording & synchronisation
Everything on one monotonic clock, logged at native rate, each stream separately.
- [x] **Xbox controller inputs** — the world-model action signal, logged raw
- [x] **Commanded EE pose** — what we asked for
- [x] **Commanded xArm inputs** — exactly what was handed to the SDK
- [x] **Measured EE pose** — best estimate of where the arm actually is
- [x] **Joint angles** (7)
- [x] **Gripper closure amount**
- [x] Other xArm telemetry: state, error/warn codes, mode
- [x] **4 camera streams** with per-frame timestamps
- [x] All three input families kept **separate**, never merged into one "action"
- [x] Per-episode boundaries from A/B, written to a log folder

## P6 — Cameras
- [x] Discover Arducams via `/sys/class/video4linux/*/name`, one per physical USB device
- [x] **Pin labels by USB port path** (D5): 2 scene + 2 right-gripper
- [x] Discovery command printing each camera's path + live preview, to name them once
- [x] MJPG, 30 fps target (repo defaults 20 but notes MJPG sustains 30+)
- [x] Warmup frames + staggered opens (USB 2.0 hub bandwidth)
- [x] Per-frame timestamps on the shared clock

## P7 — Export
- [x] LeRobot v2.1 dataset at **30 Hz** (D6)
- [x] Resample raw → 30 Hz grid **honestly** (nearest-in-time, no invented samples)
- [x] Record the raw→export time offset per episode; never assume shared origins
- [ ] Sanity check: commanded pose should predict measured pose ~1 servo lag later

## P8 — Campaigns & dashboard *(new)*
- [x] `campaigns/` directory, **gitignored**
- [x] Create a **named campaign** → auto-creates its folder
- [x] One folder per campaign; one subfolder per run
- [x] **Lightweight localhost server** (stdlib `http.server`), auto-opens a Chrome tab
- [x] Page split in half, left/right
- [x] LEFT: campaign name at the top
- [x] LEFT: **start-new-campaign button** (creates the folder under `campaigns/`)
- [x] LEFT: every run in the campaign listed as an element
- [x] RIGHT: live **controller rendering** -- joysticks, buttons, triggers
- [x] RIGHT: live **proprio** -- joint angles, end-effector position
- [x] RIGHT: live **4 camera feeds** (MJPEG, throttled so it can't starve the servo loop)
- [x] Everything on the right updates **live**
- [x] **Click a run -> synchronized playback** of its saved inputs on the right, temporarily taking over from live
- [x] Playback is client-side, so it cannot perturb an in-progress collection
- [x] Static HTML index per campaign for offline review (independent of the server)

### Run lifecycle in the UI *(new)*
- [x] **A** -> the run appears on the campaign side immediately, marked **in progress**
- [x] **B** -> its status becomes **processing**
- [x] Processing finishes -> the tag **disappears**, and only then is playback allowed
- [x] Clicking a run that is not finished processing makes it **wiggle** like an error
- [x] **Open campaign folder** button at the top
- [x] **Open run folder** button on each run
- [x] Runs show a **timestamp**
- [x] Show **A→B stopped runs so far** in the campaign
- [x] **List all individual runs** in the campaign
- [x] Per-run **time-synced summary MP4** with all **4 camera feeds**
- [x] Per-run **matplotlib plot: inputs over time**
- [x] Per-run **matplotlib plot: kinematics over time**

## P9 — Packaging
- [x] **Dockerised** — no ROS; slim base + uv
- [x] **uv** for dependency management
- [x] compose service(s) with `/dev` mount, `group_add: video`, `device_cgroup_rules: c 81:* rmw` for cameras
- [x] `NET_ADMIN` + `network_mode: host` + idempotent `ip addr add` for the arm subnet
- [ ] **"Drop it on the Ubuntu machine and hey presto"** — one command from clone to collecting
- [x] README: power-up order, camera naming, controller map
- [x] README: **explicit "start a new campaign" section** -- the command, and that it creates a new folder under the gitignored `campaigns/`

---

## Open questions (blocking nothing yet, but I need answers before P6/P9 finish)
- [x] Camera naming — done in the dashboard's SANITY box (button 3), pinned by USB path
- [x] Gripper is the **same as manipulation-mono** (0 closed / 850 open) — confirmed by Isaac 2026-09-09

---

## P10 — adversarial review findings (2026-09-09)

An adversarial agent reviewed the whole pipeline against the four lego_assemblies
failure modes. It found two FATAL bugs that would have silently ruined the campaign.
I verified each claim against the code and the installed libraries before acting.

### Fixed
- [x] **FATAL: two clock origins.** JSONL rows were stamped from process start, camera
      frames from the A press. First episode of each session misaligned by the process
      uptime; every later one silently dropped by the exporter. This IS lego failure #2.
      Now every stream shares `t_loop0`, and the export REFUSES a run whose stream
      starts differ by >5 s instead of emitting a confidently-wrong dataset.
- [x] **FATAL: exported video was not on the export grid.** The raw mp4 was copied
      verbatim with an `.npy` sidecar no LeRobot loader reads, so a loader seeking by
      timestamp got a per-camera constant offset plus drift. Now re-encoded on the grid:
      mp4 frame i IS parquet row i, by construction.
- [x] **A/B/Y never fired.** `evdev.ecodes.BTN[0x130]` is a TUPLE; the code only
      handled `list`, so only BTN_START worked. Verified against evdev 2.0.0.
- [x] **Re-home was bound to the wrong button.** `BTN_NORTH` is the physical X (0x133);
      `BTN_WEST` is Y (0x134). Now `BTN_WEST`; X stays unbound as intended.
- [x] **Pressing A stalled the servo loop ~1 s** (the 4x0.25 s camera USB stagger ran
      inside the control loop), then the loop burst-replayed the backlog with a full
      10 ms of stick each. Cameras now open on a worker; `dt` is MEASURED and clamped;
      `next_tick` re-anchors after every blocking branch.
- [x] **Pad unplugged kept driving the arm** on the last stick values, recorded as if
      the operator were holding them. `snapshot()` now returns neutral when disconnected.
- [x] **`action.commanded_pose_world` carried BASE-frame rpy** under names
      roll/pitch/yaw -- a 45-degree lie inside a column name. Split into a world column
      (xyz + yaw_world_deg + gripper) and `action.commanded_rpy_base`.
- [x] **No validity flags in the export** (lego failure #4 inverted). Added a `flags`
      column: clamped_by_workspace, servo_code, arm_error_code, pad_connected.
- [x] **Loop health was unrecorded.** New `tick` raw stream: dt, tick number, servo
      code, pad state -- so a degraded loop rate is visible instead of silently
      changing what `action` means.
- [x] Trigger deadzone (a pad resting at 0.02 flooded the slow modbus gripper link).
- [x] Camera `CAP_PROP_BUFFERSIZE=1`, actual properties recorded, timestamp appended
      only after a successful encode, mp4 frame count reconciled against timestamps.
- [x] **End-to-end round-trip test** (`tests/test_roundtrip.py`) -- the test whose
      absence let both fatal bugs through. Injects a known square wave, exports, and
      asserts the action survives AND that video frame i is row i. Includes a
      regression test that a mismatched-origin run is refused.

### Still open from the review
- [ ] Re-encode/summarise in a SUBPROCESS -- `summarize.py` loads whole videos into RAM
      in-process (a 5-min episode is ~8 GB/camera) and contends for the GIL with the
      control loop
- [ ] Record a config/git/matrix/TCP-offset snapshot into run.json, and refuse to
      collect without a recent `--verify-frame` result for the current matrix
- [ ] Workspace box corners are not reachability-checked; target can run away while
      the arm is frozen (re-anchor when tracking error persists)
- [ ] Move gripper I/O off the control thread (modbus round trips are 5-15 ms)
- [ ] `swoosh-validate`: per-run held-value fraction, rate histogram, commanded->measured
      cross-correlation lag, camera/robot lag, frame-count reconciliation
- [ ] Refuse to start an episode unless every expected camera is streaming
- [ ] Export `episodes_stats` for action/state (LeRobot normalisation needs them)
- [ ] Measure camera latency once and store it in meta
