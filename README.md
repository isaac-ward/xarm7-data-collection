# swoosh-collect

Xbox-controller teleoperation and world-model data collection for **Swoosh's right
xArm 7**. Drop it on the Ubuntu machine, plug in the pad, collect.

Hardware facts (arm IP, NIC, home pose, camera settings, the SDK pin) are carried over
from [`sisl/manipulation-mono`](https://github.com/sisl/manipulation-mono). No ROS: the
Quest stack needed it, an Xbox pad does not.

---

## Starting a new campaign

A **campaign** is a named collection session. Every campaign is a folder under
`campaigns/`, which is **gitignored** — nothing you record is ever committed.

```bash
swoosh-campaign new "lego-pick-place"      # creates campaigns/lego-pick-place/
```

That is the only step. The folder is made for you, and every run you record lands
inside it, numbered and timestamped:

```
campaigns/
  lego-pick-place/                  <- created by the command above
    campaign.json
    index.html                      <- offline review page
    run_0001_20260908_231500/
      run.json                      <- status, duration, how it ended
      raw/                          <- authoritative per-stream logs
      video/                        <- one mp4 per camera + frame timestamps
      summary.mp4                   <- 2x2 camera grid, time-synced
      inputs.png  kinematics.png
    run_0002_20260908_231902/
    ...
```

You can also make one **from the dashboard** — the `+ New campaign` button, top left —
which creates the same folder and switches to it immediately.

Other campaign commands:

```bash
swoosh-campaign list                       # every campaign, run counts
swoosh-campaign show lego-pick-place       # the runs in one campaign
```

Then collect into it:

```bash
swoosh-collect --campaign lego-pick-place
```

---

## Quick start on a fresh machine

```bash
git clone <this repo> && cd xarm7-data-collection
docker compose build                                  # ~1 min, no ROS

# 1. name the cameras once (pins each label to its USB port, survives reboots)
docker compose run --rm cameras

# 2. check the arm, and verify the 45-degree mount maths against the real robot
SANITY_ARGS="--home --verify-frame" docker compose run --rm sanity

# 3. make a campaign and collect
docker compose run --rm shell -lc 'swoosh-campaign new "my-campaign"'
CAMPAIGN=my-campaign docker compose run --rm collect
```

A Chrome tab opens on the dashboard automatically.

Without Docker, `uv sync` then use the `swoosh-*` commands directly.

---

## Controls

| input | does |
|---|---|
| **left stick** | move the end effector in the plane **parallel to the ground** |
| **right stick — up/down** | raise / lower that plane |
| **right stick — left/right** | rotate the end effector about the **vertical** axis |
| **right trigger** | gripper, **proportional** |
| **A** | start an episode |
| **B** | stop an episode |
| **Y** | re-home (behind the motion countdown) |
| **Start** | quit cleanly; the arm holds position |

The sticks **modify the desired pose** — let go and the target stays where it is,
it does not spring back. There is **no deadman**; instead the target is clamped to a
workspace box (`control.workspace_box_mm`) so a stuck stick cannot drive the arm into
the table.

---

## Dashboard

Opens at `http://127.0.0.1:8770`. Split in half:

- **Left** — campaign name, `+ New campaign`, `Open campaign folder`, and every run
  listed with its timestamp, duration and status. Each run has its own
  `folder` button.
- **Right** — live controller rendering (both sticks and the trigger), live proprio
  (all 7 joint angles and the end-effector position), and the four camera feeds.

**Run lifecycle.** Press **A** and the run appears immediately tagged `IN PROGRESS`.
Press **B** and it becomes `PROCESSING` while the summary video and plots are built on
a worker thread. When that finishes the tag disappears and the run becomes playable.
**Clicking a run before it is ready makes it wiggle** — that is the refusal, not a bug.

**Playback.** Click a finished run and the right-hand side switches from live to
replay: the controller rendering, the proprio readouts and all four camera feeds replay
together on one clock, scrubbable, with `Back to live` to return. Playback happens
entirely in the browser, so it cannot disturb a collection in progress.

---

## What gets recorded

Every stream is logged at its own native rate on **one monotonic clock**, and kept in
its own file. Nothing is merged at write time.

| file | what it is |
|---|---|
| `raw/controller.jsonl` | **Xbox inputs** — the world-model action signal |
| `raw/commanded.jsonl` | the target pose we decided on, in **world** coordinates |
| `raw/xarm_command.jsonl` | the literal arguments handed to `set_servo_cartesian` |
| `raw/arm_state.jsonl` | joints, measured pose (base **and** world), gripper, errors |
| `video/<cam>.mp4` + `_frame_times.json` | each camera with per-frame timestamps |

`commanded` and `xarm_command` look redundant until the frame maths is wrong: the first
is intent in world coordinates, the second is what actually went out after the
45-degree rotation and the workspace clamp. When they disagree, the difference tells
you where the bug is.

### Export

```bash
swoosh-export --campaign my-campaign        # -> campaigns/my-campaign/lerobot/
```

LeRobot v2.1 at 30 Hz. `action` is the **Xbox controller input**; the commanded pose
and the raw SDK arguments ship alongside as `action.commanded_pose_world` and
`action.xarm_servo_cartesian_base`, so a consumer picks deliberately. Resampling is
nearest-in-time — never interpolated, because interpolating invents a command that was
never issued — and each row carries `sync_error_s` recording how far the nearest sample
actually was.

The raw logs stay authoritative. If a framing decision turns out to be wrong it can be
re-derived without re-collecting.

---

## The 45-degree mount

The right arm is mounted 45° clockwise from vertical, so the xArm's own base frame is
**not** aligned with the room and "parallel to the ground" is not "constant base z".
`frames.py` holds the base↔world rotation, derived from manipulation-mono's fitted
`position_axes_right`.

It is a well-formed rotation whether or not its signs are right, so unit tests cannot
prove it matches the physical robot. **Verify it once against the real arm:**

```bash
SANITY_ARGS=--verify-frame docker compose run --rm sanity
```

That jogs 20 mm along each world axis and reports what actually moved. Expect forward,
left, up. If not, fix `R_WORLD_FROM_BASE` before collecting — every action label in
every dataset depends on it.

---

## Cameras

Four Arducams, labelled by **USB port path** rather than `/dev/videoN`, because
enumeration order changes across reboots and a silently swapped camera corrupts a
dataset in a way that is very hard to spot later.

```bash
docker compose run --rm cameras          # shows a frame from each, you name it once
```

Assignments are saved into `src/swoosh_collect/conf/collect.yaml` under
`cameras.by_usb_path`. If a camera is later missing, collection **refuses to start**
rather than guessing.

---

## Safety

- `collision_sensitivity` defaults to **0 (off)**, matching manipulation-mono: the
  controller's payload model doesn't know the gripper's mass, so even sensitivity 1
  spuriously trips error 31. **Real collisions are not caught — clear the workspace.**
  Set the true TCP load in xArm Studio and you can raise it back to 3.
- Every motion command outside the control loop is preceded by the
  `SWOOSH IS ABOUT TO MOVE` countdown.
- On a latched controller error the loop stops commanding motion and says so; clear it
  with `swoosh-sanity`.

## Layout

```
src/swoosh_collect/
  collect.py     teleop + record (the main loop)
  arm.py         xArm connect / home / stream / read
  frames.py      the 45-degree mount maths, and how to verify it
  xbox.py        evdev pad reader
  cameras.py     USB-path-pinned discovery + timestamped capture
  recorder.py    the four raw streams
  server.py      localhost dashboard (stdlib only)
  campaign.py    campaigns and runs on disk
  summarize.py   summary mp4, plots, campaign index
  export_lerobot.py
  sanity.py
  conf/collect.yaml    every knob, one file
design/checklist.md    what's built and what isn't
```
