# Vendored assets

Copied, not imported — this repo has no cross-repo dependencies.

| file | source | why |
|---|---|---|
| `split.min.js` | Split.js v1.6.5 (MIT) | resizable panes |
| `nouislider.min.js/.css` | noUiSlider v15 (MIT) | draggable playback scrubber |
| `three.module.js`, `STLLoader.js`, `OrbitControls.js` | three.js (MIT) | 3D arm view |
| `gripper.js`, `meshes/*.stl`, `meshes/*manifest*.json` | [michelleho-24/PO-Assembly @ lowlevel2](https://github.com/michelleho-24/PO-Assembly/tree/lowlevel2), `pi05/deploy/static` | xArm7 link + UFACTORY gripper geometry, originally from MuJoCo Menagerie `ufactory_xarm7` (Apache-2.0) |

The meshes are in metres, each already expressed in its own MJCF body frame, so each
loads with an IDENTITY local transform and only its group pose is set from FK. The
manifest warns that applying MuJoCo's compiled `geom_pos/geom_quat` double-transforms
and scrambles the arm — we do not.

The manifest carries no kinematic chain (the reference drove body poses straight from
MuJoCo FK), so `armfk.js` implements xArm7 modified-DH forward kinematics instead. That
table is VERIFIED against the real robot by the dashboard's "FK check" sanity button,
which compares FK(joint angles) with the TCP pose the controller reports.
