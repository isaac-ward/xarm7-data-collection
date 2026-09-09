// xArm7 forward kinematics, taken from UFACTORY's own MJCF rather than a DH table.
//
// WHY THIS IS NOT A DH TABLE ANY MORE. The previous version used a hand-entered
// modified-DH table "from UFACTORY's published values", carrying its own warning not to
// trust it. It was wrong: joints 6 and 7 had their link twists swapped (-90/+90 where
// the model says +90/-90), which left the wrist mis-rotated and the whole arm looking
// visibly wrong in the 3D view.
//
// The numbers below are read straight out of mujoco_menagerie `ufactory_xarm7`
// (xarm7_nohand.xml) -- the same model the vendored STL meshes and gripper manifest come
// from. PO-Assembly-LEGO/pi05/viz/fk.py runs that model through MuJoCo and validates it
// against recorded controller TCP, so this chain is the manufacturer's geometry, not an
// approximation of it.
//
// MJCF convention, per body: offset by `pos`, apply the body's fixed `quat`, then rotate
// about the joint axis (MuJoCo's default axis is +Z) by that joint's angle.
//
//   T_i = T_{i-1} * Translate(pos_i) * Rx(twist_i) * Rz(theta_i)
//
// Every fixed quat in this model is (w, x, y, z) = (1, +/-1, 0, 0), i.e. exactly +/-90
// degrees about X, so `twist` below is that angle in degrees.
//
// The MJCF stands link_base on a 0.12 m pedestal. The controller's base frame is the
// mounting flange, so that offset is deliberately NOT included -- FK here is in the same
// frame `get_position` reports.
//
// Lengths in metres.
export const CHAIN = [
  //  pos (m)                    twist about X (deg)   joint
  { pos: [0.0,    0.0,     0.267], twist:   0.0 },   // link1
  { pos: [0.0,    0.0,     0.0  ], twist: -90.0 },   // link2
  { pos: [0.0,   -0.293,   0.0  ], twist: +90.0 },   // link3
  { pos: [0.0525, 0.0,     0.0  ], twist: +90.0 },   // link4
  { pos: [0.0775, -0.3425, 0.0  ], twist: +90.0 },   // link5
  { pos: [0.0,    0.0,     0.0  ], twist: +90.0 },   // link6
  { pos: [0.076,  0.097,   0.0  ], twist: -90.0 },   // link7 / flange
];

function mul(A, B) {                    // 4x4 row-major multiply
  const C = new Array(16).fill(0);
  for (let r = 0; r < 4; r++)
    for (let c = 0; c < 4; c++)
      for (let k = 0; k < 4; k++) C[r * 4 + c] += A[r * 4 + k] * B[k * 4 + c];
  return C;
}

/** Translate(pos) * Rx(twistDeg) * Rz(thetaDeg), collapsed into one 4x4. */
function linkT(pos, twistDeg, thetaDeg) {
  const ca = Math.cos(twistDeg * Math.PI / 180), sa = Math.sin(twistDeg * Math.PI / 180);
  const ct = Math.cos(thetaDeg * Math.PI / 180), st = Math.sin(thetaDeg * Math.PI / 180);
  // Rx(a) * Rz(t):
  //   [  ct      -st      0  ]
  //   [  ca*st    ca*ct  -sa ]
  //   [  sa*st    sa*ct   ca ]
  return [
    ct,       -st,       0,    pos[0],
    ca * st,   ca * ct, -sa,   pos[1],
    sa * st,   sa * ct,  ca,   pos[2],
    0,         0,        0,    1,
  ];
}

/** Joint angles in DEGREES -> 8 cumulative 4x4 transforms: base, then link1..link7. */
export function fk(jointsDeg) {
  const out = [[1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]];
  let T = out[0];
  for (let i = 0; i < CHAIN.length; i++) {
    const { pos, twist } = CHAIN[i];
    T = mul(T, linkT(pos, twist, jointsDeg[i] || 0));
    out.push(T);
  }
  return out;
}

/** Apply a cumulative transform to a three.js Object3D. */
export function applyTo(obj, T, THREE) {
  const m = new THREE.Matrix4().set(
    T[0], T[1], T[2],  T[3],
    T[4], T[5], T[6],  T[7],
    T[8], T[9], T[10], T[11],
    T[12], T[13], T[14], T[15],
  );
  obj.matrixAutoUpdate = false;
  obj.matrix.copy(m);
}
