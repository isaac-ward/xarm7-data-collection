// xArm7 forward kinematics, modified Denavit-Hartenberg.
//
// The vendored mesh manifest carries NO kinematic chain -- the reference project drove
// body poses straight out of MuJoCo -- so the chain lives here instead.
//
// Modified DH, per link i:  T_i = Rx(alpha_{i-1}) * Tx(a_{i-1}) * Rz(theta_i) * Tz(d_i)
// Lengths in metres, angles in radians. This is UFACTORY's published xArm7 table.
//
// DO NOT TRUST THIS UNTIL IT IS CHECKED ON THE ROBOT. The dashboard's "FK check"
// sanity button compares FK(joint angles) against the TCP pose the controller reports;
// if they disagree, this table is wrong and the 3D view is decorative fiction.
export const DH = [
  //  alpha_{i-1},        a_{i-1},  d_i
  [ 0.0,               0.0,      0.267 ],   // joint 1
  [-Math.PI / 2,       0.0,      0.0   ],   // joint 2
  [ Math.PI / 2,       0.0,      0.293 ],   // joint 3
  [ Math.PI / 2,       0.0525,   0.0   ],   // joint 4
  [ Math.PI / 2,       0.0775,   0.3425],   // joint 5
  [-Math.PI / 2,       0.0,      0.0   ],   // joint 6
  [ Math.PI / 2,       0.076,    0.097 ],   // joint 7 -> flange
];

function mul(A, B) {                    // 4x4 row-major multiply
  const C = new Array(16).fill(0);
  for (let r = 0; r < 4; r++)
    for (let c = 0; c < 4; c++)
      for (let k = 0; k < 4; k++) C[r * 4 + c] += A[r * 4 + k] * B[k * 4 + c];
  return C;
}

function linkT(alpha, a, d, theta) {
  const ca = Math.cos(alpha), sa = Math.sin(alpha);
  const ct = Math.cos(theta), st = Math.sin(theta);
  // Rx(alpha) * Tx(a) * Rz(theta) * Tz(d), collapsed
  return [
     ct,      -st,     0,    a,
     st * ca,  ct * ca, -sa, -d * sa,
     st * sa,  ct * sa,  ca,  d * ca,
     0,        0,        0,   1,
  ];
}

/** Joint angles in DEGREES -> array of 8 cumulative 4x4 transforms (base, then each link). */
export function fk(jointsDeg) {
  const out = [[1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]];
  let T = out[0];
  for (let i = 0; i < 7; i++) {
    const [alpha, a, d] = DH[i];
    T = mul(T, linkT(alpha, a, d, (jointsDeg[i] || 0) * Math.PI / 180));
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
