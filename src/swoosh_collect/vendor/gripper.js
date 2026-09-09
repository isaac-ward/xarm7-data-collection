// Real UFACTORY xArm gripper sub-tree, built from meshes/gripper_manifest.json
// (equivalently the "gripper" block of meshes/manifest_full.json).
//
// Geometry comes from pi05/out/menagerie/ufactory_xarm7/xarm7.xml. Each STL's vertices are
// ALREADY in its MJCF body frame, so every mesh is added to its group with an identity local
// transform. Do NOT apply MuJoCo's compiled geom_pos/geom_quat -- those merely undo the
// compiler's inertial recentring and exactly cancel mesh_pos/mesh_quat; applying them
// double-transforms and throws the parts 5-37 mm out of place.
//
// The linkage is an exact parallelogram, so all six joints take the SAME scalar theta and the
// per-joint axis sign supplies the direction. theta = 0 is fully OPEN, theta = 0.85 rad is
// fully CLOSED -- the reverse of our data channel, hence the (1 - g).
//
// Usage:
//   const gripper = await buildGripper(THREE, new STLLoader(), '/static/meshes/', manifest);
//   link7Group.add(gripper.root);          // manifest.root.position/quaternion are baked in
//   gripper.set(0.0);                      // 0 = closed, 1 = open

export const THETA_CLOSED = 0.85;   // rad, gripper_norm = 0
export const THETA_OPEN   = 0.0;    // rad, gripper_norm = 1

/** gripper_norm (0 = closed, 1 = open) -> driver-joint angle in radians. */
export const thetaFromNorm = (g) =>
  THETA_CLOSED * (1 - Math.min(Math.max(g, 0), 1));

/** Finger inner-face separation in metres for a given theta (matches MuJoCo to 5e-7 m). */
export const separationFromTheta = (t) =>
  2 * (0.035 + 0.035465 * Math.cos(t) - 0.042039 * Math.sin(t)) - 0.052006;

/**
 * @param THREE     the three.js namespace
 * @param stlLoader an STLLoader instance
 * @param meshDir   URL prefix for the STL files, e.g. '/static/meshes/'
 * @param manifest  the parsed gripper manifest (or manifest_full.json's .gripper)
 */
export async function buildGripper(THREE, stlLoader, meshDir, manifest) {
  const groups = {};
  const moving = [];

  const root = new THREE.Group();
  root.name = 'xarm_gripper';
  root.position.fromArray(manifest.root.position);            // [0, 0, 0] -- flush on link7
  root.quaternion.fromArray(manifest.root.quaternion_xyzw);   // [0, 0, 1, 0] -- 180 deg about z
  groups[manifest.root.attach_to] = root;                     // parent lookups resolve to root

  const load = (file) => new Promise((res, rej) =>
    stlLoader.load(meshDir + file, res, undefined, rej));

  for (const name of manifest.order) {
    const p = manifest.parts[name];

    // The gripper base IS the root group; the rest hang off their named parent.
    const g = (p.parent === manifest.root.attach_to) ? root : new THREE.Group();
    if (g !== root) {
      g.name = name;
      g.position.fromArray(p.position);
      g.quaternion.fromArray(p.quaternion_xyzw);              // rest (theta = 0) orientation
      groups[p.parent].add(g);
    }
    groups[name] = g;

    const geom = await load(p.file);
    geom.computeVertexNormals();
    g.add(new THREE.Mesh(geom, new THREE.MeshStandardMaterial({
      color: new THREE.Color(p.material_hex), metalness: 0.1, roughness: 0.7,
    })));                                                      // identity local transform

    if (p.moves_with_joint) {
      moving.push({
        group: g,
        rest: g.quaternion.clone(),
        axis: new THREE.Vector3().fromArray(p.joint.axis).normalize(),
        spin: new THREE.Quaternion(),
      });
    }
  }

  /** Set the opening from one scalar: 0 = closed, 1 = open. */
  function set(gripperNorm) {
    const theta = thetaFromNorm(gripperNorm);
    for (const j of moving) {
      j.spin.setFromAxisAngle(j.axis, theta);                  // pivot is the group's origin
      j.group.quaternion.copy(j.rest).multiply(j.spin);
    }
    return theta;
  }

  set(1.0);
  return { root, groups, set, thetaFromNorm, separationFromTheta };
}
