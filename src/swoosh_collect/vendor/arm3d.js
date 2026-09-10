// 3D view of Swoosh's right arm: real xArm7 meshes, 45-degree mount, animated gripper,
// the safety workspace as a wireframe box, and the desired pose as a sphere + arrow.
import * as THREE from './three.module.js';
import { STLLoader } from './STLLoader.js';
import { OrbitControls } from './OrbitControls.js';
import { buildGripper } from './gripper.js';
import { fk, applyTo } from './armfk.js';

const MESHES = './vendor/meshes/';
const LINKS = ['link_base', 'link1', 'link2', 'link3', 'link4', 'link5', 'link6'];

export async function createArmView(container, opts = {}) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf6f7f9);

  const cam = new THREE.PerspectiveCamera(42, 1, 0.05, 40);
  // Zoomed out enough to see the whole arm and its workspace box.
  cam.position.set(1.9, -1.7, 1.35);
  cam.up.set(0, 0, 1);                       // the model is Z-up; three.js is Y-up

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  container.appendChild(renderer.domElement);

  const controls = new OrbitControls(cam, renderer.domElement);
  controls.target.set(0.35, 0, 0.45);
  controls.enableDamping = true;

  scene.add(new THREE.HemisphereLight(0xffffff, 0x8899aa, 2.0));
  const key = new THREE.DirectionalLight(0xffffff, 1.5);
  key.position.set(2, -2, 3); scene.add(key);

  // Ground plane + grid at world z = 0, the plane the operator thinks in.
  const grid = new THREE.GridHelper(3, 30, 0xc2c8cf, 0xe2e6ea);
  grid.rotation.x = Math.PI / 2; scene.add(grid);

  // world axes: X forward (red), Y left (green), Z up (blue)
  scene.add(new THREE.AxesHelper(0.28));

  // ---- the arm, mounted 45 degrees ----------------------------------------
  // mountRoot carries the physical tilt; everything inside is the arm's own base
  // frame, exactly as the controller sees it.
  const mountRoot = new THREE.Group();
  const tilt = opts.mountTiltDeg === undefined ? 45 : opts.mountTiltDeg;
  mountRoot.rotation.x = tilt * Math.PI / 180;   // tilt about world +X (forward)
  scene.add(mountRoot);

  const loader = new STLLoader();
  const load = (f) => new Promise((res, rej) => loader.load(MESHES + f, res, undefined, rej));
  const linkGroups = [];
  for (let i = 0; i < LINKS.length; i++) {
    const g = new THREE.Group();
    try {
      const geom = await load(LINKS[i] + '.stl');
      geom.computeVertexNormals();
      g.add(new THREE.Mesh(geom, new THREE.MeshStandardMaterial({
        color: i === 0 ? 0x9aa3ab : 0xe8ebee, metalness: 0.15, roughness: 0.6,
        // Half-opaque so the workspace box, the target marker and the far side of the
        // arm stay readable through it. depthWrite off stops the arm's own faces
        // z-fighting into a muddle when they overlap.
        transparent: true, opacity: 0.5, depthWrite: false,
      })));
    } catch (e) { /* a missing mesh should not kill the view */ }
    mountRoot.add(g); linkGroups.push(g);
  }

  // link7 / flange group carries the WRIST MESH and the gripper.
  // manifest.json maps link7 -> end_tool.stl, and LINKS above stops at link6, so this
  // mesh was never loaded: the arm visibly ended at link 6 and the gripper floated in
  // the gap where the wrist belongs, looking detached. It shares the flange's frame
  // (gripper_manifest's root attaches to link7 at zero offset), so it goes in here.
  const flange = new THREE.Group(); mountRoot.add(flange);
  try {
    const geom = await load('end_tool.stl');
    geom.computeVertexNormals();
    flange.add(new THREE.Mesh(geom, new THREE.MeshStandardMaterial({
      color: 0xe8ebee, metalness: 0.15, roughness: 0.6,
      transparent: true, opacity: 0.5, depthWrite: false,
    })));
  } catch (e) { console.error('3D link7 (end_tool.stl) failed to load:', e); }
  let gripper = null;
  try {
    const manifest = await (await fetch(MESHES + 'gripper_manifest.json')).json();
    gripper = await buildGripper(THREE, loader, MESHES, manifest);
    flange.add(gripper.root);
  } catch (e) {
    // Was a silent swallow, which is how a missing base_link.stl went unnoticed:
    // the gripper just never appeared, with nothing said anywhere.
    console.error('3D gripper failed to build:', e);
  }

  // ---- safety workspace, as a wireframe box -------------------------------
  let boxHelper = null;
  let framedOnce = false;
  function setWorkspace(box) {
    if (boxHelper) { scene.remove(boxHelper); boxHelper.geometry.dispose(); }
    if (!box) return;
    const lo = new THREE.Vector3(box.x[0] / 1000, box.y[0] / 1000, box.z[0] / 1000);
    const hi = new THREE.Vector3(box.x[1] / 1000, box.y[1] / 1000, box.z[1] / 1000);
    // Orbit about the box's centre IN PLAN, dropped to z = zmin -- so the view
    // rotates around the middle of the work surface, on the surface, rather than
    // around a point floating in mid air. Recomputed on every box change, which is
    // why the workspace fields feel like they move the camera with them.
    const mid = new THREE.Vector3(
      (lo.x + hi.x) / 2,
      (lo.y + hi.y) / 2,
      lo.z,                       // the ground plane, not the box's vertical middle
    );
    if (framedOnce) {
      // Keep whatever angle and zoom the operator has set; just move the pivot.
      const offset = cam.position.clone().sub(controls.target);
      controls.target.copy(mid);
      cam.position.copy(mid).add(offset);
    } else {
      // First box: look at the work area from 30 degrees above the horizontal,
      // keeping the established front-right azimuth. Elevation is set explicitly
      // rather than inherited from the initial camera position, which sat at 21 deg.
      // Look DOWN the world +X axis from 45 degrees above it. The camera therefore
      // sits behind the origin in -X and above, so the view direction is +X.
      // With cam.up = world +Z, three.js's basis puts screen-right at world -Y, which
      // is what puts world +Y on the LEFT of the picture.
      const ELEV = 45 * Math.PI / 180;
      const dir = new THREE.Vector3(-Math.cos(ELEV), 0, Math.sin(ELEV));
      // Far enough back to hold the whole box AND the arm's own base (the origin),
      // which sits outside the box: take the bounding radius about the pivot and
      // divide by tan(half-fov). Fixed multiples of the box diagonal framed too tight
      // whenever the box did not contain the base.
      const pts = [
        new THREE.Vector3(lo.x, lo.y, lo.z), new THREE.Vector3(hi.x, hi.y, hi.z),
        new THREE.Vector3(lo.x, hi.y, lo.z), new THREE.Vector3(hi.x, lo.y, hi.z),
        new THREE.Vector3(lo.x, lo.y, hi.z), new THREE.Vector3(hi.x, hi.y, lo.z),
        new THREE.Vector3(0, 0, 0),                        // the arm base
      ];
      let radius = 0;
      for (const p of pts) radius = Math.max(radius, p.distanceTo(mid));
      const halfFov = (cam.fov * Math.PI / 180) / 2;
      const dist = Math.max(radius / Math.tan(halfFov) * 1.25, 0.8);
      controls.target.copy(mid);
      cam.position.copy(mid).add(dir.multiplyScalar(dist));
      framedOnce = true;
    }
    controls.update();
    const b3 = new THREE.Box3(lo, hi);
    boxHelper = new THREE.Box3Helper(b3, new THREE.Color(0xd92d20));
    boxHelper.material.transparent = true; boxHelper.material.opacity = 0.75;
    scene.add(boxHelper);
  }

  // ---- desired pose: a sphere with an arrow (it is a pose, not a point) ----
  const target = new THREE.Group();
  const ball = new THREE.Mesh(
    new THREE.SphereGeometry(0.022, 20, 16),
    new THREE.MeshStandardMaterial({ color: 0x1668dc, transparent: true, opacity: 0.85 }));
  target.add(ball);
  const arrow = new THREE.ArrowHelper(
    new THREE.Vector3(0, 0, -1), new THREE.Vector3(0, 0, 0), 0.12, 0x1668dc, 0.045, 0.028);
  target.add(arrow);
  scene.add(target);

  // The tool's approach axis in WORLD space, refreshed every setJoints. The target
  // arrow used to be hardcoded to point mostly straight down, which was right only
  // while the gripper hung vertically -- with home now 45 degrees below horizontal
  // pointing +Y, the marker pointed out of the side of the wrist. Read the real axis
  // off the flange instead of assuming one.
  const approachWorld = new THREE.Vector3(0, 0, -1);

  function setTarget(worldXyzMm, _yawWorldDeg) {
    if (!worldXyzMm || worldXyzMm.length !== 3) { target.visible = false; return; }
    target.visible = true;
    target.position.set(worldXyzMm[0] / 1000, worldXyzMm[1] / 1000, worldXyzMm[2] / 1000);
    // Point along the gripper. yaw is already baked into the flange's orientation,
    // so it needs no separate term -- which is why yawWorldDeg is now unused here.
    arrow.setDirection(approachWorld);
  }

  function setJoints(jointsDeg, gripperNorm) {
    if (!jointsDeg || jointsDeg.length < 7) return;
    const T = fk(jointsDeg);
    for (let i = 0; i < linkGroups.length; i++) applyTo(linkGroups[i], T[i], THREE);
    applyTo(flange, T[7], THREE);
    // Tool +Z in world coordinates = third column of the flange's world matrix.
    // mountRoot carries the 45-degree physical tilt, so going through the world
    // matrix picks that up for free.
    flange.updateMatrixWorld(true);
    const m = flange.matrixWorld.elements;      // column-major
    approachWorld.set(m[8], m[9], m[10]).normalize();
    if (gripper) gripper.set(gripperNorm === undefined ? 1 : gripperNorm);
  }

  function resize() {
    const w = container.clientWidth || 320, h = container.clientHeight || 240;
    renderer.setSize(w, h, false);
    cam.aspect = w / h; cam.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(container); resize();

  (function loop() {
    requestAnimationFrame(loop);
    controls.update();
    renderer.render(scene, cam);
  })();

  setJoints([0, 0, 0, 0, 0, 0, 0], 1);
  return { setJoints, setTarget, setWorkspace, scene, cam };
}
