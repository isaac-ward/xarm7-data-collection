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
      })));
    } catch (e) { /* a missing mesh should not kill the view */ }
    mountRoot.add(g); linkGroups.push(g);
  }

  // link7 / flange group carries the gripper
  const flange = new THREE.Group(); mountRoot.add(flange);
  let gripper = null;
  try {
    const manifest = await (await fetch(MESHES + 'gripper_manifest.json')).json();
    gripper = await buildGripper(THREE, loader, MESHES, manifest);
    flange.add(gripper.root);
  } catch (e) { /* gripper optional */ }

  // ---- safety workspace, as a wireframe box -------------------------------
  let boxHelper = null;
  function setWorkspace(box) {
    if (boxHelper) { scene.remove(boxHelper); boxHelper.geometry.dispose(); }
    if (!box) return;
    const lo = new THREE.Vector3(box.x[0] / 1000, box.y[0] / 1000, box.z[0] / 1000);
    const hi = new THREE.Vector3(box.x[1] / 1000, box.y[1] / 1000, box.z[1] / 1000);
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

  function setTarget(worldXyzMm, yawWorldDeg) {
    if (!worldXyzMm || worldXyzMm.length !== 3) { target.visible = false; return; }
    target.visible = true;
    target.position.set(worldXyzMm[0] / 1000, worldXyzMm[1] / 1000, worldXyzMm[2] / 1000);
    // Point the arrow down (the tool's approach direction), spun by the commanded yaw.
    const yaw = (yawWorldDeg || 0) * Math.PI / 180;
    arrow.setDirection(new THREE.Vector3(Math.sin(yaw) * 0.35, -Math.cos(yaw) * 0.35, -1).normalize());
  }

  function setJoints(jointsDeg, gripperNorm) {
    if (!jointsDeg || jointsDeg.length < 7) return;
    const T = fk(jointsDeg);
    for (let i = 0; i < linkGroups.length; i++) applyTo(linkGroups[i], T[i], THREE);
    applyTo(flange, T[7], THREE);
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
