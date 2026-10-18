// JevLight 3D — SUMO city visualization (three.js).
// Talks to scripts/serve_3d.py / run_jevlight.py --visualize:
// GET /api/network once, then SSE frames.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { buildCity } from "/js/city.js";
import { buildSignals } from "/js/signals.js";
import { Fleet } from "/js/vehicles.js";
import { TrafficCam } from "/js/traffic_cam.js";

const state = {
  net: null,
  playing: true,
  frames: 0,
  lastFrameAt: 0,
  stepsPerSecond: 0,
  camMode: false,
};
const captureMode = new URLSearchParams(window.location.search).get("capture") === "1";

// Surface any runtime error on the page itself — a hidden failure is a
// blue sky and zero readouts, with no clue about what broke.
function showBanner(text) {
  const banner = document.getElementById("banner");
  banner.style.display = "";
  banner.textContent = text;
}
window.addEventListener("error", (event) => showBanner(`error: ${event.message}`));
window.addEventListener("unhandledrejection", (event) =>
  showBanner(`error: ${event.reason}`),
);

// --------------------------------------------------------------------- //
// Renderer, scene, lighting
// --------------------------------------------------------------------- //

const canvas = document.getElementById("scene");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
renderer.autoClear = false; // main view + camera sub-views per frame

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x9db8d8);
scene.fog = new THREE.Fog(0x9db8d8, 1200, 5200);

const camera = new THREE.PerspectiveCamera(
  55,
  window.innerWidth / window.innerHeight,
  1,
  30000,
);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.maxPolarAngle = Math.PI / 2.05; // keep the camera above ground
controls.minDistance = 15; // street level: cars stay visible when zoomed in
controls.maxDistance = 12000;

renderer.setClearColor(0x9db8d8); // any manual clear matches the sky

scene.add(new THREE.HemisphereLight(0xcfe4ff, 0x4a4438, 0.75));
const sun = new THREE.DirectionalLight(0xfff1dd, 1.6);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.bias = -0.0004;
scene.add(sun);

window.addEventListener("resize", onResize);

function onResize() {
  const width = window.innerWidth;
  const height = window.innerHeight;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height);
}

// --------------------------------------------------------------------- //
// City, signals, fleet
// --------------------------------------------------------------------- //

let fleet = null;
let signals = null;
let trafficCam = null;
let homeCamera = null;

function fitLightsAndCamera(bounds, tlsList) {
  const { min, max } = bounds;
  const cx = (min[0] + max[0]) / 2;
  const cz = (min[1] + max[1]) / 2;
  const span = Math.max(max[0] - min[0], max[1] - min[1]);
  const distance = span * 1.2;

  // Late-afternoon sun for long, readable shadows.
  sun.position.set(cx + distance * 0.55, distance * 0.8, cz + distance * 0.4);
  sun.target.position.set(cx, 0, cz);
  scene.add(sun.target);
  const shadow = sun.shadow.camera;
  shadow.left = -span * 0.75;
  shadow.right = span * 0.75;
  shadow.top = span * 0.75;
  shadow.bottom = -span * 0.75;
  shadow.near = 1;
  shadow.far = distance * 2.4;
  shadow.updateProjectionMatrix();

  // Start close above the most central junction, not in a far overview.
  let anchor = { x: cx, y: cz };
  let bestDistance = Infinity;
  for (const tls of tlsList || []) {
    const d = Math.hypot(tls.x - cx, tls.y - cz);
    if (d < bestDistance) {
      bestDistance = d;
      anchor = tls;
    }
  }
  camera.position.set(anchor.x + span * 0.055, span * 0.052, anchor.y + span * 0.1);
  controls.target.set(anchor.x, 0, anchor.y);
  controls.update();
  homeCamera = { position: camera.position.clone(), target: controls.target.clone() };
}

function buildNetwork(net) {
  scene.add(buildCity(net));
  fleet = new Fleet(scene);
  signals = buildSignals(scene, net);
  trafficCam = new TrafficCam(scene, net);
  buildTlsChips(net.tls);
  buildCamList(net.tls);
  fitLightsAndCamera(net.bounds, net.tls);
  onResize();
  document.getElementById("banner").style.display = "none";
}

// --------------------------------------------------------------------- //
// Traffic lights
// --------------------------------------------------------------------- //

function tlsColor(state) {
  let r = 0, y = 0, g = 0;
  for (const ch of state) {
    if (ch === "r" || ch === "o") r++;
    else if (ch === "y") y++;
    else if (ch === "G" || ch === "g") g++;
  }
  if (r >= y && r >= g) return "r";
  if (y >= g) return "y";
  return "g";
}

function updateTrafficLights(states) {
  signals.update(states);
  for (const [id, signalState] of Object.entries(states)) {
    const chip = document.getElementById(`tls-${cssId(id)}`);
    if (chip) chip.className = `tls ${tlsColor(signalState)}`;
  }
}

function cssId(id) {
  return id.replace(/[^a-zA-Z0-9_-]/g, "_");
}

function buildTlsChips(tlsList) {
  const bar = document.getElementById("tlsbar");
  for (const tls of tlsList) {
    const chip = document.createElement("div");
    chip.id = `tls-${cssId(tls.id)}`;
    chip.className = "tls";
    chip.innerHTML = `<span class="dot"></span>${tls.id}`;
    bar.appendChild(chip);
  }
}

// --------------------------------------------------------------------- //
// Traffic camera mode
// --------------------------------------------------------------------- //

const camPanels = [0, 1, 2, 3].map((i) =>
  document.getElementById(`cam${i}`),
);

function setCamMode(enabled) {
  state.camMode = enabled;
  document.getElementById("camview").classList.toggle("on", enabled);
  document.getElementById("cam").classList.toggle("active", enabled);
  document.getElementById("reroll").style.display = enabled ? "" : "none";
  document.getElementById("camoff").style.display = enabled ? "" : "none";
  if (!enabled) trafficCam.hide();
}

function selectJunction(junction) {
  trafficCam.setJunction(junction.id);
  document.getElementById("camtitle").textContent = `📹 ${junction.id}`;
  for (const item of document.querySelectorAll(".camitem")) {
    item.classList.toggle("selected", item.dataset.id === junction.id);
  }
  // Label each live panel with its approach direction.
  camPanels.forEach((panel, index) => {
    const view = trafficCam.cameras[index];
    panel.querySelector(".tag").textContent = view ? view.label : "—";
    panel.style.visibility = view ? "visible" : "hidden";
  });
}

function buildCamList(tlsList) {
  const list = document.getElementById("camlist");
  for (const tls of tlsList) {
    const item = document.createElement("div");
    item.className = "camitem";
    item.dataset.id = tls.id;
    item.textContent = tls.id;
    item.addEventListener("click", () => {
      setCamMode(true);
      selectJunction(tls);
    });
    list.appendChild(item);
  }
}

document.getElementById("cam").addEventListener("click", () => {
  const enable = !state.camMode;
  setCamMode(enable);
  if (enable) selectJunction(trafficCam.randomJunction());
});
document.getElementById("reroll").addEventListener("click", () => {
  selectJunction(trafficCam.randomJunction());
});
document.getElementById("camoff").addEventListener("click", () => setCamMode(false));

/** Render the 2x2 live grid: one scissored pass per approach panel.
 *  setViewport/setScissor take CSS pixels — three multiplies by the
 *  device pixel ratio internally. */
function renderCamViews() {
  const width = window.innerWidth;
  const height = window.innerHeight;
  camPanels.forEach((panel, index) => {
    const view = trafficCam.cameras[index];
    if (!view) return;
    const rect = panel.getBoundingClientRect();
    if (rect.bottom < 0 || rect.top > height) return;
    const left = rect.left;
    const bottom = height - rect.bottom;
    renderer.setViewport(left, bottom, rect.width, rect.height);
    renderer.setScissor(left, bottom, rect.width, rect.height);
    renderer.setScissorTest(true);
    renderer.clear();
    view.camera.aspect = rect.width / Math.max(rect.height, 1);
    view.camera.updateProjectionMatrix();
    renderer.render(scene, view.camera);
  });
  renderer.setScissorTest(false);
  renderer.setViewport(0, 0, width, height);
}

// --------------------------------------------------------------------- //
// HUD + commands
// --------------------------------------------------------------------- //

const el = (id) => document.getElementById(id);

function command(payload) {
  fetch("/api/command", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  }).catch((err) => console.error("command failed", err));
}

function setPlaying(playing) {
  state.playing = playing;
  el("play").textContent = playing ? "⏸ Pause" : "▶ Play";
}

el("play").addEventListener("click", () => {
  const next = !state.playing;
  setPlaying(next);
  command({ action: next ? "play" : "pause" });
});
el("step").addEventListener("click", () => {
  setPlaying(false);
  command({ action: "pause" });
  command({ action: "step" });
});
el("speed").addEventListener("change", (event) => {
  command({ action: "speed", speed: parseFloat(event.target.value) });
});
el("resetcam").addEventListener("click", () => {
  if (homeCamera) {
    camera.position.copy(homeCamera.position);
    controls.target.copy(homeCamera.target);
    controls.update();
  }
});

function updateHud(frame) {
  el("time").textContent = frame.t.toFixed(0);
  el("vehicles").textContent = frame.vehicles.length;
  state.frames += 1;
  const now = performance.now();
  if (now - state.lastFrameAt > 1000) {
    state.stepsPerSecond = (state.frames * 1000) / (now - state.lastFrameAt);
    state.frames = 0;
    state.lastFrameAt = now;
    el("rate").textContent = state.stepsPerSecond.toFixed(1);
  }
}

// --------------------------------------------------------------------- //
// Streaming
// --------------------------------------------------------------------- //

function connectStream() {
  const source = new EventSource("/api/stream");
  source.addEventListener("init", (event) => {
    const meta = JSON.parse(event.data);
    setPlaying(meta.playing);
    el("speed").value = String(meta.speed);
  });
  source.addEventListener("frame", (event) => {
    const frame = JSON.parse(event.data);
    if (frame.ended) {
      showBanner("simulation ended — restart the server for a fresh run");
      setPlaying(false);
      return;
    }
    fleet.update(frame.vehicles);
    updateTrafficLights(frame.tls);
    updateHud(frame);
  });
  source.onerror = () => showBanner("connection lost — retrying…");
  source.onopen = () => {
    el("banner").style.display = "none";
  };
}

async function boot() {
  const response = await fetch("/api/network");
  if (!response.ok) throw new Error(`/api/network ${response.status}`);
  state.net = await response.json();
  showBanner("building the city…");
  await new Promise((resolve) => setTimeout(resolve)); // let the banner paint
  buildNetwork(state.net);
  if (captureMode) {
    // Synchronous renders: SUMO is paused while this frozen frame is captured.
    window.setJevFrame = (frame) => {
      fleet.update(frame.vehicles);
      signals.update(frame.tls);
    };
    window.captureJevFrame = captureJevFrame;
  } else {
    connectStream();
  }
}

function captureJevFrame(junctionId) {
  if (!trafficCam.setJunction(junctionId)) throw new Error(`Unknown junction ${junctionId}`);
  if (trafficCam.cameras.length !== 4) throw new Error("Image mode needs four approach views");
  renderer.setPixelRatio(1);
  renderer.setSize(1024, 768);
  trafficCam.setAspect(512 / 384);
  trafficCam.hide();
  const result = document.createElement("canvas");
  result.width = 1024;
  result.height = 768;
  const context = result.getContext("2d");
  renderer.setScissorTest(true);
  trafficCam.cameras.forEach((view, i) => {
    const x = (i % 2) * 512;
    const y = Math.floor(i / 2) * 384;
    renderer.setViewport(x, 768 - y - 384, 512, 384);
    renderer.setScissor(x, 768 - y - 384, 512, 384);
    renderer.clear();
    renderer.render(scene, view.camera);
  });
  context.drawImage(canvas, 0, 0);
  trafficCam.cameras.forEach((view, i) => {
    const approach = ["south", "west", "north", "east"][
      Math.round(view.bearing / 90) % 4
    ];
    const travel = ["northbound", "eastbound", "southbound", "westbound"][
      Math.round(view.bearing / 90) % 4
    ];
    const x = (i % 2) * 512;
    const y = Math.floor(i / 2) * 384;
    context.fillStyle = "#10151e";
    context.fillRect(x, y, 512, 32);
    context.fillStyle = "white";
    context.font = "bold 20px sans-serif";
    context.fillText(`${travel} traffic (from ${approach})`, x + 12, y + 23);
  });
  return result.toDataURL("image/jpeg", 0.9);
}

boot().catch((err) => {
  showBanner(`failed to start: ${err}`);
  console.error(err);
});

// Render loop: main city view, then the camera sub-views on top.
(function animate() {
  if (captureMode) return;
  requestAnimationFrame(animate);
  controls.update();
  renderer.setScissorTest(false);
  renderer.clear();
  renderer.render(scene, camera);
  if (state.camMode && trafficCam.cameras.length) renderCamViews();
})();
