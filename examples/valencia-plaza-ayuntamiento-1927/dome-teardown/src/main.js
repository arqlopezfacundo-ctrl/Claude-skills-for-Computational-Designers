import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import { buildSpecs, TYPES, STEPS, R, H, DY, PHI_C, meshVolume, meshArea } from './model.js';

const $ = s => document.querySelector(s);
const isTouch = matchMedia('(pointer:coarse)').matches || innerWidth < 820;
const f1 = v => v.toFixed(1).replace('.', ',');

// ------------------------------------------------------------------ renderer
const canvas = $('#c');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: 'high-performance' });
renderer.setClearAlpha(0);
let quality = isTouch ? 'low' : 'high';
function applyQuality() {
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, quality === 'high' ? 2 : 1.5));
  key.shadow.mapSize.set(quality === 'high' ? 2048 : 1024, quality === 'high' ? 2048 : 1024);
  key.shadow.map?.dispose(); key.shadow.map = null;
  resize(); dirty = true;
}
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;

const scene = new THREE.Scene();
const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
scene.environmentIntensity = 0.85;

const camera = new THREE.PerspectiveCamera(38, 1, 0.5, 400);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.dampingFactor = 0.08;
controls.minDistance = 6; controls.maxDistance = 90;
controls.target.set(0, 5.5, 0);
controls.maxPolarAngle = Math.PI * 0.62;
camera.position.set(26, 20, 30);
controls.autoRotateSpeed = 1.6;
let dirty = true;
controls.addEventListener('change', () => { dirty = true; });

const key = new THREE.DirectionalLight(0xffe3bd, 2.4);
key.position.set(14, 30, 12);
key.castShadow = true;
key.shadow.camera.left = -22; key.shadow.camera.right = 22; key.shadow.camera.top = 24; key.shadow.camera.bottom = -24;
key.shadow.camera.near = 5; key.shadow.camera.far = 90;
key.shadow.bias = -0.0004; key.shadow.normalBias = 0.03; key.shadow.radius = 3;
scene.add(key);
scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x8a7a60, 0.55));

// ------------------------------------------------------------------ textures
function canvasTex(w, h, draw, repeat = [1, 1], srgb = true) {
  const c = document.createElement('canvas'); c.width = w; c.height = h;
  draw(c.getContext('2d'), w, h);
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(...repeat); t.anisotropy = 4;
  if (srgb) t.colorSpace = THREE.SRGBColorSpace;
  return t;
}
let seed = 7; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
function speckle(ctx, w, h, n, rmin, rmax, cols, alpha) {
  for (let i = 0; i < n; i++) {
    ctx.fillStyle = cols[(rnd() * cols.length) | 0]; ctx.globalAlpha = alpha * (0.4 + rnd() * 0.6);
    ctx.beginPath(); ctx.arc(rnd() * w, rnd() * h, rmin + rnd() * (rmax - rmin), 0, 6.283); ctx.fill();
  }
  ctx.globalAlpha = 1;
}
const stoneTex = canvasTex(512, 512, (ctx, w, h) => {
  ctx.fillStyle = '#b3a07e'; ctx.fillRect(0, 0, w, h);
  speckle(ctx, w, h, 160, 20, 70, ['#a48f6c', '#c2b08c', '#96805e', '#b8a582'], 0.22);
  speckle(ctx, w, h, 2600, 0.6, 2.2, ['#6f5f46', '#8a7857', '#d8ccb0', '#4f4331'], 0.5);
  ctx.strokeStyle = 'rgba(120,98,70,0.22)'; ctx.lineWidth = 1;
  for (let i = 0; i < 9; i++) { ctx.beginPath(); let x = rnd() * w, y = rnd() * h; ctx.moveTo(x, y); for (let j = 0; j < 8; j++) { x += (rnd() - 0.3) * 50; y += (rnd() - 0.5) * 24; ctx.lineTo(x, y); } ctx.stroke(); }
}, [0.5, 0.5]);
const stoneBump = canvasTex(512, 512, (ctx, w, h) => {
  ctx.fillStyle = '#808080'; ctx.fillRect(0, 0, w, h);
  speckle(ctx, w, h, 3200, 0.8, 3.2, ['#303030', '#b0b0b0', '#606060', '#d0d0d0'], 0.55);
}, [0.5, 0.5], false);
const ironTex = canvasTex(512, 512, (ctx, w, h) => {
  ctx.fillStyle = '#44564e'; ctx.fillRect(0, 0, w, h);
  speckle(ctx, w, h, 140, 14, 60, ['#3a4a43', '#4f6158', '#40504a'], 0.25);
  speckle(ctx, w, h, 220, 2, 14, ['#7a3e22', '#8c4a28', '#5d2f1a', '#a05a30'], 0.55);     // rust chips
  speckle(ctx, w, h, 3000, 0.5, 1.8, ['#222', '#789', '#111'], 0.35);
}, [3, 1]);
const ironRough = canvasTex(256, 256, (ctx, w, h) => {
  ctx.fillStyle = '#9a9a9a'; ctx.fillRect(0, 0, w, h);
  speckle(ctx, w, h, 900, 2, 14, ['#555', '#ccc', '#777', '#ddd'], 0.5);
}, [3, 1], false);
const noiseBump = canvasTex(256, 256, (ctx, w, h) => {
  ctx.fillStyle = '#808080'; ctx.fillRect(0, 0, w, h);
  speckle(ctx, w, h, 1800, 0.6, 2.4, ['#404040', '#c0c0c0'], 0.5);
}, [3, 1], false);

// arms of Valencia on the medallion
const arms = canvasTex(512, 512, (ctx, w, h) => {
  ctx.fillStyle = '#2f3a36'; ctx.fillRect(0, 0, w, h);
  const g = ctx.createRadialGradient(256, 256, 20, 256, 256, 250);
  g.addColorStop(0, '#e9c75a'); g.addColorStop(0.55, '#d6a93a'); g.addColorStop(1, '#8e6a22');
  ctx.fillStyle = g; ctx.beginPath(); ctx.arc(256, 256, 232, 0, 6.283); ctx.fill();
  ctx.lineWidth = 8; ctx.strokeStyle = '#4a3410'; ctx.stroke();
  // shield with four red bars
  ctx.save(); ctx.translate(256, 262);
  ctx.beginPath(); ctx.moveTo(-96, -110); ctx.lineTo(96, -110); ctx.lineTo(96, 40); ctx.quadraticCurveTo(96, 120, 0, 150); ctx.quadraticCurveTo(-96, 120, -96, 40); ctx.closePath();
  ctx.fillStyle = '#e8c04a'; ctx.fill(); ctx.clip();
  ctx.fillStyle = '#b2262c'; for (let i = 0; i < 4; i++) ctx.fillRect(-96 + 24 + i * 48 - 12, -110, 24, 270);
  ctx.restore();
  ctx.lineWidth = 7; ctx.strokeStyle = '#4a3410'; ctx.beginPath(); ctx.moveTo(-96, 152); ctx.stroke();
  // crown
  ctx.fillStyle = '#c9972d'; ctx.beginPath(); ctx.moveTo(176, 150); ctx.lineTo(176, 118); ctx.lineTo(206, 132); ctx.lineTo(256, 100); ctx.lineTo(306, 132); ctx.lineTo(336, 118); ctx.lineTo(336, 150); ctx.closePath(); ctx.fill();
  ctx.strokeStyle = '#4a3410'; ctx.lineWidth = 5; ctx.stroke();
}, [1, 1]);

// ------------------------------------------------------------------ materials
const M = {
  stone: new THREE.MeshStandardMaterial({ map: stoneTex, bumpMap: stoneBump, bumpScale: 1.4, roughness: 0.92, metalness: 0 }),
  iron: new THREE.MeshStandardMaterial({ map: ironTex, roughnessMap: ironRough, bumpMap: noiseBump, bumpScale: 0.6, roughness: 0.62, metalness: 0.75, side: THREE.DoubleSide }),
  steel: new THREE.MeshStandardMaterial({ color: 0x8e949a, roughness: 0.38, metalness: 0.95 }),
  lead: new THREE.MeshStandardMaterial({ color: 0x6a6d74, roughness: 0.52, metalness: 0.85, side: THREE.DoubleSide, bumpMap: noiseBump, bumpScale: 0.4 }),
  brass: new THREE.MeshStandardMaterial({ color: 0xc09a45, roughness: 0.28, metalness: 1.0 }),
};
const glassMats = new Map();
function glassMat(color) {
  if (!glassMats.has(color)) glassMats.set(color, new THREE.MeshPhysicalMaterial({
    color, transparent: true, opacity: 0.68, roughness: 0.06, metalness: 0, clearcoat: 1, clearcoatRoughness: 0.04,
    envMapIntensity: 1.5, side: THREE.DoubleSide, depthWrite: false }));
  return glassMats.get(color);
}
M.medal = [new THREE.MeshStandardMaterial({ color: 0x4a4a42, roughness: 0.5, metalness: 0.8 }),
  new THREE.MeshPhysicalMaterial({ map: arms, roughness: 0.28, metalness: 0.5, clearcoat: 0.6 }), new THREE.MeshStandardMaterial({ color: 0x3a3a34, roughness: 0.6, metalness: 0.8 })];
const densities = { stone: 2400, iron: 7800, steel: 7850, lead: 11300, glass: 2500, brass: 8500, medal: 4800 };

// ------------------------------------------------------------------ parts
const root = new THREE.Group(); scene.add(root);
const specs = buildSpecs();
const parts = [], pickables = [];
let totalMass = 0;
const maxOrder = 9;
for (const sp of specs) {
  const T = TYPES[sp.type];
  const group = new THREE.Group();
  const meshes = [];
  const bb = new THREE.Box3();
  for (const gg of sp.geos) { gg.g.computeBoundingBox(); bb.union(gg.g.boundingBox); }
  const c = bb.getCenter(new THREE.Vector3());
  let mass = 0;
  for (const gg of sp.geos) {
    const vol = T.shell && gg.mk !== 'lead' ? meshArea(gg.g) * T.shell : (gg.mk === 'lead' && T.mk === 'glass') ? meshArea(gg.g) * 0.0015 : (gg.mk === 'lead' && T.shell) ? meshArea(gg.g) * T.shell : meshVolume(gg.g);
    mass += vol * (densities[gg.mk] ?? T.den);
    gg.g.translate(-c.x, -c.y, -c.z);
    const base = gg.mk === 'glass' ? glassMat(gg.color ?? sp.color ?? 0x888888) : M[gg.mk];
    const mesh = new THREE.Mesh(gg.g, base);
    mesh.userData.base = base;
    mesh.castShadow = gg.mk !== 'glass' && gg.mk !== 'lead' || (gg.mk === 'lead' && T.mk === 'lead');
    mesh.receiveShadow = gg.mk !== 'glass';
    if (gg.mk === 'glass') mesh.renderOrder = 2;
    group.add(mesh); meshes.push(mesh);
  }
  group.position.copy(c);
  root.add(group);
  const part = { sp, T, group, meshes, base: c.clone(), vec: sp.vec.clone(), order: T.order, mass, visible: true };
  const pi = parts.length;
  part.i = pi;
  for (const m of meshes) { m.userData.part = part; pickables.push(m); }
  totalMass += mass;
  // delay for staggered explosion: later order separates first; instances stagger slightly
  part.delay = 0.5 * (maxOrder - T.order) / (maxOrder - 1) + 0.07 * (sp.idx / Math.max(1, sp.total));
  parts.push(part);
}
const byType = {};
for (const p of parts) (byType[p.sp.type] ||= []).push(p);

// ground drawing + shadow catcher
const floorY = -1.64;
const shadowFloor = new THREE.Mesh(new THREE.PlaneGeometry(80, 80), new THREE.ShadowMaterial({ opacity: 0.2 }));
shadowFloor.rotation.x = -Math.PI / 2; shadowFloor.position.y = floorY; shadowFloor.receiveShadow = true; scene.add(shadowFloor);
{
  const mat = new THREE.LineBasicMaterial({ color: 0x9e3426, transparent: true, opacity: 0.55 });
  const pts = (r, n = 128) => { const a = []; for (let i = 0; i <= n; i++) a.push(new THREE.Vector3(r * Math.cos(6.283 * i / n), floorY + 0.01, r * Math.sin(6.283 * i / n))); return a; };
  for (const r of [R, R + 0.7, R + 1.05, R + 4]) scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts(r)), mat));
  const ax = [];
  for (const a of [0, Math.PI / 2]) { ax.push(new THREE.Vector3(-18 * Math.cos(a), floorY + 0.01, -18 * Math.sin(a)), new THREE.Vector3(18 * Math.cos(a), floorY + 0.01, 18 * Math.sin(a))); }
  scene.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(ax), new THREE.LineDashedMaterial({ color: 0x9e3426, dashSize: 0.5, gapSize: 0.35, transparent: true, opacity: 0.5 })).computeLineDistances());
}

// guide lines
const guidePos = new Float32Array(parts.length * 6);
const guideGeo = new THREE.BufferGeometry();
guideGeo.setAttribute('position', new THREE.BufferAttribute(guidePos, 3));
const guides = new THREE.LineSegments(guideGeo, new THREE.LineDashedMaterial({ color: 0x6b5b45, dashSize: 0.22, gapSize: 0.16, transparent: true, opacity: 0.55, depthWrite: false }));
guides.frustumCulled = false; scene.add(guides);

// ------------------------------------------------------------------ state
let U = 0;                 // explode 0..1 (slider)
let S = 9;                 // assembly progress 0..9 (9 = complete)
let sel = null;            // { part | null, type }
let xray = false, labelsOn = false, cutOn = false, spinOn = false, guidesOn = true;
const ease = t => t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
const clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v));
const SPAN = 0.55;
function fitDist(u) {
  const extH = 11.5 + 5.0 * u, extV = 8 + 7 * u;
  const tv = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)), th = tv * camera.aspect;
  return Math.max(extH * 0.85 / th, extV / tv) * 1.04;
}
function frame_(u) { tweenPos = null; tweenTarget = new THREE.Vector3(0, 5.2 + 3.6 * u, 0); tweenDist = fitDist(u); }

function updateParts() {
  let g = 0;
  const gp = guidePos;
  const dist = new THREE.Vector3();
  for (const p of parts) {
    let e = ease(clamp((U * (1 + SPAN) - p.delay) / 1));
    let vis = true;
    if (S < 9) {                                // assembly sequence: each step drops in from its exploded place
      vis = S > p.order - 1;
      e = 1 - ease(clamp(S - (p.order - 1)));
    }
    p.group.visible = vis;
    p.group.position.copy(p.base).addScaledVector(p.vec, e);
    p.e = e;
    const show = guidesOn && e > 0.01 && vis;
    const o = g * 6;
    gp[o] = p.base.x; gp[o + 1] = p.base.y; gp[o + 2] = p.base.z;
    if (show) { gp[o + 3] = p.group.position.x; gp[o + 4] = p.group.position.y; gp[o + 5] = p.group.position.z; }
    else { gp[o + 3] = p.base.x; gp[o + 4] = p.base.y; gp[o + 5] = p.base.z; }
    g++;
  }
  guideGeo.attributes.position.needsUpdate = true;
  guides.computeLineDistances();
  dirty = true;
}

// ------------------------------------------------------------------ materials state (isolate / x-ray)
const ghostCache = new Map();
function variant(base, level) {
  const mk = Array.isArray(base) ? base : [base];
  const out = mk.map(m => {
    const key = m.uuid + ':' + level;
    if (!ghostCache.has(key)) {
      const c = m.clone();
      c.transparent = true; c.depthWrite = false;
      c.opacity = (m.transparent ? m.opacity : 1) * level;
      if (m.map) c.map = m.map;
      ghostCache.set(key, c);
    }
    return ghostCache.get(key);
  });
  return Array.isArray(base) ? out : out[0];
}
const hlCache = new Map();
function highlight(base) {
  const arr = Array.isArray(base) ? base : [base];
  const out = arr.map(m => {
    if (!hlCache.has(m.uuid)) {
      const c = m.clone();
      if (c.emissive) { c.emissive = new THREE.Color(0xc2502a); c.emissiveIntensity = 0.42; }
      if (c.transparent) c.opacity = Math.min(1, m.opacity + 0.2);
      hlCache.set(m.uuid, c);
    }
    return hlCache.get(m.uuid);
  });
  return Array.isArray(base) ? out : out[0];
}
function applyMaterials() {
  for (const p of parts) for (const m of p.meshes) {
    const base = m.userData.base;
    let lvl = 1;
    if (sel) lvl = (sel.part === p) ? 1 : (sel.type === p.sp.type ? 0.38 : 0.15);
    if (xray) lvl = Math.min(lvl, sel && sel.part === p ? 1 : 0.28);
    m.material = sel && sel.part === p ? highlight(base) : lvl === 1 ? base : variant(base, lvl);
  }
  dirty = true;
}

// ------------------------------------------------------------------ section cut
const clipPlane = new THREE.Plane(new THREE.Vector3(-1, 0, 0), 30);
let cutTarget = 30, cutVal = 30;
function setCut(on) { cutOn = on; cutTarget = on ? 0.15 : 30; if (on) { cutVal = Math.max(cutVal, 14); renderer.clippingPlanes = [clipPlane]; } dirty = true; }

// ------------------------------------------------------------------ labels
const labelDefs = [['sillar', 'Sillar', 30, [0, 0, 0]], ['cornisa', 'Cornisa', 6, [0, 0, 0]], ['anillo', 'Anillo de arranque', 0, [R, 0, 0]],
  ['nervio', 'Nervio', 3, [0, 0, 0]], ['intermedio', 'Anillo intermedio', 0, [-R * Math.cos(0.52), 0, 0]], ['vidrio', 'Vidriera', 52, [0, 0, 0]],
  ['clave', 'Anillo de clave', 0, [-R * Math.cos(PHI_C), 0, 0]], ['medallon', 'Medallón', 0, [1.45, 0, 0]],
  ['lcup', 'Cupulín', 0, [0, 0, 0]], ['remate', 'Remate', 0, [0, 0, 0]]];
const labelsEl = $('#labels');
const labels = labelDefs.map(([type, text, idx, off]) => {
  const el = document.createElement('div'); el.className = 'lbl'; el.textContent = text; labelsEl.appendChild(el);
  return { el, off: new THREE.Vector3(...off), part: byType[type][Math.min(idx, byType[type].length - 1)] };
});
const v3 = new THREE.Vector3();
function updateLabels() {
  const W = canvas.clientWidth, Hh = canvas.clientHeight;
  for (const l of labels) {
    if (!labelsOn || !l.part.group.visible) { l.el.style.opacity = 0; continue; }
    l.part.group.getWorldPosition(v3).add(l.off).project(camera);
    const x = (v3.x * 0.5 + 0.5) * W, y = (-v3.y * 0.5 + 0.5) * Hh;
    const ok = v3.z < 1 && x > 8 && x < W - 8 && y > 60 && y < Hh - 150;
    l.el.style.opacity = ok ? 1 : 0;
    l.el.style.transform = `translate(${x.toFixed(1)}px,${y.toFixed(1)}px)`;
  }
}

// ------------------------------------------------------------------ UI: data panel & lists
const bbAll = new THREE.Box3().setFromObject(root);
const totalHeight = bbAll.max.y - bbAll.min.y;
const nParts = parts.length;
const nBolts = (byType.perno || []).length;
const nGlass = (byType.vidrio || []).length;
const matCount = {};
for (const p of parts) matCount[p.T.mat] = (matCount[p.T.mat] || 0) + 1;
$('#dataTable').innerHTML = [
  ['Diámetro de arranque', `${f1(2 * R)} m`], ['Flecha de la cúpula', `${f1(H)} m`],
  ['Altura total (con tambor)', `${f1(totalHeight)} m`], ['Piezas modeladas', nParts],
  ['Paneles de vidrio', nGlass], ['Pernos', nBolts], ['Masa estimada', `${(totalMass / 1000).toFixed(1).replace('.', ',')} t`],
].map(([a, b]) => `<tr><td>${a}</td><td>${b}</td></tr>`).join('');
$('#matList').innerHTML = Object.entries(matCount).map(([k, v]) => `<li><span>${k}</span><b>${v}</b></li>`).join('');

const asmOrder = ['Tambor de sillería', 'Estructura de hierro', 'Vidriera', 'Linterna'];
const typesByAsm = {};
for (const [k, T] of Object.entries(TYPES)) (typesByAsm[T.asm] ||= []).push(k);
$('#asmList').innerHTML = asmOrder.map((a, i) => `<section><h4><i>${'ABCD'[i]}</i>${a}</h4>` +
  typesByAsm[a].map(k => `<button class="row" data-type="${k}"><span>${TYPES[k].name}</span><em>×${byType[k].length}</em><b>${TYPES[k].order}</b></button>`).join('') + '</section>').join('');
document.querySelectorAll('#asmList .row').forEach(b => b.addEventListener('click', () => {
  const t = b.dataset.type; select(byType[t][Math.floor(byType[t].length / 3)], true);
  if (innerWidth < 820) closeDrawers();
}));

// ------------------------------------------------------------------ selection + card
function select(part, focus) {
  sel = part ? { part, type: part.sp.type } : null;
  applyMaterials();
  const card = $('#card');
  if (!part) { card.classList.remove('on'); return; }
  const T = part.T, sp = part.sp;
  const orderTxt = `${T.order} de ${maxOrder} · ${STEPS[T.order - 1]}`;
  card.innerHTML = `<button id="cardX" aria-label="Cerrar">×</button>
    <small>${T.asm}</small><h3>${T.name}</h3>
    <dl><dt>Unidad</dt><dd>${sp.idx + 1} de ${sp.total}</dd>
    <dt>Material</dt><dd>${T.mat}</dd>
    <dt>Medidas</dt><dd>${sp.dims}</dd>
    <dt>Masa</dt><dd>≈ ${part.mass >= 1000 ? (part.mass / 1000).toFixed(2).replace('.', ',') + ' t' : Math.round(part.mass) + ' kg'}</dd>
    <dt>Montaje</dt><dd>${orderTxt}</dd>
    <dt>Función</dt><dd>${T.fn}</dd>
    <dt>Año</dt><dd>1915–1922 · reconstrucción</dd></dl>`;
  card.classList.add('on');
  $('#cardX').onclick = () => select(null);
  if (focus) {
    const w = new THREE.Vector3(); part.group.getWorldPosition(w);
    const big = ['anillo', 'clave', 'intermedio', 'cornisa', 'sillar', 'lcup', 'lbase', 'lcorn'].includes(part.sp.type);
    const dist = big ? 20 : 11;
    const out = new THREE.Vector3(w.x, 0, w.z);
    if (out.length() < 2.5) { out.set(camera.position.x - controls.target.x, 0, camera.position.z - controls.target.z); }
    out.normalize();
    const lift = innerWidth < 820 ? dist * 0.22 : 0;
    tweenTarget = w.clone().add(new THREE.Vector3(0, -lift, 0)); tweenPos = w.clone().addScaledVector(out, dist * 0.85).add(new THREE.Vector3(0, dist * 0.42 - lift, 0));
  }
}
let tweenTarget = null, tweenDist = 0, tweenPos = null;

// pointer picking
let down = null;
canvas.addEventListener('pointerdown', e => { down = { x: e.clientX, y: e.clientY, t: performance.now() }; });
canvas.addEventListener('pointerup', e => {
  if (!down) return;
  const d = Math.hypot(e.clientX - down.x, e.clientY - down.y), dt = performance.now() - down.t; down = null;
  if (d > 6 || dt > 500) return;
  const r = canvas.getBoundingClientRect();
  const ndc = new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  const rc = new THREE.Raycaster(); rc.setFromCamera(ndc, camera);
  const list = pickables.filter(m => m.parent.visible && !(cutOn && false));
  const hits = rc.intersectObjects(list, false).filter(h => {
    if (!cutOn) return true;
    return h.point.x <= clipPlane.constant + 0.01;
  });
  select(hits.length ? hits[0].object.userData.part : null, !!hits.length);
});

// ------------------------------------------------------------------ controls
const slider = $('#explode'), sVal = $('#sVal');
function setU(v, fromUser) {
  U = clamp(v); if (fromUser && S < 9) { S = 9; syncTimeline(); }
  slider.value = Math.round(U * 100); sVal.textContent = Math.round(U * 100) + ' %';
  $('#bExplode').classList.toggle('on', U > 0.5); $('#tgExplode').classList.toggle('on', U > 0.5);
  updateParts();
}
slider.addEventListener('input', () => { anim = null; setU(slider.value / 100, true); });
let anim = null;
function animateU(to, ms = 2600) { anim = { from: U, to, t0: performance.now(), ms }; }
function toggleExplode() { if (S < 9) { S = 9; syncTimeline(); } const to = U > 0.5 ? 0 : 1; animateU(to); frame_(to); }
$('#bExplode').addEventListener('click', toggleExplode);

function tog(id, get, set) {
  const el = $(id); el.addEventListener('click', () => { set(!get()); el.classList.toggle('on', get()); dirty = true; });
}
tog('#tgLabels', () => labelsOn, v => { labelsOn = v; });
tog('#tgGuides', () => guidesOn, v => { guidesOn = v; updateParts(); }); $('#tgGuides').classList.add('on');
tog('#tgCut', () => cutOn, v => setCut(v));
tog('#tgSpin', () => spinOn, v => { spinOn = v; controls.autoRotate = v; });
tog('#tgXray', () => xray, v => { xray = v; applyMaterials(); });
$('#tgExplode').addEventListener('click', toggleExplode);
tog('#tgQuality', () => quality === 'high', v => { quality = v ? 'high' : 'low'; applyQuality(); });
$('#tgQuality').classList.toggle('on', quality === 'high');
$('#resetView').addEventListener('click', () => { frame_(U); select(null); });

// timeline
const tl = $('#timeline'), tlTicks = $('#tlTicks');
tlTicks.innerHTML = STEPS.map((s, i) => `<button data-i="${i + 1}" title="${s}"><b>${i + 1}</b><span>${s}</span></button>`).join('');
let playing = false, playT0 = 0, playFrom = 0;
function syncTimeline() {
  tl.value = Math.round(S * 100);
  const step = S >= 9 ? 9 : Math.min(9, Math.floor(S) + 1);
  $('#tlLabel').textContent = S >= 9 && !playing ? 'Montaje completo' : `${step}. ${STEPS[step - 1]}`;
  [...tlTicks.children].forEach((b, i) => b.classList.toggle('done', i + 1 <= S - 0.0001 || S >= 9));
  $('#bPlay').textContent = playing ? '❚❚ Pausa' : '▶ Montaje';
}
tl.addEventListener('input', () => { playing = false; U = 0; slider.value = 0; sVal.textContent = '0 %'; S = tl.value / 100; syncTimeline(); updateParts(); });
tlTicks.addEventListener('click', e => {
  const b = e.target.closest('button'); if (!b) return;
  playing = false; U = 0; slider.value = 0; sVal.textContent = '0 %'; S = +b.dataset.i; syncTimeline(); updateParts();
});
$('#bPlay').addEventListener('click', () => {
  if (playing) { playing = false; syncTimeline(); return; }
  U = 0; slider.value = 0; sVal.textContent = '0 %';
  if (S >= 9) S = 0;
  playing = true; playT0 = performance.now(); playFrom = S; syncTimeline();
});

// drawers (mobile)
function closeDrawers() { document.body.classList.remove('showL', 'showR'); }
$('#openL').addEventListener('click', () => { document.body.classList.toggle('showL'); document.body.classList.remove('showR'); });
$('#openR').addEventListener('click', () => { document.body.classList.toggle('showR'); document.body.classList.remove('showL'); });
document.querySelectorAll('.drawerX').forEach(b => b.addEventListener('click', closeDrawers));

// ------------------------------------------------------------------ loop
function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  // keep the whole exploded model visible on tall (phone) screens
  camera.fov = w / h < 0.8 ? 52 : 38;
  camera.updateProjectionMatrix(); dirty = true;
}
addEventListener('resize', resize);

let last = performance.now();
function frame(now) {
  requestAnimationFrame(frame);
  const dt = Math.min(0.05, (now - last) / 1000); last = now;
  if (anim) {
    const k = clamp((now - anim.t0) / anim.ms);
    setU(anim.from + (anim.to - anim.from) * (k < 0.5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2));
    if (k >= 1) anim = null;
  }
  if (playing) {
    S = Math.min(9, playFrom + (now - playT0) / 2300);
    if (S >= 9) { playing = false; }
    syncTimeline(); updateParts();
  }
  if (cutOn || cutVal < 29) {
    const prev = cutVal;
    cutVal += (cutTarget - cutVal) * (1 - Math.exp(-dt * 2.2));
    if (Math.abs(cutVal - cutTarget) < 0.02) cutVal = cutTarget;
    clipPlane.constant = cutVal;
    if (!cutOn && cutVal > 29.5) renderer.clippingPlanes = [];
    if (prev !== cutVal) dirty = true;
  }
  if (tweenTarget) {
    controls.target.lerp(tweenTarget, 1 - Math.exp(-dt * 5));
    if (tweenPos) {
      camera.position.lerp(tweenPos, 1 - Math.exp(-dt * 4));
      if (camera.position.distanceTo(tweenPos) < 0.05 && controls.target.distanceTo(tweenTarget) < 0.03) { tweenTarget = null; tweenPos = null; }
    } else {
      const off = camera.position.clone().sub(controls.target);
      off.setLength(off.length() + (tweenDist - off.length()) * (1 - Math.exp(-dt * 4)));
      camera.position.copy(controls.target).add(off);
      if (controls.target.distanceTo(tweenTarget) < 0.03) tweenTarget = null;
    }
    dirty = true;
  }
  if (controls.update() || controls.autoRotate) dirty = true;
  if (dirty && !window.__pause) {
    dirty = false;
    updateLabels();
    renderer.render(scene, camera);
  }
}
resize(); applyQuality(); setU(0); syncTimeline();
{ const d = fitDist(0); camera.position.set(0.62 * d, 0.46 * d + 5, 0.72 * d); controls.target.set(0, 5.2, 0); }
document.body.classList.add('ready');
requestAnimationFrame(frame);
addEventListener('dome-wake', () => {
  dirty = true; resize();
  setTimeout(() => { if (U < 0.5 && S >= 9) { animateU(1, 3600); frame_(1); } }, 1100);
});
// test hooks
window.__dome = { fitDist, setU, select, parts, setCut, camera, controls, scene, renderer, get S() { return S; }, setS(v) { S = v; syncTimeline(); updateParts(); }, byType };
