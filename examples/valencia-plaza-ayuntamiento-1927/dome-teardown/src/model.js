// Geometry and part catalogue of the stained-glass dome (reconstruction).
import * as THREE from 'three';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';

export const R = 9.5;             // dome radius at springing (m)
export const H = 0.62 * R;        // rise of the ellipsoidal shell (m)
export const NR = 16;             // meridian ribs
export const DY = 0.8;            // springing level above the drum cornice
export const PHI_C = 1.35;        // crown ring latitude
export const ROW_EDGES = [0.03, 0.27, 0.52, 0.77, 1.0, 1.18, 1.33];   // 6 rows of glass
export const RING_PHI = [0.52, 1.0];
export const NSTONE = 24;

const TAU = Math.PI * 2;
export const ell = (th, ph) => new THREE.Vector3(R * Math.cos(ph) * Math.cos(th), DY + H * Math.sin(ph), R * Math.cos(ph) * Math.sin(th));
export const nrm = (th, ph) => new THREE.Vector3(Math.cos(ph) * Math.cos(th) / R, Math.sin(ph) / H, Math.cos(ph) * Math.sin(th) / R).normalize();
const radialDir = th => new THREE.Vector3(Math.cos(th), 0, Math.sin(th));

// ---------------------------------------------------------------- helpers
export function sectorGeo(r0, r1, th0, th1, y0, y1, bevel = 0.012) {
  const s = new THREE.Shape();
  const n = 6;
  const a0 = -th1, a1 = -th0;      // rotateX(-90deg) mirrors the angle
  for (let i = 0; i <= n; i++) {
    const a = a0 + (a1 - a0) * i / n;
    const x = r1 * Math.cos(a), y = r1 * Math.sin(a);
    i ? s.lineTo(x, y) : s.moveTo(x, y);
  }
  for (let i = n; i >= 0; i--) {
    const a = a0 + (a1 - a0) * i / n;
    s.lineTo(r0 * Math.cos(a), r0 * Math.sin(a));
  }
  s.closePath();
  const g = new THREE.ExtrudeGeometry(s, { depth: (y1 - y0) - 2 * bevel, bevelEnabled: true, bevelThickness: bevel,
    bevelSize: bevel, bevelSegments: 1, curveSegments: 1, steps: 1 });
  g.translate(0, 0, bevel);
  g.rotateX(-Math.PI / 2);
  g.translate(0, y0, 0);
  return g;
}

function lathe(points, seg = 96) {
  const g = new THREE.LatheGeometry(points.map(p => new THREE.Vector2(p[0], p[1])), seg);
  return g;
}

function boxAt(w, h, d, matrix) {
  const g = new THREE.BoxGeometry(w, h, d);
  g.applyMatrix4(matrix);
  return g;
}

const ni = g => (g.index ? g.toNonIndexed() : g);
function hexBolt(x, y, z, r = 0.045, h = 0.05) {
  const head = new THREE.CylinderGeometry(r, r, h, 6);
  head.translate(x, y + h / 2 + 0.012, z);
  const washer = new THREE.CylinderGeometry(r * 1.55, r * 1.55, 0.012, 16);
  washer.translate(x, y + 0.006, z);
  return mergeGeometries([ni(head), ni(washer)]);
}

// T-section rib swept along the meridian
export function ribGeo(th, phiA = 0.0, phiB = PHI_C, nseg = 22) {
  const radial = radialDir(th);
  const across = new THREE.Vector3(-Math.sin(th), 0, Math.cos(th));
  const prof = [[-0.03, -0.07], [0, -0.07], [0, 0.07], [-0.03, 0.07], [-0.03, 0.0175], [-0.2, 0.0175], [-0.2, -0.0175], [-0.03, -0.0175]];
  const rings = [];
  for (let i = 0; i <= nseg; i++) {
    const ph = phiA + (phiB - phiA) * i / nseg;
    const p0 = ell(th, ph);
    const N = nrm(th, ph);
    rings.push(prof.map(([n, s]) => p0.clone().addScaledVector(N, n).addScaledVector(across, s)));
  }
  const pos = [], uv = [];
  const push = (v, u, w) => { pos.push(v.x, v.y, v.z); uv.push(u, w); };
  for (let i = 0; i < nseg; i++) {
    for (let j = 0; j < prof.length; j++) {
      const j2 = (j + 1) % prof.length;
      const a = rings[i][j], b = rings[i][j2], c = rings[i + 1][j2], d = rings[i + 1][j];
      const u0 = i * 0.9, u1 = (i + 1) * 0.9, w0 = j * 0.3, w1 = (j + 1) * 0.3;
      push(a, u0, w0); push(b, u0, w1); push(c, u1, w1);
      push(a, u0, w0); push(c, u1, w1); push(d, u1, w0);
    }
  }
  const tri = THREE.ShapeUtils.triangulateShape(prof.map(p => new THREE.Vector2(p[0], p[1])), []);
  for (const end of [0, nseg]) {
    for (const t of tri) for (const k of (end === 0 ? [t[0], t[2], t[1]] : t)) push(rings[end][k], 0, 0);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  g.computeVertexNormals();
  return g;
}

// ring of rectangular section on the shell at latitude phi
export function shellRing(phi, nIn, mHalf, nOut = 0, seg = 96) {
  const rc = R * Math.cos(phi), yc = DY + H * Math.sin(phi);
  const n2 = new THREE.Vector2(H * Math.cos(phi), R * Math.sin(phi)).normalize();
  const t2 = new THREE.Vector2(-R * Math.sin(phi), H * Math.cos(phi)).normalize();
  const pts = [[-nIn, -mHalf], [nOut, -mHalf], [nOut, mHalf], [-nIn, mHalf], [-nIn, -mHalf]]
    .map(([n, m]) => [rc + n2.x * n + t2.x * m, yc + n2.y * n + t2.y * m]);
  return lathe(pts, seg);
}

function patchPoint(k, row, u, v, inset, off) {
  const th0 = TAU * k / NR, th1 = TAU * (k + 1) / NR;
  const p0 = ROW_EDGES[row], p1 = ROW_EDGES[row + 1];
  const g = 0.012 + inset / R;
  const ph = p0 + g + (p1 - p0 - 2 * g) * v;
  const d = (0.085 + inset) / (R * Math.cos(ph));
  const th = th0 + d + (th1 - th0 - 2 * d) * u;
  return ell(th, ph).addScaledVector(nrm(th, ph), off);
}

export function glassPatchGeo(k, row, inset, off, nu = 4, nv = 4) {
  const pos = [];
  for (let j = 0; j <= nv; j++) for (let i = 0; i <= nu; i++) {
    const p = patchPoint(k, row, i / nu, j / nv, inset, off);
    pos.push(p.x, p.y, p.z);
  }
  const idx = [];
  for (let j = 0; j < nv; j++) for (let i = 0; i < nu; i++) {
    const a = j * (nu + 1) + i, b = a + 1, c = a + nu + 2, d = a + nu + 1;
    idx.push(a, b, c, a, c, d);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setIndex(idx);
  g.computeVertexNormals();
  return g;
}

export function cameGeo(k, row, w = 0.035, off = 0.004, ns = 5) {
  const loop = (inset) => {
    const pts = [];
    for (let i = 0; i < ns; i++) pts.push([i / ns, 0]);
    for (let i = 0; i < ns; i++) pts.push([1, i / ns]);
    for (let i = 0; i < ns; i++) pts.push([1 - i / ns, 1]);
    for (let i = 0; i < ns; i++) pts.push([0, 1 - i / ns]);
    return pts.map(([u, v]) => patchPoint(k, row, u, v, inset, off));
  };
  const o = loop(-w * 0.0), inn = loop(w);
  const pos = [];
  const n = o.length;
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    for (const p of [o[i], o[j], inn[j], o[i], inn[j], inn[i]]) pos.push(p.x, p.y, p.z);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.computeVertexNormals();
  return g;
}

// ---------------------------------------------------------------- volumes
export function meshVolume(g) {
  const p = g.attributes.position, idx = g.index;
  let v = 0;
  const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3();
  const n = idx ? idx.count : p.count;
  for (let i = 0; i < n; i += 3) {
    const i0 = idx ? idx.getX(i) : i, i1 = idx ? idx.getX(i + 1) : i + 1, i2 = idx ? idx.getX(i + 2) : i + 2;
    a.fromBufferAttribute(p, i0); b.fromBufferAttribute(p, i1); c.fromBufferAttribute(p, i2);
    v += a.dot(b.clone().cross(c)) / 6;
  }
  return Math.abs(v);
}
export function meshArea(g) {
  const p = g.attributes.position, idx = g.index;
  let s = 0;
  const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3();
  const n = idx ? idx.count : p.count;
  for (let i = 0; i < n; i += 3) {
    const i0 = idx ? idx.getX(i) : i, i1 = idx ? idx.getX(i + 1) : i + 1, i2 = idx ? idx.getX(i + 2) : i + 2;
    a.fromBufferAttribute(p, i0); b.fromBufferAttribute(p, i1); c.fromBufferAttribute(p, i2);
    s += b.clone().sub(a).cross(c.clone().sub(a)).length() / 2;
  }
  return s;
}

// ---------------------------------------------------------------- catalogue
// order: assembly sequence 1..9 (reverse order = explode order)
export const STEPS = ['Sillares del tambor', 'Cornisa de coronación', 'Anillo de arranque y pernos', 'Nervios meridianos',
  'Anillos intermedios', 'Anillo de clave y placas', 'Vidriera emplomada', 'Medallón con el escudo', 'Linterna y remate'];

export const TYPES = {
  sillar: { name: 'Sillar del tambor', asm: 'Tambor de sillería', mat: 'Caliza labrada', mk: 'stone', den: 2400, order: 1,
    fn: 'Recoge el peso de la cúpula y lo reparte al muro del patio de operaciones. Dos hiladas de 24 sillares, a junta encontrada.' },
  cornisa: { name: 'Pieza de cornisa', asm: 'Tambor de sillería', mat: 'Caliza labrada', mk: 'stone', den: 2400, order: 2,
    fn: 'Remata el tambor y ofrece el asiento nivelado al anillo de hierro. Dos escalones moldurados.' },
  anillo: { name: 'Anillo de arranque', asm: 'Estructura de hierro', mat: 'Hierro laminado pintado', mk: 'iron', den: 7800, order: 3,
    fn: 'Anillo perfilado en doble T que absorbe el empuje horizontal de los 16 nervios (trabaja a tracción).' },
  perno: { name: 'Perno con arandela', asm: 'Estructura de hierro', mat: 'Acero forjado', mk: 'steel', den: 7850, order: 3,
    fn: 'Une cada nervio al anillo de arranque; dos pernos por nervio.' },
  nervio: { name: 'Nervio meridiano', asm: 'Estructura de hierro', mat: 'Hierro laminado roblonado', mk: 'iron', den: 7800, order: 4,
    fn: 'Perfil en T curvado según la meridiana de la cúpula; sostiene los paños de vidrio y la linterna.' },
  intermedio: { name: 'Anillo intermedio', asm: 'Estructura de hierro', mat: 'Hierro laminado pintado', mk: 'iron', den: 7800, order: 5,
    fn: 'Arriostra los nervios entre sí e impide su pandeo; define las hiladas de vidriera.' },
  clave: { name: 'Anillo de clave', asm: 'Estructura de hierro', mat: 'Hierro laminado pintado', mk: 'iron', den: 7800, order: 6,
    fn: 'Recoge la cabeza de los 16 nervios y sirve de base a la linterna.' },
  placa: { name: 'Placa de unión con 4 pernos', asm: 'Estructura de hierro', mat: 'Hierro y acero', mk: 'iron', den: 7800, order: 6,
    fn: 'Chapa que atornilla cada nervio al anillo de clave.' },
  vidrio: { name: 'Panel de vidriera emplomada', asm: 'Vidriera', mat: 'Vidrio coloreado y plomo', mk: 'glass', den: 2500, order: 7, shell: 0.006,
    fn: 'Paño curvo de vidrio de color unido con cintas de plomo; filtra la luz natural sobre el patio.' },
  medallon: { name: 'Medallón del escudo', asm: 'Vidriera', mat: 'Vidrio esmaltado y hierro', mk: 'medal', den: 4800, order: 8,
    fn: 'Escudo de València en el centro del óculo.' },
  lbase: { name: 'Zócalo de la linterna', asm: 'Linterna', mat: 'Hierro fundido', mk: 'iron', den: 7200, order: 9, fn: 'Asiento de la linterna sobre el anillo de clave.' },
  lcol: { name: 'Columnilla de la linterna', asm: 'Linterna', mat: 'Hierro fundido', mk: 'iron', den: 7200, order: 9, fn: 'Soporte vertical de los cristales de la linterna.' },
  lvidrio: { name: 'Cristal de la linterna', asm: 'Linterna', mat: 'Vidrio ámbar', mk: 'glass', den: 2500, order: 9, shell: 0.008, fn: 'Ilumina el vértice y ventila la cúpula.' },
  lcorn: { name: 'Entablamento de la linterna', asm: 'Linterna', mat: 'Hierro fundido', mk: 'iron', den: 7200, order: 9, fn: 'Anillo que cierra la linterna y recibe el cupulín.' },
  lcup: { name: 'Cupulín', asm: 'Linterna', mat: 'Chapa de plomo sobre bastidor', mk: 'lead', den: 11300, order: 9, shell: 0.004,
    fn: 'Cubierta de la linterna, estanca a la lluvia.' },
  remate: { name: 'Remate con bola', asm: 'Linterna', mat: 'Latón dorado', mk: 'brass', den: 8500, order: 9, fn: 'Pararrayos y remate ornamental.' },
};

function tr(m) { return new THREE.Matrix4().copy(m); }

// returns list of parts specs: {type, idx, total, geos:[{g, mk}], vec:THREE.Vector3, dims}
export function buildSpecs() {
  const specs = [];
  const add = (type, idx, total, geos, vec, dims, extra = {}) => specs.push({ type, idx, total, geos, vec, dims, ...extra });
  const fm = v => String(v.toFixed ? v.toFixed(2) : v).replace('.', ',');
  const f1 = v => v.toFixed(1).replace('.', ',');
  const f0 = v => Math.round(v * 1000) + ' mm';

  // --- A1 sillares (two courses, 24 each, staggered)
  const gap = 0.004;
  for (let c = 0; c < 2; c++) {
    const y0 = c === 0 ? -1.6 : -0.8, y1 = c === 0 ? -0.8 : 0.0;
    for (let k = 0; k < NSTONE; k++) {
      const off = c === 1 ? 0.5 : 0;
      const th0 = TAU * (k + off) / NSTONE + gap, th1 = TAU * (k + 1 + off) / NSTONE - gap;
      const mid = (th0 + th1) / 2;
      const g = sectorGeo(R - 0.1, R + 0.7, th0, th1, y0, y1);
      const rd = radialDir(mid);
      const vec = rd.clone().multiplyScalar(c === 0 ? 1.8 : 1.3).add(new THREE.Vector3(0, c === 0 ? -2.8 : -1.6, 0));
      add('sillar', c * NSTONE + k, 2 * NSTONE, [{ g, mk: 'stone' }], vec,
        `0,80 × ${fm((R + 0.7 - (R - 0.1)))} × ${fm(TAU * (R + 0.3) / NSTONE)} m · hilada ${c + 1}`);
    }
  }
  // --- A2 cornisa
  for (let k = 0; k < NSTONE; k++) {
    const th0 = TAU * k / NSTONE + gap, th1 = TAU * (k + 1) / NSTONE - gap, mid = (th0 + th1) / 2;
    const g = mergeGeometries([ni(sectorGeo(R - 0.05, R + 1.05, th0, th1, 0.0, 0.26, 0.01)),
      ni(sectorGeo(R - 0.05, R + 0.88, th0, th1, 0.26, 0.42, 0.01))]);
    add('cornisa', k, NSTONE, [{ g, mk: 'stone' }], radialDir(mid).multiplyScalar(2.6).add(new THREE.Vector3(0, -0.4, 0)),
      `0,42 × 1,10 × ${fm(TAU * (R + 0.5) / NSTONE)} m · 2 escalones`);
  }
  // --- B1 anillo de arranque (T section ring) and bolts
  {
    const ring = lathe([[R - 0.25, 0.42], [R + 0.25, 0.42], [R + 0.25, 0.47], [R + 0.04, 0.47], [R + 0.04, 0.75], [R + 0.15, 0.75],
      [R + 0.15, 0.8], [R - 0.15, 0.8], [R - 0.15, 0.75], [R - 0.04, 0.75], [R - 0.04, 0.47], [R - 0.25, 0.47], [R - 0.25, 0.42]], 128);
    add('anillo', 0, 1, [{ g: ring, mk: 'iron' }], new THREE.Vector3(0, 1.3, 0),
      `Ø ${f1(2 * R)} m · sección en doble T 500 × 380 mm · perímetro ${f1(TAU * R)} m`);
    for (let k = 0; k < NR; k++) for (const s of [-1, 1]) {
      const th = TAU * k / NR + s * 0.028;
      const g = hexBolt(R * Math.cos(th), 0.8, R * Math.sin(th));
      add('perno', k * 2 + (s > 0 ? 1 : 0), NR * 2, [{ g, mk: 'steel' }], new THREE.Vector3(Math.cos(th) * 0.5, 2.9, Math.sin(th) * 0.5),
        'M20 · cabeza hexagonal 36 mm · arandela Ø 70 mm');
    }
  }
  // --- B2 ribs
  for (let k = 0; k < NR; k++) {
    const th = TAU * k / NR;
    const g = ribGeo(th);
    // arc length of the rib
    let L = 0, pr = ell(th, 0);
    for (let i = 1; i <= 40; i++) { const p = ell(th, PHI_C * i / 40); L += p.distanceTo(pr); pr = p; }
    const rd = radialDir(th);
    add('nervio', k, NR, [{ g, mk: 'iron' }], rd.clone().multiplyScalar(2.3).add(new THREE.Vector3(0, 3.4, 0)),
      `Perfil T 140 × 200 mm · longitud ${f1(L)} m`);
  }
  // --- B3 intermediate rings
  RING_PHI.forEach((ph, i) => {
    const g = shellRing(ph, 0.14, 0.04);
    add('intermedio', i, 2, [{ g, mk: 'iron' }], new THREE.Vector3(0, 4.6 + i * 1.3, 0),
      `Ø ${f1(2 * R * Math.cos(ph))} m · barra 140 × 80 mm`);
  });
  // --- B4 crown ring and connection plates
  {
    const g = shellRing(PHI_C, 0.3, 0.12);
    add('clave', 0, 1, [{ g, mk: 'iron' }], new THREE.Vector3(0, 7.6, 0), `Ø ${f1(2 * R * Math.cos(PHI_C))} m · sección 300 × 240 mm`);
    for (let k = 0; k < NR; k++) {
      const th = TAU * k / NR, ph = PHI_C - 0.05;
      const p = ell(th, ph);
      const N = new THREE.Vector3(Math.cos(ph) * Math.cos(th) / R, Math.sin(ph) / H, Math.cos(ph) * Math.sin(th) / R).normalize();
      const across = new THREE.Vector3(-Math.sin(th), 0, Math.cos(th));
      const T = new THREE.Vector3().crossVectors(across, N).normalize();
      const m = new THREE.Matrix4().makeBasis(across, N, T).setPosition(p.clone().addScaledVector(N, 0.03));
      const parts = [boxAt(0.34, 0.03, 0.46, m)];
      for (const sx of [-1, 1]) for (const sz of [-1, 1]) {
        const c = new THREE.CylinderGeometry(0.035, 0.035, 0.03, 6);
        c.translate(sx * 0.1, 0.03, sz * 0.15);
        c.applyMatrix4(m);
        parts.push(c);
      }
      const g2 = mergeGeometries(parts.map(q => ni(q)));
      add('placa', k, NR, [{ g: g2, mk: 'iron' }], radialDir(th).multiplyScalar(0.6).add(new THREE.Vector3(0, 9.0, 0)),
        '340 × 460 × 30 mm · 4 pernos M16');
    }
  }
  // --- C stained glass panels
  const palette = [0xd0892b, 0x2f66a8, 0xb23a2e, 0x3f9060, 0xe0bb4a, 0x7c4d9e];
  for (let row = 0; row < 6; row++) for (let k = 0; k < NR; k++) {
    const th = TAU * (k + 0.5) / NR;
    const ph = (ROW_EDGES[row] + ROW_EDGES[row + 1]) / 2;
    const N = nrm(th, ph);
    const glass = glassPatchGeo(k, row, 0.035, -0.003);
    const came = cameGeo(k, row);
    const col = palette[(k * 3 + row * 2 + (row % 2) * 1) % palette.length];
    const a = patchPoint(k, row, 0, 0.5, 0, 0).distanceTo(patchPoint(k, row, 1, 0.5, 0, 0));
    const b = patchPoint(k, row, 0.5, 0, 0, 0).distanceTo(patchPoint(k, row, 0.5, 1, 0, 0));
    add('vidrio', row * NR + k, 6 * NR, [{ g: glass, mk: 'glass', color: col }, { g: came, mk: 'lead' }],
      N.clone().multiplyScalar(3.8 + row * 0.35).add(new THREE.Vector3(0, 0.4 * row, 0)),
      `${fm(a)} × ${fm(b)} m · e = 6 mm · hilada ${row + 1}/6`, { color: col, row });
  }
  // --- C2 medallion
  {
    const yc = DY + H * Math.sin(PHI_C) - 0.12;
    const g = new THREE.CylinderGeometry(1.45, 1.45, 0.07, 56);
    g.translate(0, yc, 0);
    add('medallon', 0, 1, [{ g, mk: 'medal' }], new THREE.Vector3(0, -4.4, 0), 'Ø 2,90 m · e = 70 mm');
  }
  // --- D lantern
  const L0 = DY + H * Math.sin(PHI_C) + 0.15;
  {
    const g = lathe([[1.5, L0], [2.0, L0], [2.0, L0 + 0.12], [1.86, L0 + 0.16], [1.86, L0 + 0.3], [1.62, L0 + 0.35], [1.5, L0 + 0.35], [1.5, L0]], 64);
    add('lbase', 0, 1, [{ g, mk: 'iron' }], new THREE.Vector3(0, 5.2, 0), 'Ø 4,00 m · h = 350 mm');
    for (let k = 0; k < 8; k++) {
      const a = TAU * k / 8;
      const c = new THREE.CylinderGeometry(0.07, 0.085, 1.45, 14);
      c.translate(1.62 * Math.cos(a), L0 + 0.35 + 0.725, 1.62 * Math.sin(a));
      add('lcol', k, 8, [{ g: c, mk: 'iron' }], radialDir(a).multiplyScalar(0.9).add(new THREE.Vector3(0, 6.6, 0)), 'Ø 150 mm · h = 1,45 m');
      const th0 = TAU * k / 8 + TAU / 16 - TAU / 16 + 0.09, th1 = TAU * (k + 1) / 8 - 0.09;
      const cg = new THREE.CylinderGeometry(1.6, 1.6, 1.34, 8, 1, true, Math.PI / 2 - th1, th1 - th0);
      cg.translate(0, L0 + 0.35 + 0.72, 0);
      add('lvidrio', k, 8, [{ g: cg, mk: 'glass', color: 0xd9a03a }], radialDir((th0 + th1) / 2).multiplyScalar(2.6).add(new THREE.Vector3(0, 6.6, 0)),
        'Cristal curvo e = 8 mm · h = 1,34 m', { color: 0xd9a03a });
    }
    const ent = lathe([[1.5, L0 + 1.8], [1.98, L0 + 1.8], [1.98, L0 + 1.92], [1.8, L0 + 1.98], [1.8, L0 + 2.1], [1.5, L0 + 2.1], [1.5, L0 + 1.8]], 64);
    add('lcorn', 0, 1, [{ g: ent, mk: 'iron' }], new THREE.Vector3(0, 8.2, 0), 'Ø 3,96 m · h = 300 mm');
    const pts = [];
    const ns = 14;
    for (let i = 0; i <= ns; i++) { const a = (Math.PI / 2) * i / ns; pts.push([1.8 * Math.cos(a) + 0.0001, L0 + 2.1 + 1.55 * Math.sin(a)]); }
    const back = [];
    for (let i = ns; i >= 0; i--) { const a = (Math.PI / 2) * i / ns; back.push([1.75 * Math.cos(a) + 0.0001, L0 + 2.1 - 0.0 + 1.5 * Math.sin(a)]); }
    const cup = lathe([...pts, ...back].reverse(), 64);
    add('lcup', 0, 1, [{ g: cup, mk: 'lead' }], new THREE.Vector3(0, 9.8, 0), 'Ø 3,60 m · flecha 1,55 m · e = 50 mm');
    const top = L0 + 2.1 + 1.55;
    const rod = new THREE.CylinderGeometry(0.03, 0.05, 1.5, 12); rod.translate(0, top + 0.75, 0);
    const ball = new THREE.SphereGeometry(0.2, 18, 12); ball.translate(0, top + 1.0, 0);
    const ball2 = new THREE.SphereGeometry(0.1, 14, 10); ball2.translate(0, top + 1.55, 0);
    add('remate', 0, 1, [{ g: mergeGeometries([ni(rod), ni(ball), ni(ball2)]), mk: 'brass' }],
      new THREE.Vector3(0, 12.2, 0), 'h = 1,6 m · bola Ø 400 mm');
  }
  return specs;
}
