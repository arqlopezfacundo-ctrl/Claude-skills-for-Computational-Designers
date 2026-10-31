"""Vertical (9:16) composition of the Correos film for phones.

    python film_post.py --jobs 3

 ┌──────────────────────────┐  header: title plate
 │                          │
 │   3D film / unfolding    │  1080 x 1200 viewport (rendered 900 x 1000)
 │          map             │
 ├──────────────────────────┤
 │ PLANTA · SECCIÓN · AXO   │  technical sheet drawn in sync with the works
 │ phase · 1915 ━━━━ 1922   │
 └──────────────────────────┘
"""
import argparse
import json
import math
import os
import sys
from multiprocessing import Pool

import numpy as np
import OpenEXR
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from vlc import plan  # noqa: E402

FPS, DUR = 24, 40.0
N = int(FPS * DUR)
VW, VH = 1080, 1920
VP = (0, 170, 1080, 1370)          # viewport box
PANEL_Y = 1388
T_MAP = 3.0
G0, HF, NB = 4.6, 3.3, 6
PIT_Z = -3.2
INK = np.array([0.23, 0.16, 0.11])
RED = (150, 44, 30)
INKc = (58, 40, 28)
FB = "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf"
FI = "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf"
FR = "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"
CACHE = {}


def smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def sstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def p_of_t(t):
    u = min(max((t - 4.0) / 30.5, 0.0), 1.0)
    return 0.12 * u * u * (3 - 2 * u) + 0.88 * u


def t_band(k):
    return 0.20 + 0.068 * k


def band_z(k):
    return (0.0, G0) if k == 0 else (G0 + HF * (k - 1), G0 + HF * k)


def fbm(h, w, seed, oct_=6, base=4, pers=0.55):
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(oct_):
        n = base * 2 ** o
        g = rng.random((n + 3, int(n * w / h) + 3)).astype(np.float32)
        z = ndi.zoom(g, (h / n, h / n), order=3)[:h, :w]
        z = np.pad(z, ((0, max(0, h - z.shape[0])), (0, max(0, w - z.shape[1]))), mode="edge")
        out += amp * z
        tot += amp
        amp *= pers
    out /= tot
    return (out - out.min()) / (out.max() - out.min() + 1e-6)


# --------------------------------------------------------------- statics
def paper(w, h, seed=1):
    k = ("paper", w, h, seed)
    if k in CACHE:
        return CACHE[k]
    n1 = fbm(h, w, seed, 7, 3)
    n2 = fbm(h, w, seed + 1, 4, 40)
    base = np.array([0.925, 0.885, 0.80])
    img = base[None, None] * (0.95 + 0.07 * n1[..., None] + 0.02 * (n2[..., None] - 0.5))
    CACHE[k] = img.astype(np.float32)
    return CACHE[k]


def sky_pano():
    """Equirectangular painted sky: elevation 0..90 deg (rows), azimuth 0..360."""
    if "sky" in CACHE:
        return CACHE["sky"]
    path = os.path.join(HERE, "film", "sky_pano.npy")
    if os.path.exists(path):
        CACHE["sky"] = np.load(path).astype(np.float32)
        return CACHE["sky"]
    H, W = 768, 3072
    el = np.linspace(90, 0, H)[:, None] * np.ones((1, W))
    az = np.linspace(0, 360, W)[None, :] * np.ones((H, 1))
    k = np.clip(el / 60.0, 0, 1) ** 0.6
    zen = np.array([0.30, 0.47, 0.78])
    hor = np.array([0.93, 0.84, 0.72])
    sky = hor[None, None] * (1 - k[..., None]) + zen[None, None] * k[..., None]
    sun_az, sun_el = 62.0, 24.0
    # Blender sun azimuth: rotation about z, light from the direction of sun_dir
    sv = np.array([math.cos(math.radians(sun_el)) * math.sin(math.radians(sun_az)),
                   -math.cos(math.radians(sun_el)) * math.cos(math.radians(sun_az)),
                   math.sin(math.radians(sun_el))])
    saz = (math.degrees(math.atan2(sv[1], sv[0]))) % 360
    d = np.sqrt(((az - saz + 180) % 360 - 180) ** 2 * np.cos(np.radians(el)) ** 2 + (el - sun_el) ** 2)
    sky += np.array([1.0, 0.78, 0.5])[None, None] * (0.35 * np.exp(-(d / 22) ** 2) + 0.12 * np.exp(-(d / 60) ** 2))[..., None]
    # clouds: flattened fbm, thinning to the zenith, lit from the sun side
    c = fbm(H, W, 41, 7, 3, 0.6)
    warp = fbm(H, W, 42, 4, 4)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    c = ndi.map_coordinates(c, [np.clip(yy + (warp - 0.5) * 60, 0, H - 1), (xx + (warp - 0.5) * 140) % W],
                            order=1)
    band = np.exp(-((el - 18) / 22) ** 2) + 0.35 * np.exp(-((el - 45) / 15) ** 2)
    dens = sstep(0.52, 0.72, c * (0.7 + 0.5 * band))
    dens = ndi.gaussian_filter(dens, (1.5, 3))
    gy, gx = np.gradient(ndi.gaussian_filter(dens, 6))
    toward = np.array([math.cos(math.radians(saz)), 0.4])
    lit = np.clip(0.55 + 6.0 * (-gx * np.sign(((saz - az + 180) % 360 - 180)) + gy * 0.6), 0, 1)
    hi = np.array([1.0, 0.93, 0.82])
    lo = np.array([0.66, 0.68, 0.78])
    ccol = lo[None, None] + (hi - lo)[None, None] * lit[..., None]
    sky = sky * (1 - dens[..., None] * 0.92) + ccol * dens[..., None] * 0.92
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, sky.astype(np.float16))
    CACHE["sky"] = sky.astype(np.float32)
    return CACHE["sky"]


# --------------------------------------------------------------- geometry
def correos_geo():
    if "geo" in CACHE:
        return CACHE["geo"]
    from shapely.geometry import Point
    from shapely.geometry.polygon import orient
    F = orient(plan.footprint(plan.block_at((70, 90))).simplify(0.2), 1.0)
    hall = F.buffer(-13.0, join_style=2)
    c = list(F.exterior.coords)[:-1]
    best = None
    for i in range(len(c)):
        a, b = np.array(c[i]), np.array(c[(i + 1) % len(c)])
        m = (a + b) / 2
        dd = abs(m[1] - plan.y_n(m[0]))
        if best is None or dd < best[0]:
            best = (dd, a, b)
    _, A, B = best
    t = (B - A) / np.linalg.norm(B - A)
    out = np.array([t[1], -t[0]])
    if out[1] > 0:
        out = -out
    ext = F.exterior
    s0 = ext.project(Point(*A))
    bays = []
    for i in range(len(c)):
        a, b = np.array(c[i]), np.array(c[(i + 1) % len(c)])
        ln = np.linalg.norm(b - a)
        d = (b - a) / ln
        o = np.array([d[1], -d[0]])
        if F.contains(Point(*((a + b) / 2 + o * 0.8))):
            o = -o
        n = max(1, int(round(ln / 3.9)))
        for j in range(n):
            pa = a + d * ln * j / n
            pb = a + d * ln * (j + 1) / n
            mid = (pa + pb) / 2
            s = ((ext.project(Point(*mid)) - s0) % ext.length) / ext.length
            bays.append((pa, pb, d, o, s))
    ring = F.difference(hall)
    minx, miny, maxx, maxy = ring.bounds
    inner = ring.buffer(-1.6)
    cols = []
    for ix in range(int((maxx - minx) / 6) + 1):
        for iy in range(int((maxy - miny) / 6) + 1):
            q = Point(minx + 2 + ix * 6, miny + 2 + iy * 6)
            if inner.contains(q):
                s = ((ext.project(q) - s0) % ext.length) / ext.length
                cols.append((q.x, q.y, s))
    hc = hall.centroid
    geo = dict(F=F, hall=hall, A=A, B=B, t=t, out=out, fm=(A + B) / 2, bays=bays, cols=cols,
               dome=(hc.x, hc.y))
    CACHE["geo"] = geo
    return geo


# --------------------------------------------------------------- drawings
def font(path, size):
    k = (path, size)
    if k not in CACHE:
        CACHE[k] = ImageFont.truetype(path, size)
    return CACHE[k]


def draw_panel(p, t):
    """Technical sheet: plan, section and axonometric drawn as the works go."""
    S = 2
    W, H = VW * S, (VH - PANEL_Y) * S
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    geo = correos_geo()
    F, hall = geo["F"], geo["hall"]
    ink = INKc + (235,)
    inkl = INKc + (120,)
    red = RED + (220,)
    k_cur = max([k for k in range(NB) if p >= t_band(k) - 0.01] or [-1])
    # three frames
    boxes = [(20, 10, 360, 400), (370, 10, 710, 400), (720, 10, 1060, 400)]
    for (x0, y0, x1, y1) in boxes:
        d.rectangle([x0 * S, y0 * S, x1 * S, y1 * S], outline=ink, width=2 * S)
        d.rectangle([(x0 + 5) * S, (y0 + 5) * S, (x1 - 5) * S, (y1 - 5) * S], outline=inkl, width=S)

    # ---------------- PLANTA
    x0, y0, x1, y1 = boxes[0]
    minx, miny, maxx, maxy = F.bounds
    sc = min((x1 - x0 - 60) / (maxx - minx), (y1 - y0 - 90) / (maxy - miny)) * S
    cx, cy = (x0 + x1) / 2 * S, (y0 + y1 - 30) / 2 * S
    mx, my = (minx + maxx) / 2, (miny + maxy) / 2

    def PP(x, y):
        return (cx + (x - mx) * sc, cy - (y - my) * sc)

    d.line([PP(*c) for c in F.exterior.coords], fill=inkl, width=S)          # setting-out
    if p < 0.18:
        a = min(1.0, max(0.0, (p - 0.03) / 0.09))
        for kx in np.arange(minx - 70, maxx, 3.0):
            from shapely.geometry import LineString
            ln = LineString([(kx, miny), (kx + 70 * a, miny + 70 * a)]).intersection(F)
            for g in getattr(ln, "geoms", [ln]):
                if not g.is_empty and g.geom_type == "LineString":
                    d.line([PP(*c) for c in g.coords], fill=(120, 80, 50, 110), width=S)
    if k_cur >= 0:
        prog = min(1.0, max(0.0, (p - t_band(k_cur)) / 0.045))
        for (pa, pb, dd, o, s) in geo["bays"]:
            if s > prog and p < t_band(k_cur) + 0.06:
                continue
            ln = np.linalg.norm(pb - pa)
            cu = ln / 2
            ww = 2.0 if k_cur == 0 else 1.45
            for u0, u1 in ((0, cu - ww / 2), (cu + ww / 2, ln)):
                q = [pa + dd * u0, pa + dd * u1, pa + dd * u1 - o * 0.75, pa + dd * u0 - o * 0.75]
                d.polygon([PP(*v) for v in q], fill=INKc + (230,))
            g0, g1 = pa + dd * (cu - ww / 2) - o * 0.4, pa + dd * (cu + ww / 2) - o * 0.4
            d.line([PP(*g0), PP(*g1)], fill=ink, width=S)
        d.line([PP(*c) for c in hall.exterior.coords], fill=ink, width=3 * S)
        for (x, y, s) in geo["cols"]:
            if s <= prog + 0.2 or p > t_band(k_cur) + 0.05:
                X, Y = PP(x, y)
                r = 0.35 * sc
                d.rectangle([X - r, Y - r, X + r, Y + r], fill=ink)
    dc = geo["dome"]
    X, Y = PP(*dc)
    r = 9.5 * sc
    if p > 0.6:
        d.ellipse([X - r, Y - r, X + r, Y + r], outline=red, width=2 * S)
        for k in range(16):
            a = 2 * math.pi * k / 16
            if p > 0.62 + 0.05 * k / 16:
                d.line([(X, Y), (X + r * math.cos(a), Y + r * math.sin(a))], fill=red, width=S)
    else:
        for k in range(24):
            a0, a1 = 2 * math.pi * k / 24, 2 * math.pi * (k + 0.5) / 24
            d.arc([X - r, Y - r, X + r, Y + r], math.degrees(a0), math.degrees(a1), fill=inkl, width=S)
    # north arrow + scale
    nx, ny = (x1 - 28) * S, (y0 + 34) * S
    d.polygon([(nx, ny - 18 * S), (nx - 7 * S, ny + 10 * S), (nx, ny + 4 * S), (nx + 7 * S, ny + 10 * S)], fill=ink)
    for k in range(4):
        xa = (x0 + 18) * S + k * 10 * sc
        d.rectangle([xa, (y1 - 46) * S, xa + 10 * sc, (y1 - 42) * S], fill=ink if k % 2 == 0 else None, outline=ink)
    lvl = "+0,00" if k_cur <= 0 else "+%.2f" % band_z(k_cur)[0]
    _label(d, S, (x0 + x1) / 2, y1 - 26, "PLANTA · NIVEL %s" % lvl.replace(".", ","))

    # ---------------- SECCIÓN A-A'
    x0, y0, x1, y1 = boxes[1]
    fm, out = geo["fm"], geo["out"]
    depth = float(np.dot(np.array([plan.footprint(plan.block_at((70, 90))).bounds[2],
                                   plan.footprint(plan.block_at((70, 90))).bounds[3]]) - fm, -out))
    D = 58.0
    zs_max = 52.0
    sc2 = min((x1 - x0 - 70) / (D + 16), (y1 - y0 - 80) / (zs_max + 8)) * S
    gx0 = (x0 + 30) * S
    gy = (y1 - 50) * S

    def SP(u, z):        # u: metres from the plaza front inwards
        return (gx0 + (u + 8) * sc2, gy - z * sc2)

    d.line([SP(-8, 0), SP(D + 8, 0)], fill=ink, width=2 * S)
    if p < 0.2:
        a = min(1.0, max(0.0, (p - 0.03) / 0.09))
        d.rectangle([SP(0, 0)[0], SP(0, 0)[1], SP(D, PIT_Z * a)[0], SP(D, PIT_Z * a)[1]], outline=ink, width=S)
    d.line([SP(0, PIT_Z), SP(D, PIT_Z)], fill=inkl, width=S)
    for k in range(NB):
        z0, z1 = band_z(k)
        f = min(1.0, max(0.0, (p - t_band(k)) / 0.06))
        if f <= 0:
            continue
        zt = z0 + (z1 - z0) * f
        for (u0, u1) in ((0, 13.0), (D - 13.0, D)):
            d.rectangle([SP(u0, zt)[0], SP(u0, zt)[1], SP(u1, z0)[0], SP(u1, z0)[1]],
                        fill=(196, 160, 120, 90), outline=ink, width=S)
            for uu in (u0, u1):
                d.line([SP(uu, z0), SP(uu, zt)], fill=ink, width=3 * S)
        if f >= 1 and p < 0.86:   # scaffolds on the outer faces
            for uu in (-1.5, D + 1.5):
                for zz in np.arange(z0, z1, 0.8):
                    d.line([SP(uu, zz), SP(uu, zz + 0.4)], fill=inkl, width=S)
    zt_all = band_z(NB - 1)[1]
    if p > 0.45:
        hz = band_z(2)[1] + 0.6
        d.line([SP(13, hz), SP(D - 13, hz)], fill=ink, width=2 * S)
    if p > 0.6:
        cxu = D / 2
        r = 9.5
        pts = [SP(cxu + r * math.cos(a), band_z(2)[1] + 2.2 + r * 0.62 * math.sin(a)) for a in np.linspace(0, math.pi, 30)]
        d.line(pts, fill=red, width=2 * S)
    if p > 0.6:
        d.rectangle([SP(-1.2, zt_all + 6)[0], SP(-1.2, zt_all + 6)[1], SP(1.2, 0)[0], SP(1.2, 0)[1]], outline=ink,
                    width=S)
    if p > 0.66:
        f = min(1.0, (p - 0.66) / 0.12)
        top = zt_all + 7.4 + 21 * f
        d.line([SP(-1.2, zt_all + 7.4), SP(0, top)], fill=ink, width=S)
        d.line([SP(1.2, zt_all + 7.4), SP(0, top)], fill=ink, width=S)
        if p > 0.79:
            X, Y = SP(0, top + 2.4)
            rr = 2.2 * sc2
            d.ellipse([X - rr, Y - rr, X + rr, Y + rr], outline=red, width=S)
    for k in range(NB):
        z = band_z(k)[1]
        if p > t_band(k) + 0.06:
            d.text(((x1 - 52) * S, SP(0, z)[1] - 8 * S), ("+%.2f" % z).replace(".", ","), font=font(FR, 11 * S),
                   fill=ink)
    _label(d, S, (x0 + x1) / 2, y1 - 26, "SECCIÓN A-A'")

    # ---------------- AXONOMETRÍA
    x0, y0, x1, y1 = boxes[2]
    c30, s30 = math.cos(math.radians(30)), math.sin(math.radians(30))
    sc3 = 3.0 * S
    ox, oy = (x0 + x1) / 2 * S, ((y0 + y1) / 2 + 40) * S
    pts = list(F.exterior.coords)[:-1]
    cxm = np.mean([q[0] for q in pts])
    cym = np.mean([q[1] for q in pts])

    def AP(x, y, z):
        X = (x - cxm) * c30 - (y - cym) * c30
        Y = (x - cxm) * s30 + (y - cym) * s30 + z
        return (ox + X * sc3, oy - Y * sc3 + 20 * S)

    hcur = 0.0
    for k in range(NB):
        f = min(1.0, max(0.0, (p - t_band(k)) / 0.06))
        if f > 0:
            hcur = band_z(k)[0] + (band_z(k)[1] - band_z(k)[0]) * f
    d.line([AP(*q, 0) for q in pts] + [AP(*pts[0], 0)], fill=inkl, width=S)
    if hcur > 0:
        top = [AP(*q, hcur) for q in pts]
        d.polygon(top, fill=(196, 160, 120, 70), outline=ink)
        for q in pts:
            d.line([AP(*q, 0), AP(*q, hcur)], fill=ink, width=S)
        for k in range(NB):
            z = band_z(k)[1]
            if z < hcur - 0.1:
                d.line([AP(*q, z) for q in pts] + [AP(*pts[0], z)], fill=inkl, width=S)
        hp = list(hall.exterior.coords)
        d.line([AP(*q, hcur) for q in hp], fill=ink, width=S)
    if p > 0.62:
        dc = geo["dome"]
        r = 9.5
        zb = band_z(2)[1] + 2.2
        ring = [AP(dc[0] + r * math.cos(a), dc[1] + r * math.sin(a), max(zb, hcur * 0 + zb)) for a in np.linspace(0, 2 * math.pi, 40)]
        d.line(ring, fill=red, width=S)
        for a in np.linspace(0, 2 * math.pi, 9)[:-1]:
            arc = [AP(dc[0] + r * math.cos(ph) * math.cos(a), dc[1] + r * math.cos(ph) * math.sin(a), zb + r * 0.62 * math.sin(ph))
                   for ph in np.linspace(0, math.pi / 2, 10)]
            d.line(arc, fill=red, width=S)
    if p > 0.66:
        fm = geo["fm"]
        f = min(1.0, (p - 0.66) / 0.12)
        zb = band_z(NB - 1)[1] + 7.4
        d.line([AP(fm[0], fm[1], zb), AP(fm[0], fm[1], zb + 21 * f)], fill=ink, width=2 * S)
    _label(d, S, (x0 + x1) / 2, y1 - 26, "AXONOMETRÍA")
    im = im.resize((VW, VH - PANEL_Y), Image.LANCZOS)
    return np.asarray(im, np.float32) / 255


def _label(d, S, xc, y, txt):
    f = font(FB, 14 * S)
    w = d.textbbox((0, 0), txt, font=f)[2]
    d.text((xc * S - w / 2, y * S), txt, font=f, fill=INKc + (230,))


PHASES = [(0.00, "Replanteo y excavación del sótano"), (0.09, "Cimientos y muros de sótano"),
          (0.20, "Estructura de acero y muros de sillería, planta a planta"),
          (0.58, "Cubiertas, cúpula de vidriera y torre"), (0.72, "Ornamentación y desmontaje de andamios"),
          (0.92, "Inauguración del Palacio, 1923")]


def chrome(p, t):
    """Static frame: header, viewport rules, phase label and the 1915-1922 rule."""
    im = Image.new("RGBA", (VW, VH), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    f1, f2 = font(FB, 50), font(FI, 28)
    title = "PALACIO DE COMUNICACIONES"
    w = d.textbbox((0, 0), title, font=f1)[2]
    d.text(((VW - w) / 2, 40), title, font=f1, fill=RED + (240,))
    sub = "Correos y Telégrafos · València · M. Á. Navarro Pérez · 1915-1922"
    w = d.textbbox((0, 0), sub, font=f2)[2]
    d.text(((VW - w) / 2, 108), sub, font=f2, fill=INKc + (230,))
    x0, y0, x1, y1 = VP
    d.rectangle([x0 + 10, y0 - 6, x1 - 10, y1 + 6], outline=INKc + (220,), width=3)
    d.rectangle([x0 + 16, y0, x1 - 16, y1], outline=INKc + (120,), width=1)
    # phase caption and progress rule in the panel foot
    if t < 3.0 or t > 37.0:
        label = "Plano de emplazamiento"
        a = 1.0
    else:
        k = max(i for i, (p0, _) in enumerate(PHASES) if p >= p0)
        label = PHASES[k][1]
        dd = min([abs(p - PHASES[k][0]) if k else 1.0] + ([abs(PHASES[k + 1][0] - p)] if k + 1 < len(PHASES) else [1.0]))
        a = min(1.0, dd / 0.01)
    f3 = font(FI, 34)
    w = d.textbbox((0, 0), label, font=f3)[2]
    y = VH - 118
    d.text(((VW - w) / 2, y), label, font=f3, fill=INKc + (int(235 * a),))
    pr = p if 3.0 <= t <= 37.0 else (0.0 if t < 3.0 else 1.0)
    bx0, bx1, by = VW / 2 - 300, VW / 2 + 300, VH - 52
    d.line([(bx0, by), (bx1, by)], fill=INKc + (90,), width=2)
    d.line([(bx0, by), (bx0 + (bx1 - bx0) * pr, by)], fill=RED + (230,), width=5)
    f4 = font(FR, 26)
    d.text((bx0 - 78, by - 16), "1915", font=f4, fill=INKc + (220,))
    d.text((bx1 + 14, by - 16), "1922", font=f4, fill=INKc + (220,))
    return np.asarray(im, np.float32) / 255


# --------------------------------------------------------------- 3D frame
def read_exr(path):
    f = OpenEXR.File(path, separate_channels=True)
    out = {}
    for part in f.parts:
        for k, v in part.channels.items():
            if "Noisy" not in k:
                out[k] = np.asarray(v.pixels, np.float32)
    return out


def get(ch, name):
    for k, v in ch.items():
        if k.endswith(name):
            return v
    raise KeyError(name)


def tonemap(x):
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    y = np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)
    return np.where(y <= 0.0031308, 12.92 * y, 1.055 * np.power(y, 1 / 2.4) - 0.055)


def ray_dirs(meta, W, H):
    Mw = np.array(meta["cam_world"])
    th = 18.0 / meta["lens"]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    x = (xx + 0.5 - W / 2) / (W / 2) * th
    y = (H / 2 - yy - 0.5) / (W / 2) * th
    dcam = np.stack([x, y, -np.ones_like(x)], -1)
    dw = dcam @ Mw[:3, :3].T
    return dw / np.linalg.norm(dw, axis=-1, keepdims=True)


def project(meta, pts, W, H):
    Mi = np.array(meta["cam"])
    th = 18.0 / meta["lens"]
    P = np.c_[np.asarray(pts, float), np.ones(len(pts))] @ Mi.T
    z = -P[:, 2]
    x = W / 2 + P[:, 0] / np.maximum(z, 1e-3) / th * (W / 2)
    y = H / 2 - P[:, 1] / np.maximum(z, 1e-3) / th * (W / 2)
    return np.c_[x, y], z


def ink_lines(depth, nrm):
    z = np.log(np.clip(depth, 0.1, 5000))
    lap = ndi.laplace(z)
    e_d = sstep(0.04, 0.25, -lap)
    gn = 0
    for c in range(3):
        n = ndi.gaussian_filter(nrm[..., c], 0.8)
        gn = gn + np.abs(np.roll(n, -1, 0) - n) + np.abs(np.roll(n, -1, 1) - n)
    e_n = sstep(0.3, 0.8, gn)
    e = np.maximum(e_d, e_n * 0.7)
    dens = ndi.uniform_filter(e, 9)
    return np.clip(e * (1 - sstep(0.22, 0.45, dens) * 0.7), 0, 1)


def frame3d(meta, path):
    ch = read_exr(path)
    rgb = np.dstack([get(ch, "Combined.R"), get(ch, "Combined.G"), get(ch, "Combined.B")])
    depth = get(ch, "Depth.Z")
    nrm = np.dstack([get(ch, "Normal.X"), get(ch, "Normal.Y"), get(ch, "Normal.Z")])
    H, W = depth.shape
    t = meta["t"]
    img = tonemap(rgb * 1.12)
    skym = depth > 2500
    dirs = ray_dirs(meta, W, H)
    el = np.degrees(np.arcsin(np.clip(dirs[..., 2], -1, 1)))
    az = np.degrees(np.arctan2(dirs[..., 1], dirs[..., 0])) % 360
    pano = sky_pano()
    ph, pw = pano.shape[:2]
    rr = np.clip((90 - np.maximum(el, 0)) / 90 * (ph - 1), 0, ph - 1)
    cc = ((az + t * 0.25) % 360) / 360 * (pw - 1)
    sky = np.stack([ndi.map_coordinates(pano[..., c], [rr, cc], order=1, mode="wrap") for c in range(3)], -1)
    img = np.where(skym[..., None], sky, img)
    # aerial perspective: warm haze with distance, towards the horizon colour
    dist = np.where(skym, 3000, depth)
    hz = (1 - np.exp(-dist / 900.0)) * 0.42
    haze = np.array([0.90, 0.83, 0.74])
    img = np.where(skym[..., None], img, img * (1 - hz[..., None]) + haze * hz[..., None])
    # grade: warm lights / cool shade, gentle contrast
    lum = img.mean(2)
    sh = 1 - sstep(0.08, 0.5, lum)
    img = img + sh[..., None] * np.array([-0.015, 0.0, 0.025]) + (1 - sh[..., None]) * np.array([0.02, 0.006, -0.018])
    # sun shafts and suspended dust (strongest at eye level)
    low = meta["cam_world"][2][3] < 30
    sun = np.array(meta["sun"])
    pts, zz = project(meta, [np.array(meta["cam_world"])[:3, 3] + sun * 1000], W, H)
    lit = sstep(0.6, 0.9, lum) * (~skym)
    small = ndi.zoom(lit, 0.25, order=1)
    sxy = pts[0]
    vd = np.array([W / 2 - sxy[0], H / 2 - sxy[1]])
    vd = vd / (np.linalg.norm(vd) + 1e-6)
    acc = np.zeros_like(small)
    for k in range(18):
        tt = k / 18
        acc += (1 - tt) ** 1.5 * ndi.shift(small, (vd[1] * tt * 40, vd[0] * tt * 40), order=1, mode="constant")
    acc /= 9.0
    shaft = ndi.zoom(acc, (H / acc.shape[0], W / acc.shape[1]), order=1)[:H, :W]
    img = 1 - (1 - img) * (1 - (0.22 if low else 0.1) * shaft[..., None] * np.array([1.0, 0.85, 0.6]))
    if low:
        rng = np.random.default_rng(7)
        Pp = rng.random((700, 6))
        x = (Pp[:, 0] * W + 9 * t * (0.3 + Pp[:, 2]) + 14 * np.sin(t * (0.3 + Pp[:, 3]) + Pp[:, 4] * 6)) % W
        y = (Pp[:, 1] * H - 4 * t * (0.2 + Pp[:, 3]) + 10 * np.cos(t * (0.25 + Pp[:, 2]) + Pp[:, 5] * 6)) % H
        xi, yi = x.astype(int), y.astype(int)
        vis = (0.3 + shaft[yi, xi]) * (0.4 + 0.6 * Pp[:, 5]) * (0.5 + 0.5 * np.sin(t * 2.0 + Pp[:, 4] * 30))
        layer = np.zeros((H, W), np.float32)
        np.add.at(layer, (yi, xi), vis * 1.6)
        layer = ndi.gaussian_filter(layer, 0.9 + 0.6 * (Pp[:, 2].mean()))
        img = 1 - (1 - img) * (1 - np.clip(layer, 0, 1)[..., None] * np.array([1.0, 0.92, 0.75]))
    # work dust puffs
    if meta.get("dust"):
        dp = np.array(meta["dust"], float)
        xy, zc = project(meta, dp[:, :3], W, H)
        layer = np.zeros((H, W), np.float32)
        rng = np.random.default_rng(meta["frame"] // 3)
        for (px, py), z_, s_ in zip(xy, zc, dp[:, 3]):
            if z_ <= 1:
                continue
            for k in range(3):
                X = int(px + rng.normal(0, 120 / z_))
                Y = int(py - abs(rng.normal(0, 160 / z_)))
                if 0 <= X < W and 0 <= Y < H:
                    layer[Y, X] += s_ * 30 / max(z_, 5)
        layer = np.clip(ndi.gaussian_filter(layer, 6) * 120, 0, 0.4)
        img = img * (1 - layer[..., None]) + np.array([0.86, 0.78, 0.66]) * layer[..., None]
    # bloom, vignette, grain
    hl = np.clip(img - 0.8, 0, 1)
    img = img + 0.35 * ndi.gaussian_filter(hl, (12, 12, 0))
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    img = img * (1 - 0.22 * sstep(0.55, 1.45, r))[..., None]
    img = img + (np.random.default_rng(meta["frame"]).random((H, W, 1)) - 0.5) * 0.025
    return np.clip(img, 0, 1), ink_lines(depth, nrm)


def load_map(i):
    path = os.path.join(HERE, "film", "map", "m%04d.png" % max(0, min(i, int(T_MAP * FPS))))
    im = Image.open(path).convert("RGB")
    return np.asarray(im, np.float32) / 255


def compose(fi):
    t = fi / FPS
    p = p_of_t(t)
    vw, vh = VP[2] - VP[0], VP[3] - VP[1]
    meta_path = os.path.join(HERE, "film", "exr", "f%04d.json" % fi)
    exr = os.path.join(HERE, "film", "exr", "f%04d.exr" % fi)
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else None
    have3d = meta is not None and meta.get("rendered") and os.path.exists(exr)
    pap = paper(900, 1000, 5)
    if have3d:
        img3, lines = frame3d(meta, exr)
        Hh, Ww = lines.shape
        rv = CACHE.setdefault("reveal", (0.6 * fbm(Hh, Ww, 7, 6, 3) +
                                         0.4 * np.clip(np.sqrt(((np.mgrid[0:Hh, 0:Ww][1] - Ww / 2) / Ww) ** 2 +
                                                               ((np.mgrid[0:Hh, 0:Ww][0] - Hh / 2) / Hh) ** 2) * 1.6, 0, 1)))
        col_on = smooth((t - 2.9) / 1.4) * (1 - smooth((t - 36.3) / 1.0))
        ink_on = smooth((t - 2.4) / 0.6) * (1 - smooth((t - 37.0) / 0.6))
        thr = col_on * 1.25 - 0.08
        cm = 1 - sstep(thr - 0.035, thr, rv)
        rim = np.exp(-((rv - thr) / 0.012) ** 2) * (0 < col_on < 1)
        drawing = pap * (1 - (lines * ink_on * 0.9)[..., None]) + INK * (lines * ink_on * 0.9)[..., None]
        view = drawing * (1 - cm[..., None]) + img3 * cm[..., None]
        view = view * (1 - 0.15 * rim[..., None] * np.array([0.3, 0.5, 0.7]))
        lw = lines * (0.35 * (1 - col_on)) * cm
        view = view * (1 - lw[..., None]) + INK * lw[..., None]
    else:
        view = pap.copy()
    # the folded plan at the head and tail of the loop
    if t < 3.1:
        mp = load_map(int(round(t * FPS)))
        a = 1 - smooth((t - 2.4) / 0.6) if have3d else 1.0
        view = mp * a + view * (1 - a)
    elif t > 36.9:
        mp = load_map(int(round((DUR - t) * FPS)))
        a = smooth((t - 37.0) / 0.6) if have3d else 1.0
        view = mp * a + view * (1 - a)
    vimg = Image.fromarray((np.clip(view, 0, 1) * 255).astype(np.uint8)).resize((vw - 32, vh), Image.LANCZOS)
    vimg = vimg.filter(ImageFilter.UnsharpMask(radius=1.2, percent=60, threshold=2))
    out = paper(VW, VH, 9).copy()
    out[VP[1]:VP[3], VP[0] + 16:VP[2] - 16] = np.asarray(vimg, np.float32) / 255
    pan = draw_panel(p if 3.0 <= t <= 37.0 else (0.0 if t < 3.0 else 1.0), t)
    k = pan[..., 3:4]
    out[PANEL_Y:] = out[PANEL_Y:] * (1 - k) + pan[..., :3] * k
    ch = chrome(p, t)
    k = ch[..., 3:4]
    out = out * (1 - k) + ch[..., :3] * k
    out = np.clip(out, 0, 1)
    return (out * 255 + np.random.default_rng(fi).random(out.shape) * 0.9).astype(np.uint8)


def work(fi):
    dst = os.path.join(HERE, "film", "frames", "v%04d.png" % fi)
    if os.path.exists(dst):
        return fi
    t = fi / FPS
    if 3.0 <= t <= 37.0:
        exr = os.path.join(HERE, "film", "exr", "f%04d.exr" % fi)
        if not os.path.exists(exr) or (os.path.getmtime(exr) > __import__("time").time() - 10):
            return None
    if t < 3.1 or t > 36.9:
        if not os.path.exists(os.path.join(HERE, "film", "map", "m%04d.png" % min(72, int(round(min(t, DUR - t) * FPS))))):
            return None
    Image.fromarray(compose(fi)).save(dst, compress_level=1)
    return fi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--frames", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.join(HERE, "film", "frames"), exist_ok=True)
    sky_pano()
    frames = range(N) if not a.frames else [int(x) for x in a.frames.split(",")]
    if a.force:
        for fi in frames:
            pth = os.path.join(HERE, "film", "frames", "v%04d.png" % fi)
            if os.path.exists(pth):
                os.remove(pth)
    with Pool(a.jobs) as pool:
        done = [f for f in pool.imap_unordered(work, frames) if f is not None]
    print("composed", len(done))


if __name__ == "__main__":
    main()
