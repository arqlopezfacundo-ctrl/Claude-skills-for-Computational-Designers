"""Painterly compositing of the Cycles passes into the final frames.

    python post.py --src render/exr --dst render/frames [--jobs 4]

Per frame: filmic tone curve and warm/cool grade, aerial haze from depth,
ink linework from depth/normal discontinuities, a light Kuwahara + pigment
edge darkening, sun shafts with suspended dust, work-front dust puffs,
oil-painted halftone clouds, watercolour paper.  The loop opens and closes
on Goerlich's 1927 plan drawn in ink on paper: linework draws on, colour
washes in, construction runs, colour recedes, back to the plan."""
import argparse
import json
import math
import os
import sys
from multiprocessing import Pool

import numpy as np
import OpenEXR
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from vlc import plan  # noqa: E402

FPS, DURATION = 24, 20.0
N = int(FPS * DURATION)
INK = np.array([0.24, 0.17, 0.12])
RED_INK = np.array([0.62, 0.20, 0.14])
PAPER = np.array([0.935, 0.895, 0.815])
CACHE = {}


# ---------------------------------------------------------------- utilities
def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def sstep(a, b, x):
    t = min(max((x - a) / (b - a), 0.0), 1.0)
    return t * t * (3 - 2 * t)


def fbm(h, w, seed, octaves=6, base=4, persistence=0.55):
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        n = base * 2 ** o
        g = rng.random((n + 3, int(n * w / h) + 3)).astype(np.float32)
        z = ndi.zoom(g, (h / n, h / n), order=3)[:h, :w]
        if z.shape != (h, w):
            z = np.pad(z, ((0, h - z.shape[0]), (0, w - z.shape[1])), mode="edge")
        out += amp * z
        tot += amp
        amp *= persistence
    out /= tot
    out = (out - out.min()) / (out.max() - out.min() + 1e-6)
    return out


def read_exr(path):
    f = OpenEXR.File(path, separate_channels=True)
    out = {}
    for part in f.parts:
        for k, v in part.channels.items():
            if "Noisy" in k:
                continue
            out[k] = np.asarray(v.pixels, np.float32)
    return out


def get(ch, name):
    for k, v in ch.items():
        if k.endswith(name):
            return v
    raise KeyError(name + " in " + ", ".join(ch))


def tonemap(x):
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    y = np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)
    return np.where(y <= 0.0031308, 12.92 * y, 1.055 * np.power(y, 1 / 2.4) - 0.055)


def project(meta, pts):
    """World points (n,3) -> pixel coords (n,2) for the ortho camera."""
    Mi = np.array(meta["cam"])
    W, H = meta["res"]
    P = np.c_[np.asarray(pts, float), np.ones(len(pts))] @ Mi.T
    s = (W / 2) / (meta["ortho"] / 2)
    return np.c_[W / 2 + P[:, 0] * s, H / 2 - P[:, 1] * s]


def proj_dir(meta, v):
    Mi = np.array(meta["cam"])[:3, :3]
    d = Mi @ np.asarray(v)
    d = np.array([d[0], -d[1]])
    return d / (np.linalg.norm(d) + 1e-9)


# ------------------------------------------------------------ static layers
def statics(W, H):
    key = (W, H)
    if key in CACHE:
        return CACHE[key]
    S = {}
    n1 = fbm(H, W, 1, 7, 3)
    n2 = fbm(H, W, 2, 4, 40, 0.6)
    fib = ndi.gaussian_filter(np.random.default_rng(3).random((H, W)).astype(np.float32), (0.6, 3.0))
    fib = (fib - fib.mean()) / (fib.std() + 1e-6)
    paper = PAPER[None, None, :] * (0.955 + 0.06 * n1[..., None] + 0.018 * (n2[..., None] - 0.5)
                                     + 0.008 * fib[..., None])
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    paper *= (1 - 0.10 * smoothstep(0.6, 1.5, r))[..., None]
    S["paper"] = paper.astype(np.float32)
    S["grain"] = (n2 - 0.5).astype(np.float32)
    # reveal field: organic wash spreading from the island outwards
    S["reveal"] = (0.62 * fbm(H, W, 7, 6, 3) + 0.38 * np.clip(r / 1.3, 0, 1)).astype(np.float32)
    S["inkfield"] = (0.5 * fbm(H, W, 8, 5, 6) + 0.5 * np.clip(r / 1.3, 0, 1)).astype(np.float32)
    # torn watercolour border
    edge = np.minimum.reduce([xx, W - 1 - xx, yy, H - 1 - yy]) / (0.5 * H)
    S["border"] = smoothstep(0.02, 0.16, edge + 0.10 * (fbm(H, W, 9, 6, 6) - 0.5)).astype(np.float32)
    # painted clouds (two layers, wider than the frame so they can drift)
    S["clouds"] = [paint_clouds(H, int(W * 1.6), seed) for seed in (21, 22)]
    S["dust"] = np.random.default_rng(5).random((900, 6)).astype(np.float32)
    CACHE[key] = S
    return S


def paint_clouds(H, W, seed):
    """Oil-painted cumulus with halftone shadow dots, RGBA float."""
    d = fbm(H, W, seed, 7, 2, 0.58)
    warp = fbm(H, W, seed + 100, 4, 3)
    sh = (warp - 0.5) * 40
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = ndi.map_coordinates(d, [np.clip(yy + sh, 0, H - 1), np.clip(xx + sh * 0.7, 0, W - 1)], order=1)
    # clouds hug the top band and the side edges of the frame
    band = smoothstep(0.55, 0.0, yy / H) * 1.0 + 0.3 * smoothstep(0.8, 1.0, yy / H)
    dens = smoothstep(0.42, 0.62, d * (0.45 + 0.75 * band))
    dens = ndi.gaussian_filter(dens, 2.0)
    gy, gx = np.gradient(ndi.gaussian_filter(dens, 9))
    nrm = np.sqrt(gx ** 2 + gy ** 2) + 1e-4
    light = np.clip(0.5 + 0.5 * ((-gx * -0.6 + -gy * -0.8) / nrm) * np.clip(nrm * 60, 0, 1), 0, 1)
    light = light * 0.6 + 0.4 * ndi.gaussian_filter(dens, 14) / (dens.max() + 1e-4)
    strokes = ndi.gaussian_filter(np.random.default_rng(seed).random((H, W)).astype(np.float32),
                                  (1.0, 7.0))
    strokes = (strokes - strokes.mean()) / (strokes.std() + 1e-6)
    hi = np.array([1.0, 0.95, 0.86])
    mid = np.array([0.93, 0.86, 0.80])
    lo = np.array([0.66, 0.70, 0.82])
    col = lo[None, None] + (mid - lo)[None, None] * smoothstep(0.0, 0.55, light)[..., None]
    col = col + (hi - mid)[None, None] * smoothstep(0.55, 1.0, light)[..., None]
    col *= (1 + 0.035 * strokes[..., None])
    # halftone dots in the shade (screen-print look of the reference)
    f = 0.55
    dots = (np.sin(xx * f) * np.sin(yy * f) * 0.5 + 0.5)
    shade = 1 - smoothstep(0.15, 0.6, light)
    ht = (dots < shade * 0.65).astype(np.float32)
    col = col * (1 - 0.10 * ht[..., None])
    edge_ht = (dots < (1 - smoothstep(0.05, 0.4, dens)) * 0.9).astype(np.float32)
    alpha = dens * (1 - 0.6 * edge_ht * (dens < 0.5))
    return np.dstack([col, alpha]).astype(np.float32)


# ------------------------------------------------------------- plan drawing
def plan_layer(meta, W, H, draw_frac=1.0):
    """Goerlich's 1927 plan in ink, projected with the current camera."""
    im = Image.new("L", (W, H), 0)
    red = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(im)
    dr = ImageDraw.Draw(red)
    sc = W / 1920

    def poly(pts, z, width, dd=d, closed=True):
        pp = project(meta, [(x, y, z) for x, y in pts])
        seq = [tuple(p) for p in pp]
        if closed:
            seq.append(seq[0])
        dd.line(seq, fill=255, width=max(1, int(round(width * sc))), joint="curve")

    from shapely.geometry import Point, Polygon
    poly(list(plan.ISLAND.exterior.coords), 0.0, 2.6)
    poly(list(plan.ISLAND.buffer(-1.0).exterior.coords), 0.0, 1.2)
    poly(list(Point(plan.PATIO_C).buffer(plan.PATIO_R, 48).exterior.coords), 0.0, 1.8)
    poly(list(Point(plan.PATIO_C).buffer(plan.PATIO_R + 1.6, 48).exterior.coords), 0.0, 1.0)
    for c, a, b in plan.parterres():
        poly(plan.ellipse(c, a, b, 40), 0.0, 1.6)
        poly(plan.ellipse(c, a - 1.2, b - 1.0, 40), 0.0, 1.0)
    poly(list(Point(plan.C2[0] + 1.5, plan.C2[1]).buffer(3.4, 32).exterior.coords), 0.0, 1.4)
    for blk in plan.BLOCKS:
        if blk.distance(Point(10, 0)) > 260:
            continue
        poly(list(blk.exterior.coords), 0.0, 1.5)
        f = plan.footprint(blk)
        if f is not None:
            poly(list(f.exterior.coords), 0.0, 0.9)
    # tram tracks and the axis in red pencil
    rails = plan.ISLAND.buffer(8.5, join_style=1).exterior
    poly(list(rails.coords), 0.0, 1.0, dd=dr)
    poly([(plan.X_MIN - 30, plan.axis_y(plan.X_MIN)), (plan.X_MAX + 30, plan.axis_y(plan.X_MAX))],
         0.0, 0.8, dd=dr, closed=False)
    a = np.asarray(im, np.float32) / 255
    b = np.asarray(red, np.float32) / 255
    a = ndi.gaussian_filter(a, 0.5)
    b = ndi.gaussian_filter(b, 0.5)
    return a, b


def cartouche(W, H):
    if ("cart", W) in CACHE:
        return CACHE[("cart", W)]
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    s = W / 1920
    try:
        f1 = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf", int(26 * s))
        f2 = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf", int(19 * s))
        f3 = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf", int(15 * s))
    except OSError:
        f1 = f2 = f3 = ImageFont.load_default()
    x, y = int(70 * s), int(H - 150 * s)
    red = (158, 52, 36, 235)
    ink = (62, 44, 30, 225)
    tw = max(d.textbbox((0, 0), "URBANIZACIÓN DE LA PLAZA DE EMILIO CASTELAR", font=f1)[2],
             d.textbbox((0, 0), "Plataforma central y mercado de flores — J. Goerlich Lleó, 1927",
                        font=f2)[2]) + 14 * s
    d.rectangle([x - 22 * s, y - 20 * s, x + tw + 22 * s, y + 104 * s], fill=(238, 228, 208, 235))
    d.rectangle([x - 16 * s, y - 14 * s, x + tw + 16 * s, y + 98 * s], outline=red,
                width=max(1, int(2 * s)))
    d.rectangle([x - 10 * s, y - 8 * s, x + tw + 10 * s, y + 92 * s], outline=red, width=1)
    d.text((x, y), "URBANIZACIÓN DE LA PLAZA DE EMILIO CASTELAR", font=f1, fill=red)
    d.text((x, y + 36 * s), "Plataforma central y mercado de flores — J. Goerlich Lleó, 1927",
           font=f2, fill=ink)
    d.text((x, y + 64 * s), "VALÈNCIA  ·  hoy Plaza del Ayuntamiento  ·  Plano D2", font=f3, fill=ink)
    arr = np.asarray(im, np.float32) / 255
    CACHE[("cart", W)] = arr
    return arr


# ------------------------------------------------------------------ passes
def ink_lines(depth, nrm, W):
    z = depth
    lap = ndi.laplace(z)
    # depth steps: only the near side of a silhouette gets the line
    e_d = smoothstep(0.25, 1.2, -lap) * 0.9
    gn = 0
    for c in range(3):
        n = ndi.gaussian_filter(nrm[..., c], 1.0)
        gn = gn + np.abs(np.roll(n, -1, 0) - n) + np.abs(np.roll(n, -1, 1) - n)
    e_n = smoothstep(0.30, 0.8, gn)
    e = np.maximum(e_d, e_n * 0.75)
    # foliage and tiny clutter make blots: damp lines where edges are dense
    dens = ndi.uniform_filter(e, 9)
    e = e * (1 - smoothstep(0.22, 0.45, dens) * 0.75)
    return np.clip(e, 0, 1)


def kuwahara(img, r=2):
    """Fast 4-quadrant Kuwahara on luminance variance (painterly flattening)."""
    k = r + 1
    lum = img.mean(2)
    m = [None] * 4
    v = [None] * 4
    offs = [(-r // 2 - 0, -r // 2), (-r // 2, r // 2), (r // 2, -r // 2), (r // 2, r // 2)]
    for i, (dy, dx) in enumerate(offs):
        mu = ndi.uniform_filter(img, (k, k, 1))
        mu = np.roll(mu, (dy, dx), (0, 1))
        l1 = np.roll(ndi.uniform_filter(lum, k), (dy, dx), (0, 1))
        l2 = np.roll(ndi.uniform_filter(lum * lum, k), (dy, dx), (0, 1))
        m[i] = mu
        v[i] = l2 - l1 * l1
    v = np.stack(v)
    idx = np.argmin(v, 0)
    out = np.zeros_like(img)
    for i in range(4):
        out += m[i] * (idx == i)[..., None]
    return out


def shafts(lit, sdir, steps=26, length=150):
    """Directional blur of sunlit areas away from the sun (god rays)."""
    acc = np.zeros_like(lit)
    tot = 0
    for k in range(steps):
        t = k / steps
        w = (1 - t) ** 1.6
        dy, dx = -sdir[1] * t * length, -sdir[0] * t * length
        acc += w * ndi.shift(lit, (dy, dx), order=1, mode="constant")
        tot += w
    return acc / tot


# ------------------------------------------------------------------- frame
def phase_curves(t):
    ink_on = sstep(0.30, 1.35, t) * (1 - sstep(18.85, 19.45, t))
    col_on = sstep(1.05, 2.85, t) * (1 - sstep(17.30, 19.05, t))
    return ink_on, col_on


def compose(meta, src):
    W, H = meta["res"]
    S = statics(W, H)
    t = meta["t"]
    fi = meta["frame"]
    ink_on, col_on = phase_curves(t)
    pa, pr = plan_layer(meta, W, H)
    paper = S["paper"].copy()

    if meta.get("rendered") and os.path.exists(src):
        ch = read_exr(src)
        rgb = np.dstack([get(ch, "Combined.R"), get(ch, "Combined.G"), get(ch, "Combined.B")])
        depth = get(ch, "Depth.Z")
        nrm = np.dstack([get(ch, "Normal.X"), get(ch, "Normal.Y"), get(ch, "Normal.Z")])
        sdir = proj_dir(meta, meta["sun"])
        # --- grade
        lum = rgb.mean(2, keepdims=True)
        rgb = rgb * 1.18
        img = tonemap(rgb)
        shadow = 1 - smoothstep(0.08, 0.45, img.mean(2))
        img = img + shadow[..., None] * np.array([-0.012, 0.0, 0.03])     # cool shade
        img = img + (1 - shadow[..., None]) * np.array([0.02, 0.008, -0.02])  # warm light
        # --- aerial haze from depth
        zc = np.median(depth)
        hz = smoothstep(zc - 25, zc + 140, depth) * 0.30
        hz = hz * (0.8 + 0.4 * S["grain"] + 0.2)
        haze_col = np.array([0.90, 0.83, 0.72])
        img = img * (1 - hz[..., None]) + haze_col * hz[..., None]
        # --- painterly flattening + pigment edge darkening
        kw = kuwahara(img, 2)
        img = 0.55 * img + 0.45 * kw
        bl = ndi.gaussian_filter(img, (3, 3, 0))
        img = np.clip(img - 0.35 * np.clip(bl - img, 0, 1) * 0 + 0.22 * (img - bl), 0, 1)
        # --- sun shafts and suspended dust
        lit = smoothstep(0.55, 0.85, img.mean(2))
        lit_small = ndi.zoom(lit, 0.25, order=1)
        sh = shafts(lit_small, sdir, steps=22, length=60)
        sh = ndi.zoom(sh, (H / sh.shape[0], W / sh.shape[1]), order=1)[:H, :W]
        glow = np.array([1.0, 0.82, 0.55])
        img = 1 - (1 - img) * (1 - 0.16 * sh[..., None] * glow)
        img = add_dust(img, S, meta, sh, t)
        img = add_puffs(img, meta, t, W, H)
        # --- clouds
        img = add_clouds(img, S, t, W, H)
        # --- linework
        e = ink_lines(depth, nrm, W)
        line_layer = e
    else:
        img = paper.copy()
        line_layer = np.zeros((H, W), np.float32)

    # --- watercolour paper texture over the painting, torn border
    img = img * (0.94 + 0.10 * (S["paper"] / PAPER[None, None] - 0.95))
    img = img * (1 + 0.025 * S["grain"][..., None])
    border = S["border"]
    # --- colour reveal mask with pigment rim
    rv = S["reveal"]
    thr = col_on * 1.25 - 0.08
    cm = smoothstep(thr - 0.035, thr, rv)
    cm = 1 - cm
    rim = np.exp(-((rv - thr) / 0.012) ** 2) * (col_on > 0) * (col_on < 1)
    cm = cm * border
    out = paper * (1 - cm[..., None]) + img * cm[..., None]
    out = out * (1 - 0.18 * rim[..., None] * np.array([0.3, 0.5, 0.7]))
    # --- ink linework draw-on
    im_thr = ink_on * 1.25 - 0.08
    im = (1 - smoothstep(im_thr - 0.03, im_thr, S["inkfield"])) * border
    lw = line_layer * im * (0.9 - 0.62 * col_on)
    out = out * (1 - lw[..., None]) + INK * lw[..., None]
    # --- the 1927 plan, always beneath; strong when only paper shows
    pk = 0.85 * (1 - col_on) + 0.10
    out = out * (1 - (pa * pk)[..., None]) + INK * (pa * pk)[..., None]
    rk = 0.7 * (1 - col_on) + 0.06
    out = out * (1 - (pr * rk)[..., None]) + RED_INK * (pr * rk)[..., None]
    # --- cartouche
    ca = cartouche(W, H)
    k = ca[..., 3:4] * 0.92
    out = out * (1 - k) + ca[..., :3] * k
    # vignette
    out = np.clip(out, 0, 1)
    return (out * 255 + np.random.default_rng(fi).random(out.shape) * 0.9).astype(np.uint8)


def add_dust(img, S, meta, sh, t):
    W, H = meta["res"]
    P = S["dust"]
    x = (P[:, 0] * W + 14 * t * (0.4 + P[:, 2]) + 18 * np.sin(t * (0.3 + P[:, 3]) + P[:, 4] * 6)) % W
    y = (P[:, 1] * H - 6 * t * (0.2 + P[:, 3]) + 12 * np.cos(t * (0.25 + P[:, 2]) + P[:, 5] * 6)) % H
    xi, yi = x.astype(int), y.astype(int)
    vis = sh[yi, xi] * (0.4 + 0.6 * P[:, 5]) * (0.5 + 0.5 * np.sin(t * 2.0 + P[:, 4] * 30))
    layer = np.zeros((H, W), np.float32)
    np.add.at(layer, (yi, xi), vis * 2.2)
    big = P[:, 2] > 0.8
    np.add.at(layer, (np.clip(yi[big] + 1, 0, H - 1), xi[big]), vis[big] * 1.5)
    layer = ndi.gaussian_filter(layer, 0.8)
    c = np.array([1.0, 0.9, 0.7])
    return 1 - (1 - img) * (1 - np.clip(layer, 0, 1)[..., None] * c)


def add_puffs(img, meta, t, W, H):
    if not meta.get("dust"):
        return img
    pts = np.array(meta["dust"], float)
    xy = project(meta, pts[:, :3])
    layer = np.zeros((H, W), np.float32)
    rng = np.random.default_rng(meta["frame"] // 3)
    for (x, y), s in zip(xy, pts[:, 3]):
        for k in range(3):
            xx = int(x + rng.normal(0, 6))
            yy = int(y - abs(rng.normal(0, 8)) - 4)
            if 0 <= xx < W and 0 <= yy < H:
                layer[yy, xx] += s * rng.uniform(0.5, 1.0)
    layer = ndi.gaussian_filter(layer, 7) * 160
    layer = np.clip(layer, 0, 0.45)
    c = np.array([0.86, 0.76, 0.62])
    return img * (1 - layer[..., None]) + c * layer[..., None]


def add_clouds(img, S, t, W, H):
    out = img
    if ("cmask", W) not in CACHE:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        centre = np.exp(-((xx - W * 0.45) / (0.24 * W)) ** 2)
        m = smoothstep(0.42, 0.0, yy / H) * (1 - 0.85 * centre) + \
            0.5 * smoothstep(0.78, 1.0, yy / H) * smoothstep(0.55, 0.95, xx / W)
        CACHE[("cmask", W)] = np.clip(m, 0, 1)[..., None]
    cm = CACHE[("cmask", W)]
    for k, cl in enumerate(S["clouds"]):
        cw = cl.shape[1]
        off = int((t * (9 + 6 * k) + k * 300) % (cw - W))
        c = cl[:, off:off + W]
        a = c[..., 3:4] * (0.95 if k == 0 else 0.6) * cm
        out = out * (1 - a) + c[..., :3] * a
    return out


def work(args):
    meta_path, src, dst = args
    if os.path.exists(dst):
        return dst
    with open(meta_path) as fh:
        meta = json.load(fh)
    out = compose(meta, src)
    Image.fromarray(out).save(dst, compress_level=1)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(HERE, "render", "exr"))
    ap.add_argument("--dst", default=os.path.join(HERE, "render", "frames"))
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--frames", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.dst, exist_ok=True)
    frames = range(N) if not a.frames else [int(x) for x in a.frames.split(",")]
    jobs = []
    for fi in frames:
        mp = os.path.join(a.src, "f%04d.json" % fi)
        if not os.path.exists(mp):
            continue
        dst = os.path.join(a.dst, "f%04d.png" % fi)
        if a.force and os.path.exists(dst):
            os.remove(dst)
        jobs.append((mp, os.path.join(a.src, "f%04d.exr" % fi), dst))
    with Pool(a.jobs) as pool:
        for i, d in enumerate(pool.imap_unordered(work, jobs)):
            if i % 20 == 0:
                print(i, len(jobs), d, flush=True)


if __name__ == "__main__":
    main()
