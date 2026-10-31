"""Opening/closing of the film: an old site plan, folded in four and
crumpled, opens on a drafting table and flattens until it fills the frame,
where the 3D model takes over.  The closing plays the same unfolding in
reverse, so the loop closes on the folded sheet.

    python film_map.py --out film/map --res 900x1000 --samples 24
"""
import argparse
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bpy  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Vector  # noqa: E402
from PIL import Image, ImageDraw, ImageFilter, ImageFont  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

from vlc import plan  # noqa: E402
from vlc.common import MB, G, L, new_mat, smooth, clamp  # noqa: E402

FPS = 24
T_MAP = 3.0                      # seconds of unfolding
N = int(FPS * T_MAP) + 1
SW, SH = 0.9, 1.0                # sheet size (m), same aspect as the frame
VIEW_C = (71.0, 92.0)            # world point at the sheet centre
VIEW_W = 259.0                   # metres of city across the sheet width


def fbm(h, w, seed, oct_=6, base=4):
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w), np.float32)
    amp = 1.0
    tot = 0
    for o in range(oct_):
        n = base * 2 ** o
        g = rng.random((n + 3, int(n * w / h) + 3)).astype(np.float32)
        z = ndi.zoom(g, (h / n, h / n), order=3)[:h, :w]
        z = np.pad(z, ((0, max(0, h - z.shape[0])), (0, max(0, w - z.shape[1]))), mode="edge")
        out += amp * z
        tot += amp
        amp *= 0.55
    out /= tot
    return (out - out.min()) / (out.max() - out.min() + 1e-6)


def sheet_texture(path, W=1800, H=2000):
    """Aged site plan (plano de emplazamiento) with foxing and charred edges."""
    from shapely.geometry import Point
    from shapely.geometry.polygon import orient
    rng = np.random.default_rng(3)
    base = np.array([226, 210, 172], np.float32) / 255
    n1 = fbm(H, W, 1, 7, 3)
    n2 = fbm(H, W, 2, 5, 24)
    img = base[None, None] * (0.86 + 0.18 * n1[..., None]) * (0.97 + 0.05 * n2[..., None])
    # tide marks and foxing spots
    for k in range(40):
        cx, cy, r = rng.uniform(0, W), rng.uniform(0, H), rng.uniform(6, 60)
        yy, xx = np.ogrid[:H, :W]
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r
        ring = np.exp(-((d - 1) / 0.08) ** 2) * 0.08 + np.exp(-d ** 2 * 2) * 0.05
        img *= (1 - ring[..., None] * np.array([0.3, 0.55, 0.9]))
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).convert("RGBA")
    d = ImageDraw.Draw(im, "RGBA")
    s = W / VIEW_W

    def P(x, y):
        return (W / 2 + (x - VIEW_C[0]) * s, H / 2 - (y - VIEW_C[1]) * s)

    ink = (58, 40, 28, 235)
    red = (150, 46, 32, 230)
    for blk in plan.BLOCKS:
        if blk.distance(Point(*VIEW_C)) > 260:
            continue
        pts = [P(*c) for c in blk.exterior.coords]
        d.polygon(pts, fill=(196, 170, 128, 70), outline=ink, width=3)
        f = plan.footprint(blk)
        if f is not None and blk.distance(Point(70, 90)) > 1:
            d.line([P(*c) for c in f.exterior.coords], fill=(58, 40, 28, 150), width=2)
            # parcel hatching
            minx, miny, maxx, maxy = f.bounds
            for k in np.arange(minx - (maxy - miny), maxx, 4.0):
                from shapely.geometry import LineString
                ln = LineString([(k, miny), (k + (maxy - miny), maxy)]).intersection(f)
                for g in getattr(ln, "geoms", [ln]):
                    if not g.is_empty and g.geom_type == "LineString":
                        d.line([P(*c) for c in g.coords], fill=(58, 40, 28, 70), width=1)
    isl = plan.ISLAND
    d.line([P(*c) for c in isl.exterior.coords], fill=ink, width=3)
    for c, a, b in plan.parterres():
        d.line([P(*q) for q in plan.ellipse(c, a, b, 40)] + [P(*plan.ellipse(c, a, b, 40)[0])], fill=ink, width=2)
    # the Palacio, in red wash
    F = orient(plan.footprint(plan.block_at((70, 90))).simplify(0.2), 1.0)
    hall = F.buffer(-13.0, join_style=2)
    d.polygon([P(*c) for c in F.exterior.coords], fill=(170, 60, 44, 70), outline=red, width=5)
    d.polygon([P(*c) for c in hall.exterior.coords], fill=(226, 210, 172, 255), outline=red, width=3)
    hc = hall.centroid
    r = 9.5 * s
    cx, cy = P(hc.x, hc.y)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=red, width=3)
    d.line([P(*c) for c in plan.ISLAND.buffer(8.5).exterior.coords], fill=(58, 40, 28, 120), width=2)
    # lettering
    fb = "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf"
    fi = "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf"
    fr = "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"
    F1, F2, F3 = ImageFont.truetype(fb, 64), ImageFont.truetype(fi, 40), ImageFont.truetype(fr, 30)

    def label(txt, xy, font, fill, angle=0):
        tw = d.textbbox((0, 0), txt, font=font)
        lay = Image.new("RGBA", (tw[2] + 20, tw[3] + 20), (0, 0, 0, 0))
        ImageDraw.Draw(lay).text((10, 10), txt, font=font, fill=fill)
        lay = lay.rotate(angle, expand=True, resample=Image.BICUBIC)
        im.alpha_composite(lay, (int(xy[0] - lay.width / 2), int(xy[1] - lay.height / 2)))

    label("PLAZA DE EMILIO CASTELAR", P(10, 4), F2, ink, 8)
    label("Palacio de Comunicaciones", P(68, 96), F3, red, 9)
    label("MARQUÉS DE SOTELO", P(150, 0), F3, ink, 0)
    # cartouche
    x0, y0 = 70, H - 330
    d.rectangle([x0, y0, x0 + 860, y0 + 250], fill=(230, 214, 178, 230), outline=red, width=5)
    d.rectangle([x0 + 12, y0 + 12, x0 + 848, y0 + 238], outline=red, width=2)
    d.text((x0 + 40, y0 + 30), "PLANO DE EMPLAZAMIENTO", font=F1, fill=red)
    d.text((x0 + 40, y0 + 112), "Palacio de Comunicaciones · Valencia", font=F2, fill=ink)
    d.text((x0 + 40, y0 + 170), "M. Á. Navarro, arquitecto  ·  Escala 1:1000", font=F3, fill=ink)
    # north arrow and scale bar
    nx, ny = W - 190, 210
    d.polygon([(nx, ny - 110), (nx - 34, ny + 50), (nx, ny + 18), (nx + 34, ny + 50)], fill=ink)
    d.text((nx - 18, ny + 60), "N", font=F2, fill=ink)
    for k in range(5):
        xa = W - 640 + k * 20 * s
        d.rectangle([xa, H - 120, xa + 20 * s, H - 100], fill=ink if k % 2 == 0 else (0, 0, 0, 0), outline=ink, width=2)
    d.text((W - 640, H - 90), "0        20       40       60       80      100 m", font=F3, fill=ink)
    # fold lines, charred and torn edges (alpha)
    arr = np.asarray(im, np.float32) / 255
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    for k, (dist) in enumerate((np.abs(xx - W / 2), np.abs(yy - H / 2))):
        arr[..., :3] *= (1 - 0.18 * np.exp(-(dist / 4.0) ** 2))[..., None]
    edge = np.minimum.reduce([xx, W - 1 - xx, yy, H - 1 - yy]) / 60.0
    ragged = edge + 0.9 * (fbm(H, W, 9, 6, 8) - 0.5)
    burn = smooth_np(clamp_np(1.2 - ragged, 0, 1))
    arr[..., :3] *= (1 - 0.75 * burn[..., None] * np.array([0.6, 0.75, 0.9]))
    arr[..., 3] = clamp_np(ragged * 2.2 - 0.25, 0, 1)
    Image.fromarray((arr * 255).astype(np.uint8)).save(path)


def clamp_np(x, a, b):
    return np.clip(x, a, b)


def smooth_np(x):
    return x * x * (3 - 2 * x)


# ---------------------------------------------------------------------------
NX, NY = 72, 80
_crumple = None


def sheet_state(t):
    """Fold angles and crumple amount for time t of the unfolding."""
    a1 = math.pi * (1 - smooth((t - 0.35) / 0.95))      # top half over bottom
    a2 = math.pi * (1 - smooth((t - 1.15) / 1.0))       # right half over left
    cr = 1.0 - 0.85 * smooth((t - 1.6) / 1.3)
    lift = 0.05 * math.sin(math.pi * clamp((t - 0.35) / 1.9)) ** 2
    return a1, a2, cr, lift


def sheet_verts(t):
    global _crumple
    if _crumple is None:
        n = fbm(NY + 1, NX + 1, 5, 4, 3) - 0.5
        n2 = fbm(NY + 1, NX + 1, 6, 3, 6) - 0.5
        _crumple = (n * 0.012 + n2 * 0.004)
    a1, a2, cr, lift = sheet_state(t)
    xs = np.linspace(-SW / 2, SW / 2, NX + 1)
    ys = np.linspace(-SH / 2, SH / 2, NY + 1)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)
    # fold 2 (vertical crease x=0), applied first
    right = X > 0
    Xr = X * math.cos(a2) - Z * math.sin(a2)
    Zr = X * math.sin(a2) + Z * math.cos(a2) + 0.003 * math.sin(a2 / 2) ** 2
    X1 = np.where(right, Xr, X)
    Z1 = np.where(right, Zr, Z)
    # fold 1 (horizontal crease y=0) on the result
    top = Y > 0
    Yt = Y * math.cos(a1) - Z1 * math.sin(a1)
    Zt = Y * math.sin(a1) + Z1 * math.cos(a1) + 0.006 * math.sin(a1 / 2) ** 2
    Y2 = np.where(top, Yt, Y)
    Z2 = np.where(top, Zt, Z1)
    # crumples and residual creases sampled at the final (x, y): every layer of
    # the folded stack gets the same relief, so layers never intersect
    from scipy.ndimage import map_coordinates
    gi = (Y2 + SH / 2) / SH * NY
    gj = (X1 + SW / 2) / SW * NX
    crm = map_coordinates(_crumple, [np.clip(gi, 0, NY), np.clip(gj, 0, NX)], order=1, mode="nearest")
    flat = math.sin(a1 / 2) ** 2 + math.sin(a2 / 2) ** 2   # crumples are stronger while folded
    crease = 0.004 * np.exp(-(X / 0.012) ** 2) * (1 - math.sin(a2 / 2) ** 2) + \
        0.003 * np.exp(-(Y / 0.012) ** 2) * (1 - math.sin(a1 / 2) ** 2)
    Z2 = Z2 + crm * cr * (0.6 + 0.4 * min(flat, 1.0)) + crease
    Z2 = Z2 + lift * np.exp(-((X1 / 0.35) ** 2 + (Y2 / 0.35) ** 2)) + 0.004
    return np.stack([X1, Y2, Z2], -1).reshape(-1, 3)


def build_scene(res, samples, tex_path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.view_settings.view_transform = "Standard"
    sc.render.image_settings.media_type = "IMAGE"
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_depth = "16"
    w = bpy.data.worlds.new("w")
    sc.world = w
    nt = w.node_tree
    bg = nt.nodes.get("Background") or nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = (0.42, 0.36, 0.30, 1)
    bg.inputs["Strength"].default_value = 0.35
    out = nt.nodes.get("World Output") or nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(bg.outputs[0], out.inputs[0])
    # warm low window light + soft fill
    sun = bpy.data.lights.new("sun", "SUN")
    sun.energy = 3.4
    sun.color = (1.0, 0.82, 0.62)
    sun.angle = math.radians(3)
    so = bpy.data.objects.new("sun", sun)
    sc.collection.objects.link(so)
    so.rotation_euler = (math.radians(58), 0, math.radians(62))
    area = bpy.data.lights.new("fill", "AREA")
    area.energy = 60
    area.size = 2.0
    area.color = (0.75, 0.82, 1.0)
    ao = bpy.data.objects.new("fill", area)
    sc.collection.objects.link(ao)
    ao.location = (-1.2, -1.0, 1.8)
    ao.rotation_euler = (math.radians(40), 0, math.radians(-50))
    # table: walnut
    m = new_mat("walnut")
    g = G(m)
    tc = g.texco().outputs["Object"]
    sep = g.sep(tc)
    v = g.comb(g.math("MULTIPLY", sep[0], 1.5), g.math("MULTIPLY", sep[1], 38.0), 0)
    grain = g.noise(v, scale=1.0, detail=8, rough=0.6, dist=0.4).outputs["Fac"]
    plank = g.math("FRACT", g.math("MULTIPLY", sep[1], 3.2))
    seam = g.math("LESS_THAN", plank, 0.012)
    col = g.ramp(grain, [(0.3, L(44, 26, 16)), (0.55, L(80, 48, 28)), (0.75, L(104, 66, 38))])
    col = g.mix(g.math("MULTIPLY", seam, 0.8), col, L(20, 12, 8))
    nrm = g.bump(g.math("ADD", grain, g.math("MULTIPLY", seam, -1.0)), strength=0.12, distance=0.002)
    g.finish(g.bsdf(col, rough=0.32, normal=nrm, spec=0.5))
    mb = MB()
    mb.box((0, 0, -0.04), (7.0, 7.0, 0.04), (1, 1, 1), 0)
    mb.build("table", [m])
    # props: dividers, inkwell, pen, ruler, rolled sheet
    brass = new_mat("brass")
    g = G(brass)
    g.finish(g.bsdf(L(150, 118, 60), rough=0.35, metallic=1.0, spec=0.5))
    glass = new_mat("inkglass")
    g = G(glass)
    g.finish(g.bsdf(L(20, 24, 30), rough=0.05, spec=1.0, transmission=0.6))
    mb = MB()
    hinge = np.array([0.72, 0.38, 0.012])
    for dx in (-0.05, 0.05):
        mb.beam(tuple(hinge), (hinge[0] + dx * 1.6, hinge[1] - 0.17, 0.004), 0.008, (1, 1, 1), 0)
    mb.cyl(tuple(hinge - [0, 0, 0.008]), 0.014, 0.016, 12, (1, 1, 1), 0)
    mb.box((0.0, -0.66, 0.0), (0.62, 0.035, 0.004), (1, 1, 1), 0, rot=0.12)
    mb.build("brass", [brass])
    mb = MB()
    mb.cyl((-0.66, 0.42, 0.0), 0.05, 0.07, 20, (1, 1, 1), 0, r2=0.045)
    mb.build("inkwell", [glass])
    mb = MB()
    mb.cyl((-0.66, 0.42, 0.07), 0.025, 0.025, 14, L(30, 30, 30), 0)
    mb.beam((-0.86, 0.22, 0.01), (-0.6, 0.08, 0.006), 0.008, L(40, 30, 24), 0)
    mb.beam((-0.6, 0.08, 0.006), (-0.57, 0.065, 0.004), 0.004, L(150, 130, 70), 0)
    rolled = MB()
    rolled.cyl((0, 0, -0.32), 0.035, 0.64, 20, L(220, 204, 168), 0)
    for i_, v in enumerate(rolled.v):
        rolled.v[i_] = (0.75 + v[2] * 0.2, -0.55 + v[2], 0.035 + v[0])
    mb.merge(rolled)
    pm = new_mat("propplain")
    g = G(pm)
    g.finish(g.bsdf(g.attr("Col").outputs["Color"], rough=0.6))
    mb.build("pen_roll", [pm])
    # the sheet
    img = bpy.data.images.load(tex_path)
    sm = new_mat("sheet")
    g = G(sm)
    uvn = g.uv()
    tex = g.node("ShaderNodeTexImage")
    tex.image = img
    g.link(uvn, tex.inputs["Vector"])
    geo = g.geo()
    back = g.mix(geo.outputs["Backfacing"], tex.outputs["Color"], L(222, 206, 170))
    pos = geo.outputs["Position"]
    nn = g.noise(pos, scale=60, detail=3).outputs["Fac"]
    nrm = g.bump(nn, strength=0.06, distance=0.0005)
    bsdf = g.bsdf(back, rough=0.85, normal=nrm, sss=0.0)
    tr = g.node("ShaderNodeBsdfTranslucent")
    tr.inputs["Color"].default_value = (*L(230, 210, 170), 1)
    mx = g.node("ShaderNodeMixShader")
    mx.inputs[0].default_value = 0.12
    g.link(bsdf, mx.inputs[1])
    g.link(tr.outputs[0], mx.inputs[2])
    tp = g.node("ShaderNodeBsdfTransparent")
    mx2 = g.node("ShaderNodeMixShader")
    g.link(tex.outputs["Alpha"], mx2.inputs[0])
    g.link(tp.outputs[0], mx2.inputs[1])
    g.link(mx.outputs[0], mx2.inputs[2])
    g.finish(mx2.outputs[0])
    verts = sheet_verts(0.0)
    faces = []
    for j in range(NY):
        for i in range(NX):
            a = j * (NX + 1) + i
            faces.append((a, a + 1, a + NX + 2, a + NX + 1))
    me = bpy.data.meshes.new("sheet")
    me.from_pydata([tuple(v) for v in verts], [], faces)
    uv = me.uv_layers.new(name="UVMap")
    uvs = []
    for f in faces:
        for vi in f:
            i, j = vi % (NX + 1), vi // (NX + 1)
            uvs.append((i / NX, j / NY))
    uv.data.foreach_set("uv", np.array(uvs, np.float32).ravel())
    for p in me.polygons:
        p.use_smooth = True
    me.materials.append(sm)
    ob = bpy.data.objects.new("sheet", me)
    sc.collection.objects.link(ob)
    cd = bpy.data.cameras.new("cam")
    cd.lens = 32
    cd.sensor_fit = "HORIZONTAL"
    cd.sensor_width = 36
    cd.dof.use_dof = True
    cam = bpy.data.objects.new("cam", cd)
    sc.collection.objects.link(cam)
    sc.camera = cam
    return me, cam


def place_cam(cam, t):
    # oblique and close at first, ending straight above with the sheet filling the frame
    k = smooth(clamp((t - 0.2) / 2.8))
    h_end = (SW / 2) / math.tan(math.atan(18 / 32)) * 1.0
    p0 = np.array([0.32, -1.05, 1.05])
    p1 = np.array([0.0, 0.0, h_end])
    pos = p0 + (p1 - p0) * k
    tgt = np.array([-0.05, -0.12, 0.0]) * (1 - k)
    cam.location = Vector(pos)
    from mathutils import Quaternion
    q0 = (Vector((-0.05, -0.12, 0.0)) - Vector(p0)).to_track_quat("-Z", "Y")
    q = q0.slerp(Quaternion(), k)
    cam.rotation_mode = "XYZ"
    cam.rotation_euler = q.to_euler()
    cam.data.dof.focus_distance = float(np.linalg.norm(pos - tgt))
    cam.data.dof.aperture_fstop = 2.0 + 14 * k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "film", "map"))
    ap.add_argument("--res", default="900x1000")
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--frames", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tex = os.path.join(a.out, "sheet_texture.png")
    if not os.path.exists(tex):
        sheet_texture(tex)
    res = tuple(int(v) for v in a.res.split("x"))
    me, cam = build_scene(res, a.samples, tex)
    frames = range(N) if not a.frames else [int(x) for x in a.frames.split(",")]
    for i in frames:
        t = i / FPS
        v = sheet_verts(t)
        me.vertices.foreach_set("co", v.astype(np.float32).ravel())
        me.update()
        place_cam(cam, t)
        bpy.context.scene.render.filepath = os.path.join(a.out, "m%04d.png" % i)
        bpy.ops.render.render(write_still=True)
        print("map frame", i, flush=True)


if __name__ == "__main__":
    main()
