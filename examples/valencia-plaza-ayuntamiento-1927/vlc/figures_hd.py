"""Organic, articulated figures for close-up shots: 1910s Valencian workers
(blusa, faja, alpargatas, boina or straw hat) and townsfolk, mules and
draught horses with collar harness, two-wheeled carts with spoked wheels.

Bodies are metaball skeletons (capsules and ellipsoids on a jointed rig:
hip-knee-ankle, shoulder-elbow-wrist, neck, head) polygonised to meshes;
each vertex is coloured by the body part that dominates it, so clothing,
skin, sash and footwear read correctly.  Tools held in the hands are real
geometry (hammer, shovel, hod, barrow, plan sheet)."""
import math

import bpy
import numpy as np
from mathutils import Quaternion, Vector

from .common import MB, L, jitter_col
from . import figures as F0

K = 1.62          # metaball radius factor -> visible radius when blended
_N = [0]


def _quat_x_to(d):
    d = Vector(d)
    if d.length < 1e-9:
        return Quaternion()
    return Vector((1, 0, 0)).rotation_difference(d.normalized())


class Rig:
    """Collects metaball elements tagged with a part name."""

    def __init__(self):
        self.els = []     # (type, co, radius, extra, part)

    def ball(self, p, r, part):
        self.els.append(("BALL", tuple(p), r, None, part))

    def ell(self, p, r, sx, sy, sz, part, rot=None):
        self.els.append(("ELLIPSOID", tuple(p), r, (sx, sy, sz, rot), part))

    def cap(self, a, b, r, part):
        a, b = np.array(a, float), np.array(b, float)
        self.els.append(("CAPSULE", tuple((a + b) / 2), r, (np.linalg.norm(b - a) / 2, b - a), part))

    def mesh(self, name, colors, res=0.028):
        _N[0] += 1
        mbd = bpy.data.metaballs.new("mbz%d" % _N[0])
        mbd.resolution = res
        mbd.render_resolution = res
        mbd.threshold = 0.6
        ob = bpy.data.objects.new("mbz%d" % _N[0], mbd)
        bpy.context.scene.collection.objects.link(ob)
        for typ, co, r, extra, part in self.els:
            e = mbd.elements.new()
            e.type = typ
            e.co = co
            e.radius = r * K
            if typ == "CAPSULE":
                hl, d = extra
                e.size_x = hl
                e.rotation = _quat_x_to(d)
            elif typ == "ELLIPSOID":
                sx, sy, sz, rot = extra
                e.size_x, e.size_y, e.size_z = sx, sy, sz
                if rot is not None:
                    e.rotation = rot
        dg = bpy.context.evaluated_depsgraph_get()
        me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
        bpy.data.objects.remove(ob)
        bpy.data.metaballs.remove(mbd)
        me.name = name
        # colour each vertex by the nearest element surface
        V = np.zeros((len(me.vertices), 3))
        me.vertices.foreach_get("co", V.ravel())
        best = np.full(len(V), 1e9)
        lab = np.zeros(len(V), int)
        for i, (typ, co, r, extra, part) in enumerate(self.els):
            c = np.array(co)
            if typ == "CAPSULE":
                hl, d = extra
                d = np.array(d) / (np.linalg.norm(d) + 1e-9)
                tt = np.clip((V - c) @ d, -hl, hl)
                dist = np.linalg.norm(V - (c + tt[:, None] * d), axis=1) - r
            elif typ == "ELLIPSOID":
                sx, sy, sz, rot = extra
                dist = np.linalg.norm((V - c) / np.array([sx, sy, sz]), axis=1) * r - r
            else:
                dist = np.linalg.norm(V - c, axis=1) - r
            m = dist < best
            best[m] = dist[m]
            lab[m] = i
        parts = [e[4] for e in self.els]
        vc = np.array([colors.get(parts[i], (1, 0, 1)) for i in range(len(parts))])[lab]
        nl = len(me.loops)
        li = np.zeros(nl, np.int32)
        me.loops.foreach_get("vertex_index", li)
        col = np.ones((nl, 4), np.float32)
        col[:, :3] = vc[li]
        ca = me.color_attributes.new("Col", "FLOAT_COLOR", "CORNER")
        ca.data.foreach_set("color", col.ravel())
        for p in me.polygons:
            p.use_smooth = True
        return me


def _rx(v, a):
    """Rotate about x; +a swings a downward vector forward (+y)."""
    y, z = v[1], v[2]
    c, s = math.cos(a), math.sin(a)
    return np.array([v[0], y * c - z * s, y * s + z * c])


def _about(p, piv, a):
    return piv + _rx(np.asarray(p) - piv, a)


# ---------------------------------------------------------------------------
# People
# ---------------------------------------------------------------------------
def person_rig(st, legs=((0, 0.05), (0, 0.05)), arms=((0.05, 0.15), (-0.05, 0.15)),
               lean=0.0, bob=0.0, head_tilt=0.0, twist=0.0):
    """legs: ((thigh, knee), ...) arms: ((shoulder, elbow), ...)"""
    s = st.scale
    R = Rig()
    hipz = (0.92 + bob) * s
    piv = np.array([0, 0, hipz])

    def up(p):        # upper body leans forward about the hip
        return _about(p, piv, -lean)

    ends = {}
    for k, x in enumerate((-0.095 * s, 0.095 * s)):
        a, kn = legs[k]
        hip = np.array([x, 0, hipz])
        knee = hip + _rx(np.array([0, 0, -0.45 * s]), a)
        ank = knee + _rx(np.array([0, 0, -0.43 * s]), a - kn)
        toe = ank + np.array([0, 0.17 * s, -0.045 * s])
        thigh_part = "skirt" if st.skirt is not None else "trousers"
        R.cap(hip, knee, 0.094 * s, thigh_part)
        R.cap(knee, ank, 0.066 * s, "legs")
        R.cap(ank + np.array([0, -0.04 * s, 0.0]), toe, 0.052 * s, "shoes")
    if st.skirt is not None:
        for z, r in ((1.0, 0.155), (0.8, 0.2), (0.58, 0.245), (0.38, 0.285), (0.22, 0.30)):
            R.ell(up((0, 0.02 * s, z * s)), r * s, 1.0, 0.85, 0.75, "skirt")
    # trunk
    R.ell(up((0, 0, 0.98 * s)), 0.16 * s, 1.15, 0.85, 0.8, "trousers")      # pelvis
    R.ell(up((0, 0, 1.10 * s)), 0.155 * s, 1.1, 0.8, 0.7, "sash")         # faja
    R.ell(up((0, 0.01 * s, 1.28 * s)), 0.18 * s, 1.12, 0.8, 1.0, "shirt")  # chest
    R.ell(up((0, 0, 1.40 * s)), 0.16 * s, 1.35, 0.75, 0.55, "shirt")        # shoulders
    neck = up((0, 0.01 * s, 1.50 * s))
    R.cap(neck, up((0, 0.02 * s, 1.56 * s)), 0.058 * s, "skin")
    hc = up((0, 0.03 * s, 1.65 * s)) + np.array([0, math.sin(head_tilt) * 0.05 * s, 0])
    R.ell(hc, 0.113 * s, 0.88, 1.0, 1.12, "skin")
    R.ball(hc + np.array([0, 0.095 * s, -0.01 * s]), 0.022 * s, "skin")        # nose
    if st.hair is not None:
        R.ell(hc + np.array([0, -0.025 * s, 0.04 * s]), 0.112 * s, 0.95, 1.0, 0.85, "hair")
    # arms
    hands = []
    for k, x in enumerate((-0.2 * s, 0.2 * s)):
        b, e = arms[k]
        sh = up((x, 0, 1.40 * s))
        el = sh + _rx(_rx(np.array([0, 0, -0.30 * s]), b), lean * 0)
        el = _about(sh + _rx(np.array([0, 0, -0.30 * s]), b), sh, 0)
        wr = el + _rx(np.array([0, 0, -0.27 * s]), b + e)
        if lean:
            el = _about(el, piv, 0) if False else el
        R.cap(sh, el, 0.064 * s, "shirt")
        R.cap(el, wr, 0.052 * s, "forearm")
        hd = wr + _rx(np.array([0, 0, -0.07 * s]), b + e)
        R.ball(hd, 0.05 * s, "skin")
        hands.append((wr, hd, b + e))
    ends["hands"] = hands
    ends["head"] = hc
    return R, ends


def person_mesh(st, name, colors, pose_kw, tool=None, tool_phase=0.0):
    R, ends = person_rig(st, **pose_kw)
    me = R.mesh(name, colors)
    # hats and held tools as plain geometry, merged into one mesh
    mb = MB()
    hc = ends["head"]
    s = st.scale
    hat = colors.get("hat", (0.1, 0.1, 0.1))
    top = hc + np.array([0, -0.005, 0.085 * s])
    if st.hat_kind == "beret":
        mb.cyl(tuple(top - [0, 0, 0.01]), 0.125 * s, 0.045 * s, 12, hat, 0, r2=0.105 * s)
    elif st.hat_kind == "straw":
        mb.cyl(tuple(top - [0, 0, 0.02]), 0.23 * s, 0.018 * s, 16, hat, 0)
        mb.cyl(tuple(top - [0, 0, 0.0]), 0.115 * s, 0.095 * s, 12, hat, 0, r2=0.1 * s)
        mb.cyl(tuple(top + [0, 0, 0.0]), 0.118 * s, 0.025 * s, 12, L(60, 40, 30), 0)
    elif st.hat_kind == "fedora":
        mb.cyl(tuple(top - [0, 0, 0.02]), 0.18 * s, 0.015 * s, 16, hat, 0)
        mb.cyl(tuple(top - [0, 0, 0.0]), 0.11 * s, 0.12 * s, 12, hat, 0, r2=0.095 * s)
    elif st.hat_kind == "cap":
        mb.cyl(tuple(top - [0, 0, 0.01]), 0.12 * s, 0.06 * s, 12, hat, 0, r2=0.115 * s)
        mb.box(tuple(top + [0, 0.12 * s, -0.01]), (0.17 * s, 0.11 * s, 0.015 * s), hat, 0)
    elif st.hat_kind == "cloche":
        mb.dome(tuple(top - [0, 0, 0.04]), 0.13 * s, n=12, rings=4, col=hat, mat=0, squash=0.9)
    wood, iron = L(150, 112, 74), L(70, 72, 76)
    if tool in ("hammer", "shovel", "plan", "hod_hand"):
        (wr, hd, ang) = ends["hands"][1]
        d = _rx(np.array([0, 0, -1.0]), ang)
        if tool == "hammer":       # mason's hammer / maceta
            tip = hd + d * 0.32
            mb.beam(tuple(hd - d * 0.05), tuple(tip), 0.025, wood, 1)
            side = np.array([1.0, 0, 0])
            mb.beam(tuple(tip - side * 0.07), tuple(tip + side * 0.07), 0.06, iron, 2)
        elif tool == "shovel":
            (wl, hl, al) = ends["hands"][0]
            a0 = hl - (hd - hl) * 0.4
            a1 = hd + (hd - hl) * 1.6
            mb.beam(tuple(a0), tuple(a1), 0.035, wood, 1)
            dd = (a1 - a0) / (np.linalg.norm(a1 - a0) + 1e-9)
            n = np.cross(dd, [1, 0, 0])
            n = n / (np.linalg.norm(n) + 1e-9)
            c0 = a1 + dd * 0.02
            i = mb.add_verts([c0 - np.array([0.12, 0, 0]), c0 + np.array([0.12, 0, 0]),
                              c0 + dd * 0.3 + np.array([0.11, 0, 0]) + n * 0.03,
                              c0 + dd * 0.3 - np.array([0.11, 0, 0]) + n * 0.03])
            mb.face((i, i + 1, i + 2, i + 3), iron, 2)
        elif tool == "plan":
            (wl, hl, al) = ends["hands"][0]
            c = (hl + hd) / 2 + np.array([0, 0.05, 0.02])
            i = mb.add_verts([c + [-0.32, -0.12, 0.0], c + [0.32, -0.12, 0.0], c + [0.32, 0.2, 0.06],
                              c + [-0.32, 0.2, 0.06]])
            mb.face((i, i + 1, i + 2, i + 3), L(236, 228, 206), 0)
    if tool == "hod":              # V-trough of bricks carried on the shoulder
        c = np.array([0.17 * s, -0.02, 1.52 * s])
        mb.box(tuple(c + [0, 0, 0]), (0.18, 0.55, 0.16), wood, 1)
        for j in range(5):
            mb.box(tuple(c + [0, -0.2 + 0.1 * j, 0.15]), (0.14, 0.09, 0.06), L(176, 86, 56), 0)
        mb.beam(tuple(c + [0, 0.1, -0.02]), tuple(c + [0, 0.25, -0.75]), 0.035, wood, 1)
    if tool == "plank":
        c = np.array([0.16 * s, 0.0, 1.56 * s])
        mb.box(tuple(c), (0.24, 3.6, 0.05), L(176, 140, 96), 1)
    if tool == "barrow":
        bm = wheelbarrow()
        for i, v in enumerate(bm.v):
            bm.v[i] = (v[0], v[1] + 0.78, v[2] + 0.06)
        mb.merge(bm)
    if tool == "basket":
        c = hc + np.array([0, 0, 0.16])
        mb.cyl(tuple(c), 0.22, 0.18, 14, L(186, 150, 96), 1, r2=0.27)
        mb.cyl(tuple(c + [0, 0, 0.16]), 0.25, 0.04, 14, L(200, 60, 70), 0)
    if mb.v:
        extra = mb.mesh(name + "_x")
        # join the two meshes
        import bmesh
        bm_ = bmesh.new()
        bm_.from_mesh(me)
        bm_.from_mesh(extra)
        bm_.to_mesh(me)
        bm_.free()
        bpy.data.meshes.remove(extra)
    return me


def wheelbarrow(mat_wood=1, mat_iron=2):
    """Timber barrow with iron-tyred spoked wheel, tray, handles and legs."""
    mb = MB()
    w = L(146, 108, 70)
    iron = L(60, 60, 62)
    # tray: tapered box (open top) from planks
    z0, z1 = 0.32, 0.62
    bot = [(-0.22, 0.25), (0.22, 0.25), (0.2, 0.85), (-0.2, 0.85)]
    topp = [(-0.33, 0.12), (0.33, 0.12), (0.28, 0.98), (-0.28, 0.98)]
    for k in range(4):
        a, b = bot[k], bot[(k + 1) % 4]
        c, d = topp[(k + 1) % 4], topp[k]
        i = mb.add_verts([(*a, z0), (*b, z0), (*c, z1), (*d, z1)])
        mb.face((i, i + 1, i + 2, i + 3), jitter_col(w, np.random.default_rng(k), 0.1) if False else w, mat_wood)
    i = mb.add_verts([(*p, z0) for p in bot])
    mb.face((i + 3, i + 2, i + 1, i), w, mat_wood)
    # handles running under the tray to the wheel
    for x in (-0.24, 0.24):
        mb.beam((x * 1.25, -0.55, 0.74), (x * 0.45, 1.2, 0.3), 0.045, w, mat_wood)
        mb.beam((x, 0.35, 0.32), (x * 1.05, 0.38, 0.0), 0.04, w, mat_wood)   # legs
    # wheel
    wc = np.array([0, 1.22, 0.24])
    r = 0.24
    n = 20
    for k in range(n):
        a0, a1 = 2 * math.pi * k / n, 2 * math.pi * (k + 1) / n
        p0 = wc + np.array([0, r * math.cos(a0), r * math.sin(a0)])
        p1 = wc + np.array([0, r * math.cos(a1), r * math.sin(a1)])
        mb.beam(tuple(p0), tuple(p1), 0.045, iron, mat_iron)
    for k in range(8):
        a = 2 * math.pi * k / 8
        mb.beam(tuple(wc), tuple(wc + np.array([0, r * math.cos(a), r * math.sin(a)])), 0.022, w, mat_wood)
    mb.beam(tuple(wc - [0.06, 0, 0]), tuple(wc + [0.06, 0, 0]), 0.07, iron, mat_iron)
    return mb


def make_hd_styles():
    """Fewer, richer styles than the distant crowd."""
    S = []
    SK = F0.SKIN
    blue = [L(48, 70, 112), L(66, 88, 124), L(40, 56, 88)]
    sashes = [L(150, 30, 34), L(30, 28, 30), L(40, 50, 90)]
    for k in range(8):
        r = k % 4
        if r == 0:
            st = F0.Style("worker", blue[k % 3], L(70, 64, 58), L(28, 26, 26), SK[k % 4], "beret")
        elif r == 1:
            st = F0.Style("worker", L(226, 220, 204), L(92, 72, 52), L(212, 188, 130), SK[k % 4], "straw")
        elif r == 2:
            st = F0.Style("worker", L(150, 146, 136), L(56, 66, 96), L(70, 66, 60), SK[k % 4], "cap")
        else:
            st = F0.Style("worker", L(206, 192, 162), blue[k % 3], L(28, 26, 26), SK[k % 4], "beret")
        st.sash = sashes[k % 3]
        st.rolled = k % 2 == 0
        S.append(st)
    suits = [L(36, 36, 42), L(60, 56, 52), L(198, 188, 160), L(110, 104, 92), L(46, 52, 70)]
    hats = [L(30, 30, 30), L(214, 196, 150), L(80, 74, 66)]
    for k in range(5):
        st = F0.Style("man", suits[k], suits[k], hats[k % 3], SK[k % 4], "fedora" if k % 2 else "straw")
        st.sash, st.rolled = suits[k], False
        S.append(st)
    dresses = [L(150, 40, 52), L(60, 82, 124), L(226, 216, 196), L(92, 112, 72), L(40, 40, 44)]
    for k in range(5):
        d = dresses[k]
        st = F0.Style("woman", d, SK[(k + 1) % 4], jitter_col(d, np.random.default_rng(k), 0.3),
                      SK[(k + 1) % 4], "cloche", skirt=d, hair=L(40, 28, 20) if k % 2 else L(80, 56, 34))
        st.sash, st.rolled = d, False
        S.append(st)
    st = F0.Style("child", L(230, 220, 200), L(70, 70, 80), L(30, 30, 30), SK[0], "cap", scale=0.68)
    st.sash, st.rolled = L(70, 70, 80), False
    S.append(st)
    st = F0.Style("foreman", L(70, 64, 56), L(60, 56, 50), L(40, 38, 36), SK[0], "fedora")
    st.sash, st.rolled = L(70, 64, 56), False
    S.append(st)
    return S


def colors_for(st):
    return {"shirt": st.shirt, "trousers": st.trousers if st.skirt is None else st.skirt,
            "legs": st.trousers if st.kind != "woman" else L(60, 50, 46),
            "skin": st.skin, "forearm": st.skin if st.rolled else st.shirt,
            "sash": st.sash, "shoes": L(196, 180, 150) if st.kind == "worker" else L(28, 22, 20),
            "hair": st.hair or L(40, 30, 24), "skirt": st.skirt or st.trousers, "hat": st.hat}


def build_hd_library(M):
    """Pose meshes keyed like figures.build_pose_library: (style, pose, k)."""
    styles = make_hd_styles()
    lib = {}
    nW = sum(1 for s in styles if s.kind == "worker")
    for si, st in enumerate(styles):
        cols = colors_for(st)
        nm = "hd%d" % si

        def add(key, kw, tool=None):
            me = person_mesh(st, "%s_%s_%d" % (nm, key[0], key[1]), cols, kw, tool)
            me.materials.append(M["people"])
            me.materials.append(M["wood"])
            me.materials.append(M["metal"])
            lib[(si, key[0], key[1])] = me

        for k in range(F0.N_WALK):
            ph = 2 * math.pi * k / F0.N_WALK
            sa = math.sin(ph)
            kn = lambda o: 0.08 + 0.55 * max(0.0, math.sin(ph + o + 0.9))
            legs = ((0.36 * sa, kn(0)), (-0.36 * sa, kn(math.pi)))
            bob = 0.018 * math.cos(2 * ph)
            add(("walk", k), dict(legs=legs, arms=((-0.32 * sa, 0.25), (0.32 * sa, 0.25)), bob=bob))
            if st.kind == "worker":
                add(("hod", k), dict(legs=legs, arms=((-0.3 * sa, 0.3), (1.2, 1.75)), bob=bob,
                                     lean=0.08), tool="hod")
                add(("plank", k), dict(legs=legs, arms=((-0.3 * sa, 0.3), (1.0, 1.9)), bob=bob), tool="plank")
                legs_b = ((0.3 * sa, kn(0)), (-0.3 * sa, kn(math.pi)))
                add(("barrow", k), dict(legs=legs_b, arms=((0.45, 0.05), (0.45, 0.05)), lean=0.18, bob=bob),
                    tool="barrow")
            if st.kind == "woman" and si % 2 == 0:
                add(("basket", k), dict(legs=legs, arms=((2.9, 0.4), (-0.3 * sa, 0.25)), bob=bob), tool="basket")
        add(("stand", 0), dict(arms=((0.06, 0.2), (-0.04, 0.25)), legs=((0.04, 0.05), (-0.03, 0.08))))
        if st.kind in ("worker", "foreman"):
            for k in range(F0.N_WORK):
                ph = 2 * math.pi * k / F0.N_WORK
                sw = 0.5 + 0.5 * math.sin(ph)
                add(("work", k), dict(legs=((0.25, 0.25), (-0.15, 0.2)),
                                      arms=((0.55, 0.9), (0.4 + 1.3 * sw, 0.9 - 0.7 * sw)),
                                      lean=0.32 + 0.1 * math.sin(ph + 0.6), head_tilt=0.3), tool="hammer")
                dg = math.sin(ph) ** 2
                add(("dig", k), dict(legs=((0.35, 0.45), (-0.2, 0.35)),
                                     arms=((0.6 + 0.5 * dg, 0.5), (0.9 + 0.5 * dg, 0.2)),
                                     lean=0.25 + 0.4 * dg), tool="shovel")
        if st.kind == "foreman":
            add(("plan", 0), dict(arms=((0.8, 0.75), (0.8, 0.75)), head_tilt=0.4), tool="plan")
            for k in range(F0.N_WORK):
                ph = 2 * math.pi * k / F0.N_WORK
                add(("point", k), dict(arms=((0.3, 0.3), (1.7 + 0.3 * math.sin(ph), 0.1))))
    return styles, nW, lib


# ---------------------------------------------------------------------------
# Animals and carts
# ---------------------------------------------------------------------------
def mule_mesh(name, coat, phase, M, cart=None, scale=1.0, rng=None):
    s = scale
    R = Rig()
    bob = 0.02 * math.cos(2 * phase)
    z = 1.22 * s + bob
    R.ell((0, 0.0, z), 0.34 * s, 1.0, 2.2, 1.0, "coat")              # barrel
    R.ball((0, 0.58 * s, z + 0.05 * s), 0.30 * s, "coat")            # chest
    R.ball((0, -0.6 * s, z + 0.08 * s), 0.31 * s, "coat")            # rump
    nb = np.array([0, 0.78 * s, z + 0.18 * s])
    nt = np.array([0, 1.05 * s, z + 0.62 * s + 0.03 * math.sin(phase)])
    R.cap(nb, nt, 0.16 * s, "coat")
    for j in range(5):                                                 # mane
        t = j / 4
        R.ball(nb + (nt - nb) * t + np.array([0, -0.08 * s, 0.13 * s]), 0.05 * s, "mane")
    hd0 = nt + np.array([0, 0.05 * s, 0.02 * s])
    hd1 = hd0 + np.array([0, 0.36 * s, -0.36 * s])
    R.cap(hd0, hd1, 0.12 * s, "coat")
    R.ball(hd1 + np.array([0, 0.03 * s, -0.02 * s]), 0.10 * s, "muzzle")
    for x in (-0.07, 0.07):                                            # long mule ears
        R.cap(hd0 + np.array([x * s, -0.02 * s, 0.08 * s]), hd0 + np.array([x * 2.2 * s, -0.08 * s, 0.38 * s]),
              0.035 * s, "coat")
    R.cap((0, -0.88 * s, z + 0.12 * s), (0, -1.0 * s, z - 0.55 * s), 0.05 * s, "mane")   # tail
    # four legs, lateral-sequence walk
    offs = {"LF": 0.0, "RH": 0.5 * math.pi, "RF": math.pi, "LH": 1.5 * math.pi}
    for key, (x, y) in {"LF": (-0.17, 0.55), "RF": (0.17, 0.55), "LH": (-0.17, -0.58), "RH": (0.17, -0.58)}.items():
        ph = phase + offs[key]
        a = 0.32 * math.sin(ph)
        flex = 0.6 * max(0.0, math.sin(ph + 0.8))
        top = np.array([x * s, y * s, z - 0.05 * s])
        hind = y < 0
        kn = top + _rx(np.array([0, 0, -0.5 * s]), a + (0.15 if hind else 0))
        ft = kn + _rx(np.array([0, 0, -0.46 * s]), a - (flex if not hind else -flex * 0.6) - (0.25 if hind else 0))
        R.cap(top, kn, (0.12 if hind else 0.10) * s, "coat")
        R.cap(kn, ft, 0.05 * s, "leg")
        R.ball(ft + np.array([0, 0.02 * s, 0.0]), 0.065 * s, "hoof")
    cols = {"coat": coat, "mane": (coat[0] * 0.35, coat[1] * 0.33, coat[2] * 0.3),
            "muzzle": (min(1, coat[0] * 1.8 + 0.1),) * 3, "leg": (coat[0] * 0.6, coat[1] * 0.55, coat[2] * 0.5),
            "hoof": L(40, 36, 32)}
    me = R.mesh(name, cols, res=0.035)
    mb = MB()
    leather = L(70, 44, 30)
    brass = L(176, 146, 70)
    # collar (horse-collar ring), bridle, blinkers, back pad, breeching
    cc = nb + np.array([0, 0.05 * s, 0.05 * s])
    for k in range(16):
        a0, a1 = 2 * math.pi * k / 16, 2 * math.pi * (k + 1) / 16
        p0 = cc + np.array([0.22 * s * math.cos(a0), 0.0, 0.28 * s * math.sin(a0)])
        p1 = cc + np.array([0.22 * s * math.cos(a1), 0.0, 0.28 * s * math.sin(a1)])
        mb.beam(tuple(p0), tuple(p1), 0.09 * s, leather, 1)
    for x in (-1, 1):
        mb.beam(tuple(cc + [x * 0.24 * s, 0, 0.25 * s]), tuple(cc + [x * 0.2 * s, 0.03, 0.42 * s]), 0.04, brass, 2)
        mb.box(tuple(hd0 + [x * 0.13 * s, 0.12 * s, -0.08 * s]), (0.02, 0.12 * s, 0.1 * s), leather, 1)
        mb.beam(tuple(cc + [x * 0.24 * s, 0, -0.05 * s]), (x * 0.3 * s, -0.95 * s, z - 0.05 * s), 0.035, leather, 1)
    mb.beam(tuple(hd0 + [0, 0.02, 0.06]), tuple(hd1 + [0, 0.02, 0.0]), 0.03, leather, 1)
    mb.box((0, -0.05 * s, z + 0.3 * s), (0.5 * s, 0.45 * s, 0.08 * s), L(110, 40, 34), 1)
    if cart is not None:
        cmb = cart_mb(cart, rng, phase)
        mb.merge(cmb)
    if mb.v:
        extra = mb.mesh(name + "_h")
        import bmesh
        bm_ = bmesh.new()
        bm_.from_mesh(me)
        bm_.from_mesh(extra)
        bm_.to_mesh(me)
        bm_.free()
        bpy.data.meshes.remove(extra)
    me.materials.append(M["people"])
    me.materials.append(M["wood"])
    me.materials.append(M["metal"])
    me.materials.append(M["ashlar"])
    return me


def cart_mb(load, rng, phase):
    """Two-wheeled Valencian cart (carro) behind the animal (-y)."""
    mb = MB()
    w = L(150, 112, 74)
    iron = L(58, 58, 60)
    yc = -2.6
    # shafts
    for x in (-0.42, 0.42):
        mb.beam((x, 0.3, 1.05), (x * 1.2, yc + 1.2, 0.98), 0.08, w, 1)
    # bed with side rails and stakes
    mb.box((0, yc, 0.92), (1.5, 2.6, 0.08), w, 1)
    for x in (-0.75, 0.75):
        mb.beam((x, yc - 1.3, 1.25), (x, yc + 1.3, 1.25), 0.06, w, 1)
        for j in range(6):
            yy = yc - 1.2 + 2.4 * j / 5
            mb.beam((x, yy, 0.96), (x, yy, 1.3), 0.05, w, 1)
    mb.beam((-0.8, yc, 0.75), (0.8, yc, 0.75), 0.09, iron, 2)                  # axle
    rot = phase * 0.35
    for x in (-0.85, 0.85):                                                     # wheels
        wc = np.array([x, yc, 0.75])
        r = 0.75
        n = 28
        for k in range(n):
            a0, a1 = 2 * math.pi * k / n, 2 * math.pi * (k + 1) / n
            p0 = wc + np.array([0, r * math.cos(a0), r * math.sin(a0)])
            p1 = wc + np.array([0, r * math.cos(a1), r * math.sin(a1)])
            mb.beam(tuple(p0), tuple(p1), 0.09, w, 1)
            mb.beam(tuple(p0 + [0.0, 0, 0] + (p0 - wc) * 0.06), tuple(p1 + (p1 - wc) * 0.06), 0.06, iron, 2)
        for k in range(12):
            a = 2 * math.pi * k / 12 + rot
            mb.beam(tuple(wc), tuple(wc + np.array([0, r * math.cos(a), r * math.sin(a)])), 0.05, w, 1)
        mb.cyl(tuple(wc - [0.12, 0, 0]), 0.12, 0.24, 10, iron, 2)
        hub = MB()
        hub.cyl((0, 0, -0.14), 0.13, 0.28, 10, w, 1)
        for i_, v in enumerate(hub.v):
            hub.v[i_] = (wc[0] + v[2], wc[1] + v[1], wc[2] + v[0])
        mb.merge(hub)
    if load == "stone":
        for j, (x, y) in enumerate(((-0.35, -0.6), (0.35, -0.6), (0, 0.5))):
            mb.box((x, yc + y, 0.96), (0.62, 0.9, 0.5), L(226, 214, 190), 3, rot=0.05 * j)
    elif load == "earth":
        mb.sphere((0, yc, 1.05), 0.75, n=12, rings=5, col=L(120, 92, 62), mat=0, sz=0.45, noise=0.12,
                  rng=rng)
    elif load == "bricks":
        for j in range(4):
            mb.box((0, yc - 0.9 + j * 0.6, 0.96), (1.2, 0.5, 0.45), L(176, 88, 58), 0)
    elif load == "timber":
        for j in range(7):
            mb.beam(((j - 3) * 0.17, yc - 1.6, 1.0 + (j % 2) * 0.16), ((j - 3) * 0.17, yc + 1.8, 1.0 + (j % 2) * 0.16),
                    0.16, L(170, 136, 92), 1)
    return mb
