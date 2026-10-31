"""Low-poly figures with pose libraries: 1930s workers and townsfolk, mules
and carts, period motor cars, trams, pigeons.  Each figure style gets a set
of pose meshes (walk cycle, work cycle, carry, barrow) that are swapped per
frame on linked-duplicate objects, so motion is fluid while the renderer
only stores a handful of meshes."""
import math
import random

import numpy as np

from .common import MB, L, jitter_col

SKIN = [L(222, 176, 140), L(196, 146, 108), L(170, 120, 86), L(232, 190, 160)]

N_WALK = 8
N_WORK = 6


def _rot(p, piv, a):
    """Rotate about an x-parallel axis through piv; +a tips +z towards +y."""
    y, z = p[1] - piv[1], p[2] - piv[2]
    c, s = math.cos(a), math.sin(a)
    return (p[0], piv[1] + y * c + z * s, piv[2] + z * c - y * s)


def _rotz(p, a):
    c, s = math.cos(a), math.sin(a)
    return (p[0] * c - p[1] * s, p[0] * s + p[1] * c, p[2])


def _box_pts(x0, x1, y0, y1, z0, z1):
    return [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
            (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]


def _emit_box(mb, pts, col, mat=0):
    i = mb.add_verts(pts)
    for f in ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5),
              (2, 3, 7, 6), (3, 0, 4, 7)):
        mb.face(tuple(i + k for k in f), col, mat)


def limb(mb, piv, sx, sy, ln, fwd, col, bend=None, mat=0, side=0.0):
    """Box hanging from pivot; fwd>0 swings the free end towards +y."""
    pts = _box_pts(piv[0] - sx / 2, piv[0] + sx / 2, piv[1] - sy / 2,
                   piv[1] + sy / 2, piv[2] - ln, piv[2])
    pts = [_rot(p, piv, -fwd) for p in pts]
    if side:
        pts = [(piv[0] + (p[0] - piv[0]) + side * (piv[2] - p[2]), p[1], p[2])
               for p in pts]
    if bend is not None:
        pts = [_rot(p, bend[0], bend[1]) for p in pts]
    _emit_box(mb, pts, col, mat)


class Style:
    def __init__(self, kind, shirt, trousers, hat, skin, hat_kind="beret",
                 scale=1.0, skirt=None, hair=None, load=None):
        self.kind = kind
        self.shirt, self.trousers, self.hat, self.skin = shirt, trousers, hat, skin
        self.hat_kind = hat_kind
        self.scale = scale
        self.skirt = skirt
        self.hair = hair
        self.load = load


def person(st, legs=(0, 0), arms=(0, 0), lean=0.0, load=None, barrow=False,
           arm_side=0.0):
    mb = MB()
    s = st.scale
    hip = (0, 0, 0.88 * s)
    bend = (hip, lean) if lean else None
    lw = 0.13 * s
    # legs
    for k, x in enumerate((-0.09 * s, 0.09 * s)):
        limb(mb, (x, 0, 0.88 * s), lw, lw * 1.1, 0.86 * s, legs[k], st.trousers)
        # shoes
        a = legs[k]
        foot = _rot((x, 0.05 * s, 0.03 * s), (x, 0, 0.88 * s), -a)
        _emit_box(mb, _box_pts(foot[0] - 0.06 * s, foot[0] + 0.06 * s,
                               foot[1] - 0.08 * s, foot[1] + 0.1 * s,
                               max(0.0, foot[2] - 0.04 * s), foot[2] + 0.05 * s),
                  L(30, 24, 20))
    if st.skirt is not None:
        n = 8
        r0, r1 = 0.27 * s, 0.17 * s
        z0, z1 = 0.4 * s, 0.98 * s
        b = mb.add_verts([(r0 * math.cos(2 * math.pi * k / n),
                           r0 * math.sin(2 * math.pi * k / n) * 0.85, z0) for k in range(n)])
        t = mb.add_verts([(r1 * math.cos(2 * math.pi * k / n),
                           r1 * math.sin(2 * math.pi * k / n) * 0.8, z1) for k in range(n)])
        for k in range(n):
            mb.face((b + k, b + (k + 1) % n, t + (k + 1) % n, t + k), st.skirt)
        mb.face(tuple(range(b + n - 1, b - 1, -1)), st.skirt)
    # torso (slightly tapered)
    tp = _box_pts(-0.19 * s, 0.19 * s, -0.11 * s, 0.11 * s, 0.86 * s, 1.44 * s)
    tp = [(x * (0.88 if z < 1.0 * s else 1.0), y, z) for x, y, z in tp]
    if bend:
        tp = [_rot(p, *bend) for p in tp]
    _emit_box(mb, tp, st.shirt)
    # arms
    for k, x in enumerate((-0.25 * s, 0.25 * s)):
        limb(mb, (x, 0, 1.40 * s), 0.09 * s, 0.1 * s, 0.62 * s, arms[k],
             st.shirt, bend=bend, side=arm_side * (1 if k else -1))
        # hands
        hnd = _rot((x, 0, 1.40 * s - 0.66 * s), (x, 0, 1.40 * s), -arms[k])
        if bend:
            hnd = _rot(hnd, *bend)
        _emit_box(mb, _box_pts(hnd[0] - 0.045 * s, hnd[0] + 0.045 * s,
                               hnd[1] - 0.045 * s, hnd[1] + 0.045 * s,
                               hnd[2] - 0.03 * s, hnd[2] + 0.07 * s), st.skin)
    # neck + head
    head_c = (0, 0, 1.58 * s)
    if bend:
        head_c = _rot(head_c, *bend)
    sub = MB()
    sub.sphere((0, 0, 0), 0.115 * s, n=8, rings=5, col=st.skin)
    if st.hair is not None:
        sub.sphere((0, -0.02 * s, 0.03 * s), 0.12 * s, n=8, rings=4, col=st.hair, sz=0.9)
    if st.hat_kind == "beret":
        sub.cyl((0, 0, 0.07 * s), 0.13 * s, 0.05 * s, 8, st.hat, r2=0.11 * s)
    elif st.hat_kind == "straw":
        sub.cyl((0, 0, 0.06 * s), 0.24 * s, 0.025 * s, 10, st.hat)
        sub.cyl((0, 0, 0.08 * s), 0.12 * s, 0.09 * s, 8, st.hat)
    elif st.hat_kind == "fedora":
        sub.cyl((0, 0, 0.06 * s), 0.19 * s, 0.02 * s, 10, st.hat)
        sub.cyl((0, 0, 0.07 * s), 0.115 * s, 0.11 * s, 8, st.hat, r2=0.1 * s)
    elif st.hat_kind == "cloche":
        sub.dome((0, 0, 0.02 * s), 0.14 * s, n=8, rings=3, col=st.hat)
    elif st.hat_kind == "cap":
        sub.cyl((0, 0, 0.06 * s), 0.125 * s, 0.06 * s, 8, st.hat)
        sub.box((0, 0.12 * s, 0.06 * s), (0.18 * s, 0.12 * s, 0.02 * s), st.hat)
    mb.merge(sub, offset=head_c)
    # loads
    if load == "shoulder":
        c = (0.12 * s, 0.0, 1.52 * s)
        _emit_box(mb, _box_pts(c[0] - 0.2, c[0] + 0.2, c[1] - 0.35, c[1] + 0.35,
                               c[2], c[2] + 0.28), L(205, 190, 160))
    elif load == "hod":   # bricks on the shoulder
        c = (0.14 * s, -0.05, 1.48 * s)
        _emit_box(mb, _box_pts(c[0] - 0.14, c[0] + 0.14, c[1] - 0.3, c[1] + 0.3,
                               c[2], c[2] + 0.22), L(170, 86, 56))
    elif load == "plank":
        c = (0.15 * s, 0.0, 1.55 * s)
        _emit_box(mb, _box_pts(c[0] - 0.12, c[0] + 0.12, c[1] - 1.8, c[1] + 1.8,
                               c[2], c[2] + 0.06), L(170, 140, 96), 1)
    elif load == "plan":
        hnd = (0.0, 0.42 * s, 1.08 * s)
        _emit_box(mb, _box_pts(-0.32, 0.32, hnd[1] - 0.05, hnd[1] + 0.35,
                               hnd[2], hnd[2] + 0.015), L(236, 230, 210))
    elif load == "basket":
        mb.cyl((0.0, 0.0, 1.70 * s), 0.25, 0.18, 8, L(180, 140, 80), r2=0.3,
               z_top_col=L(200, 60, 70))
    if barrow:
        w = 1
        bm = MB()
        bm.box((0, 0.95, 0.35), (0.62, 0.85, 0.32), L(140, 110, 80), w)
        bm.box((0, 0.95, 0.66), (0.5, 0.7, 0.06), L(120, 90, 62), w)  # earth
        for x in (-0.2, 0.2):
            bm.beam((x, 0.3, 0.72), (x, 1.5, 0.3), 0.06, L(120, 96, 70), w)
        bm.cyl((0, 1.5, 0.0), 0.2, 0.08, 8, L(60, 50, 40), w)
        mb.merge(bm)
    return mb


def make_styles(rng):
    S = []
    blue = [L(52, 74, 112), L(70, 92, 128), L(44, 60, 92)]
    for k in range(10):
        sk = SKIN[k % len(SKIN)]
        r = k % 5
        if r == 0:
            S.append(Style("worker", blue[k % 3], blue[(k + 1) % 3], L(28, 26, 26), sk, "beret"))
        elif r == 1:
            S.append(Style("worker", L(226, 220, 204), L(92, 72, 52), L(210, 186, 128), sk, "straw"))
        elif r == 2:
            S.append(Style("worker", L(150, 146, 136), L(56, 66, 96), L(70, 66, 60), sk, "cap"))
        elif r == 3:
            S.append(Style("worker", L(210, 196, 166), L(64, 78, 104), L(28, 26, 26), sk, "beret"))
        else:
            S.append(Style("worker", L(120, 78, 56), L(70, 64, 58), L(210, 186, 128), sk, "straw"))
    W = len(S)
    suits = [L(36, 36, 42), L(60, 56, 52), L(198, 188, 160), L(110, 104, 92), L(46, 52, 70)]
    hats = [L(30, 30, 30), L(214, 196, 150), L(80, 74, 66)]
    for k in range(10):
        S.append(Style("man", suits[k % 5], jitter_col(suits[k % 5], rng, 0.1),
                       hats[k % 3], SKIN[k % 4], "fedora" if k % 3 != 1 else "straw"))
    dresses = [L(150, 40, 52), L(60, 82, 124), L(226, 216, 196), L(92, 112, 72),
               L(186, 146, 96), L(40, 40, 44), L(196, 120, 130), L(110, 70, 110)]
    for k in range(10):
        d = dresses[k % len(dresses)]
        S.append(Style("woman", d, SKIN[(k + 1) % 4], jitter_col(d, rng, 0.3),
                       SKIN[(k + 1) % 4], "cloche", skirt=d,
                       hair=L(40, 28, 20) if k % 2 else L(80, 56, 34)))
    for k in range(4):
        S.append(Style("child", [L(200, 60, 60), L(230, 220, 200), L(60, 90, 140), L(220, 180, 70)][k],
                       L(70, 70, 80), L(30, 30, 30), SKIN[k], "cap", scale=0.68))
    S.append(Style("foreman", L(70, 64, 56), L(60, 56, 50), L(40, 38, 36), SKIN[0], "fedora"))
    return S, W


def build_pose_library(rng):
    """Returns styles and dict (style_idx, pose) -> MB."""
    styles, n_workers = make_styles(rng)
    lib = {}
    for si, st in enumerate(styles):
        for k in range(N_WALK):
            ph = 2 * math.pi * k / N_WALK
            a = 0.42 * math.sin(ph)
            lib[(si, "walk", k)] = person(st, legs=(a, -a), arms=(-0.45 * a, 0.45 * a))
            if st.kind == "worker":
                lib[(si, "hod", k)] = person(st, legs=(a, -a), arms=(-0.4 * a, 2.6),
                                             load="hod", arm_side=0.1)
                lib[(si, "plank", k)] = person(st, legs=(a, -a), arms=(-0.4 * a, 2.7),
                                               load="plank")
                lib[(si, "barrow", k)] = person(st, legs=(0.8 * a, -0.8 * a),
                                                arms=(0.55, 0.55), lean=0.12,
                                                barrow=True)
            if st.kind == "woman" and si % 3 == 0:
                lib[(si, "basket", k)] = person(st, legs=(a * 0.7, -a * 0.7),
                                                arms=(2.9, -0.3 * a), load="basket")
        lib[(si, "stand", 0)] = person(st, arms=(0.05, -0.05))
        if st.kind in ("worker", "foreman"):
            for k in range(N_WORK):
                ph = 2 * math.pi * k / N_WORK
                sw = 0.5 + 0.75 * (0.5 + 0.5 * math.sin(ph))
                lean = 0.35 + 0.15 * math.sin(ph + 0.6)
                lib[(si, "work", k)] = person(st, legs=(0.2, -0.15),
                                              arms=(sw, sw * 0.8 + 0.1), lean=lean)
                # shovelling / lifting variant
                sw2 = 0.2 + 1.1 * (0.5 + 0.5 * math.sin(ph))
                lib[(si, "dig", k)] = person(st, legs=(0.3, -0.2),
                                             arms=(sw2, sw2), lean=0.25 + 0.4 * math.sin(ph) ** 2)
        if st.kind == "foreman":
            lib[(si, "plan", 0)] = person(st, arms=(0.9, 0.9), load="plan")
            for k in range(N_WORK):
                ph = 2 * math.pi * k / N_WORK
                lib[(si, "point", k)] = person(st, arms=(0.4, 1.9 + 0.5 * math.sin(ph)),
                                               arm_side=0.15 * math.sin(ph))
    return styles, n_workers, lib


# ---------------------------------------------------------------------------
# Animals and vehicles (all face +y)
# ---------------------------------------------------------------------------
def horse(col, legs_phase=0.0, cart=None, rng=None):
    mb = MB()
    _emit_box(mb, _box_pts(-0.27, 0.27, -0.85, 0.85, 0.95, 1.55), col)
    mb.beam((0, 0.75, 1.4), (0, 1.15, 2.0), 0.28, col)
    _emit_box(mb, _box_pts(-0.12, 0.12, 1.0, 1.55, 1.85, 2.1), col)
    mb.beam((0, -0.85, 1.45), (0, -1.15, 0.95), 0.1, L(40, 32, 26))
    for k, (x, y) in enumerate(((-0.17, 0.7), (0.17, 0.7), (-0.17, -0.7), (0.17, -0.7))):
        ph = legs_phase + (0 if k in (0, 3) else math.pi)
        limb(mb, (x, y, 1.0), 0.11, 0.12, 1.0, 0.35 * math.sin(ph), col)
    if cart is not None:
        w = 1
        mb.beam((-0.4, 0.6, 1.1), (-0.5, -2.3, 0.95), 0.08, L(120, 96, 70), w)
        mb.beam((0.4, 0.6, 1.1), (0.5, -2.3, 0.95), 0.08, L(120, 96, 70), w)
        mb.box((0, -3.2, 0.85), (1.6, 2.4, 0.5), L(130, 100, 70), w)
        for x in (-0.86, 0.86):
            wb = MB()
            wb.cyl((0, 0, 0), 0.75, 0.08, 12, L(70, 56, 44), w)
            for i, v in enumerate(wb.v):
                wb.v[i] = (x + v[2] * (1 if x > 0 else -1), -3.2 + v[1], 0.75 + v[0])
            mb.merge(wb)
        if cart == "earth":
            mb.sphere((0, -3.2, 1.35), 0.75, n=8, rings=4, col=L(120, 92, 62), sz=0.45,
                      noise=0.15, rng=rng)
        elif cart == "bricks":
            for i in range(3):
                mb.box((0, -3.9 + i * 0.7, 1.35), (1.3, 0.6, 0.45), L(176, 88, 58))
        elif cart == "barrels":
            for i in range(2):
                mb.cyl((0, -3.6 + i * 0.8, 1.35), 0.35, 0.8, 8, L(110, 80, 52))
        # driver
        dr = person(Style("man", L(52, 72, 110), L(52, 72, 110), L(28, 26, 26), SKIN[1]),
                    legs=(1.2, 1.2), arms=(0.9, 0.9))
        mb.merge(dr, offset=(0, -2.1, 0.55))
    return mb


def car(col, rng):
    mb = MB()
    m = 1  # paint material slot
    dark = L(22, 22, 24)
    mb.box((0, 0, 0.35), (1.62, 4.0, 0.5), col, m)                 # chassis/body
    mb.box((0, 1.25, 0.85), (1.0, 1.4, 0.32), col, m)              # bonnet
    mb.box((0, -0.45, 0.85), (1.5, 2.1, 0.8), col, m)              # cabin
    mb.box((0, -0.45, 1.65), (1.42, 2.0, 0.08), dark, m)           # roof
    mb.box((0, 0.6, 0.95), (1.46, 0.06, 0.65), L(40, 52, 60), m)   # windscreen
    mb.box((0, 2.02, 0.45), (0.9, 0.06, 0.55), L(160, 150, 120), m)  # grille
    for x in (-0.82, 0.82):
        mb.box((x, 0, 0.3), (0.28, 4.3, 0.12), dark, m)            # running boards
        for y in (-1.35, 1.35):
            wb = MB()
            wb.cyl((0, 0, 0), 0.38, 0.22, 10, dark, m)
            for i, v in enumerate(wb.v):
                wb.v[i] = (x + v[2] * (1 if x > 0 else -1), y + v[1], 0.38 + v[0])
            mb.merge(wb)
    mb.box((0.0, -0.45, 0.9), (1.52, 1.6, 0.42), L(30, 34, 38), m)  # side windows band
    return mb


def tram(rng):
    mb = MB()
    m = 1
    low = L(196, 150, 60)
    up = L(232, 220, 186)
    trim = L(90, 56, 36)
    mb.box((0, 0, 0.35), (2.2, 9.0, 0.6), trim, m)
    mb.box((0, 0, 0.95), (2.25, 7.4, 1.0), low, m)
    mb.box((0, 0, 1.95), (2.2, 7.4, 0.9), L(30, 34, 38), m)    # window band
    for y in np.linspace(-3.4, 3.4, 8):
        mb.box((0, y, 1.95), (2.28, 0.16, 0.9), up, m)          # pillars
    mb.box((0, 0, 2.85), (2.3, 7.6, 0.25), up, m)
    mb.box((0, 0, 3.1), (1.4, 6.6, 0.25), L(80, 70, 60), m)    # clerestory
    for s_ in (-1, 1):                                          # platforms
        mb.box((0, s_ * 4.1, 0.95), (2.1, 0.9, 1.9), up, m)
        mb.box((0, s_ * 4.1, 2.85), (2.1, 1.1, 0.2), up, m)
    mb.beam((0, -2.0, 3.35), (0, 2.2, 6.0), 0.07, L(40, 40, 40), m)  # trolley pole
    mb.box((0, -2.0, 3.3), (0.4, 0.5, 0.12), L(40, 40, 40), m)
    return mb


def pigeon(up):
    mb = MB()
    g = L(110, 112, 120)
    _emit_box(mb, _box_pts(-0.07, 0.07, -0.16, 0.16, -0.05, 0.05), g)
    a = 0.6 if up else -0.4
    for s_ in (-1, 1):
        x1 = s_ * 0.32
        i = mb.add_verts([(s_ * 0.05, -0.08, 0), (s_ * 0.05, 0.08, 0),
                          (x1, 0.04, 0.3 * math.sin(a) * 1.0), (x1, -0.1, 0.3 * math.sin(a))])
        mb.face((i, i + 1, i + 2, i + 3), L(150, 150, 160))
    return mb
