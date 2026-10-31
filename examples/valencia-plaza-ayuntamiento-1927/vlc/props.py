"""Construction tools and site equipment of the 1910s, modelled for
close-up: stonecutters' bankers with mallets, chisels and squares; mortar
troughs with hoes; buckets, sieves, ladders, trestles with a frame saw,
rope coils, pulley blocks, a geared hand winch, water barrels, baskets,
lime box, barrows, half-dressed ashlar blocks.

Material slots used by every prop: 0 wood, 1 iron, 2 stone (ashlar),
3 rope/fabric (people material, vertex colour), 4 earth."""
import math

import numpy as np

from .common import MB, L, jitter_col

WOOD = L(150, 112, 74)
DARKW = L(110, 80, 52)
IRON = L(62, 62, 64)
ROPE = L(196, 176, 132)
STONE = L(226, 214, 190)


def _rotz(mb, ang, origin=(0, 0, 0)):
    ca, sa = math.cos(ang), math.sin(ang)
    ox, oy, oz = origin
    mb.v = [(ox + x * ca - y * sa, oy + x * sa + y * ca, oz + z) for x, y, z in mb.v]
    return mb


def mallet(mb, p, ang=0.0):
    sub = MB()
    sub.cyl((0, 0, 0), 0.05, 0.13, 10, DARKW, 0)          # maceta head lying
    for i, v in enumerate(sub.v):
        sub.v[i] = (v[2] - 0.065, v[1], v[0] + 0.05)
    sub.beam((0.0, 0, 0.05), (0.0, 0.28, 0.04), 0.022, WOOD, 0)
    mb.merge(_rotz(sub, ang), offset=p)


def chisel(mb, p, ang=0.0, length=0.24):
    sub = MB()
    sub.beam((0, 0, 0.012), (0, length, 0.012), 0.016, IRON, 1)
    sub.beam((0, length, 0.012), (0, length + 0.02, 0.012), 0.03, IRON, 1)
    mb.merge(_rotz(sub, ang), offset=p)


def square(mb, p, ang=0.0):
    sub = MB()
    sub.box((0.15, 0, 0), (0.3, 0.025, 0.006), IRON, 1)
    sub.box((0, 0.1, 0), (0.025, 0.2, 0.006), IRON, 1)
    mb.merge(_rotz(sub, ang), offset=p)


def banker(mb, p, ang, rng, with_block=True):
    """Stonecutter's banker: timber trestle bench with a block being dressed."""
    sub = MB()
    for x in (-0.55, 0.55):
        for y in (-0.3, 0.3):
            sub.beam((x, y, 0), (x * 0.9, y * 0.8, 0.7), 0.08, WOOD, 0)
    sub.box((0, 0, 0.7), (1.35, 0.8, 0.1), DARKW, 0)
    if with_block:
        sub.box((0, 0, 0.8), (0.9, 0.55, 0.45), STONE, 2, rot=rng.uniform(-0.1, 0.1))
        # rough, not yet dressed half
        sub.sphere((0.3, 0.05, 1.12), 0.22, n=7, rings=4, col=STONE, mat=2, sz=0.45, noise=0.25, rng=rng)
    mallet(sub, (-0.55, -0.32, 0.8), 0.3)
    chisel(sub, (0.45, -0.34, 0.8), 1.4)
    chisel(sub, (0.52, -0.3, 0.8), 1.2, 0.2)
    square(sub, (-0.2, 0.3, 0.8), 0.2)
    mb.merge(_rotz(sub, ang), offset=p)


def mortar_trough(mb, p, ang=0.0):
    sub = MB()
    bot = [(-0.6, -0.3), (0.6, -0.3), (0.6, 0.3), (-0.6, 0.3)]
    top = [(-0.75, -0.42), (0.75, -0.42), (0.75, 0.42), (-0.75, 0.42)]
    for k in range(4):
        a, b = bot[k], bot[(k + 1) % 4]
        c, d = top[(k + 1) % 4], top[k]
        i = sub.add_verts([(*a, 0.05), (*b, 0.05), (*c, 0.38), (*d, 0.38)])
        sub.face((i, i + 1, i + 2, i + 3), DARKW, 0)
    i = sub.add_verts([(x * 0.93, y * 0.9, 0.3) for x, y in top])
    sub.face((i, i + 1, i + 2, i + 3), L(196, 190, 176), 4)           # mortar surface
    # mixing hoe (batidera) leaning in
    sub.beam((0.4, 0.0, 0.32), (1.4, 0.4, 1.3), 0.035, WOOD, 0)
    sub.box((0.35, 0.0, 0.3), (0.22, 0.03, 0.14), IRON, 1, rot=0.4)
    mb.merge(_rotz(sub, ang), offset=p)


def bucket(mb, p, full=True):
    sub = MB()
    sub.cyl((0, 0, 0), 0.13, 0.28, 14, DARKW, 0, r2=0.16)
    for z in (0.04, 0.24):
        sub.cyl((0, 0, z), 0.135 + z * 0.1, 0.025, 14, IRON, 1, r2=0.137 + z * 0.1)
    if full:
        sub.cyl((0, 0, 0.24), 0.15, 0.01, 14, L(190, 186, 176), 4)
    for k in range(9):
        a0, a1 = math.pi * k / 9, math.pi * (k + 1) / 9
        sub.beam((0.16 * math.cos(a0), 0, 0.28 + 0.16 * math.sin(a0)),
                 (0.16 * math.cos(a1), 0, 0.28 + 0.16 * math.sin(a1)), 0.012, IRON, 1)
    mb.merge(sub, offset=p)


def ladder(mb, p0, p1, width=0.5):
    p0, p1 = np.array(p0, float), np.array(p1, float)
    d = p1 - p0
    ln = np.linalg.norm(d)
    side = np.cross(d, [0, 0, 1])
    side = side / (np.linalg.norm(side) + 1e-9) * width / 2
    for s in (-1, 1):
        mb.beam(tuple(p0 + side * s), tuple(p1 + side * s), 0.06, WOOD, 0)
    for k in range(int(ln / 0.3)):
        q = p0 + d * (k + 0.5) / (ln / 0.3)
        mb.beam(tuple(q - side), tuple(q + side), 0.035, WOOD, 0)


def rope_coil(mb, p, r=0.32, turns=5):
    for t in range(turns):
        z = p[2] + 0.035 + t * 0.045
        rr = r - t * 0.02
        n = 18
        for k in range(n):
            a0, a1 = 2 * math.pi * k / n, 2 * math.pi * (k + 1) / n
            mb.beam((p[0] + rr * math.cos(a0), p[1] + rr * math.sin(a0), z),
                    (p[0] + rr * math.cos(a1), p[1] + rr * math.sin(a1), z), 0.04, ROPE, 3)


def pulley_block(mb, p, size=0.3):
    """Timber shell with iron sheave and hook (motón)."""
    mb.box((p[0], p[1], p[2] - size), (0.14, size * 0.9, size * 1.4), DARKW, 0)
    sub = MB()
    sub.cyl((0, 0, -0.04), size * 0.38, 0.08, 16, IRON, 1)
    for i, v in enumerate(sub.v):
        sub.v[i] = (p[0] + v[2] + 0.075, p[1] + v[1], p[2] - size * 0.3 + v[0])
    mb.merge(sub)
    for k in range(8):
        a0, a1 = math.pi * 1.1 * k / 8, math.pi * 1.1 * (k + 1) / 8
        mb.beam((p[0], p[1] + 0.06 * math.cos(a0), p[2] - size * 1.05 - 0.06 * math.sin(a0)),
                (p[0], p[1] + 0.06 * math.cos(a1), p[2] - size * 1.05 - 0.06 * math.sin(a1)), 0.025, IRON, 1)


def gear(mb, c, r, teeth, axis_y=True, width=0.06, col=IRON, phase=0.0):
    """Spur gear in the plane perpendicular to x (axis along x)."""
    n = teeth
    sub = MB()
    sub.cyl((0, 0, -width / 2), r, width, n, col, 1)
    for k in range(n):
        a = 2 * math.pi * k / n + phase
        sub.box((0, 0, 0), (0.0001, 0.0001, 0.0001), col, 1)
        q = ((r + 0.02) * math.cos(a), (r + 0.02) * math.sin(a))
        sub.box((q[0], q[1], -width / 2), (0.05, 0.035, width), col, 1, rot=a)
    for i, v in enumerate(sub.v):
        sub.v[i] = (c[0] + v[2], c[1] + v[0], c[2] + v[1])
    mb.merge(sub)


def winch_frame(mb, p, ang=0.0):
    """Static part of a geared hand winch (torno): trestle frame."""
    sub = MB()
    for x in (-0.6, 0.6):
        sub.beam((x, -0.5, 0), (x, 0.0, 1.0), 0.12, WOOD, 0)
        sub.beam((x, 0.5, 0), (x, 0.0, 1.0), 0.12, WOOD, 0)
        sub.beam((x, -0.55, 0.05), (x, 0.55, 0.05), 0.12, WOOD, 0)
    sub.beam((-0.6, 0, 1.0), (0.6, 0, 1.0), 0.1, IRON, 1)
    mb.merge(_rotz(sub, ang), offset=p)


def winch_drum(phase):
    """Rotating part: drum with rope turns, gears and crank handles."""
    sub = MB()
    drum = MB()
    drum.cyl((0, 0, -0.5), 0.2, 1.0, 14, DARKW, 0)
    for i, v in enumerate(drum.v):
        drum.v[i] = (v[2], v[0], v[1])
    sub.merge(drum)
    for k in range(10):     # rope turns
        x = -0.4 + 0.08 * k
        n = 12
        for j in range(n):
            a0, a1 = 2 * math.pi * j / n + phase, 2 * math.pi * (j + 1) / n + phase
            sub.beam((x, 0.22 * math.cos(a0), 0.22 * math.sin(a0)), (x + 0.008, 0.22 * math.cos(a1), 0.22 * math.sin(a1)),
                     0.035, ROPE, 3)
    gear(sub, (0.56, 0, 0), 0.42, 30, phase=phase)
    gear(sub, (0.66, 0, 0.55), 0.12, 9, phase=-phase * 3.5)
    for s in (-1, 1):       # crank handles
        a = -phase * 3.5 + (0 if s > 0 else math.pi)
        c = np.array([0.72 if s > 0 else -0.72, 0, 0.55])
        e = c + np.array([0, 0.35 * math.cos(a), 0.35 * math.sin(a)])
        sub.beam(tuple(c), tuple(e), 0.05, IRON, 1)
        sub.beam(tuple(e), tuple(e + np.array([0.25 * s, 0, 0])), 0.05, WOOD, 0)
    return sub


def trestles_with_saw(mb, p, ang=0.0):
    sub = MB()
    for x in (-0.9, 0.9):
        for y in (-0.25, 0.25):
            sub.beam((x, y, 0), (x, 0, 0.75), 0.07, WOOD, 0)
        sub.beam((x, -0.3, 0.75), (x, 0.3, 0.75), 0.1, WOOD, 0)
    sub.beam((-1.5, 0, 0.83), (1.5, 0, 0.83), 0.18, L(176, 140, 96), 0)
    # frame saw resting on the beam
    sub.beam((0.1, -0.2, 0.95), (0.9, 0.2, 0.95), 0.02, IRON, 1)
    sub.beam((0.1, -0.2, 0.95), (0.1, -0.2, 1.35), 0.04, WOOD, 0)
    sub.beam((0.9, 0.2, 0.95), (0.9, 0.2, 1.35), 0.04, WOOD, 0)
    sub.beam((0.1, -0.2, 1.15), (0.9, 0.2, 1.15), 0.035, WOOD, 0)
    sub.beam((0.1, -0.2, 1.35), (0.9, 0.2, 1.35), 0.01, ROPE, 3)
    for k in range(10):     # sawdust
        sub.sphere((rng_off(k) * 0.6, rng_off(k + 3) * 0.4, 0.0), 0.06, n=6, rings=3, col=L(214, 190, 140),
                   mat=4, sz=0.3)
    mb.merge(_rotz(sub, ang), offset=p)


def rng_off(k):
    return math.sin(k * 12.9898) % 1.0 - 0.5


def barrel(mb, p, full=True):
    sub = MB()
    rows = []
    n = 16
    for j in range(7):
        z = 0.9 * j / 6
        r = 0.27 + 0.05 * math.sin(math.pi * j / 6)
        rows.append([(r * math.cos(2 * math.pi * k / n), r * math.sin(2 * math.pi * k / n), z) for k in range(n)])
    for j in range(6):
        b = sub.add_verts(rows[j] + rows[j + 1])
        for k in range(n):
            k2 = (k + 1) % n
            sub.face((b + k, b + k2, b + n + k2, b + n + k), WOOD, 0,
                     [(k / n * 1.7, rows[j][0][2]), ((k + 1) / n * 1.7, rows[j][0][2]),
                      ((k + 1) / n * 1.7, rows[j + 1][0][2]), (k / n * 1.7, rows[j + 1][0][2])])
    for z, r in ((0.12, 0.29), (0.45, 0.325), (0.78, 0.29)):
        sub.cyl((0, 0, z - 0.03), r + 0.005, 0.06, n, IRON, 1)
    sub.cyl((0, 0, 0.84), 0.27, 0.02, n, L(60, 80, 90) if full else WOOD, 4)
    mb.merge(sub, offset=p)


def basket(mb, p, col=L(186, 150, 96), fill=None):
    mb.cyl(p, 0.2, 0.22, 16, col, 3, r2=0.26)
    if fill is not None:
        mb.cyl((p[0], p[1], p[2] + 0.2), 0.24, 0.02, 16, fill, 4)
    for k in range(9):
        a0, a1 = math.pi * k / 9, math.pi * (k + 1) / 9
        mb.beam((p[0] + 0.25 * math.cos(a0), p[1], p[2] + 0.22 + 0.2 * math.sin(a0)),
                (p[0] + 0.25 * math.cos(a1), p[1], p[2] + 0.22 + 0.2 * math.sin(a1)), 0.025, col, 3)


def sieve(mb, p, ang):
    """Riddle for sand, leaning on two legs."""
    sub = MB()
    for k in range(16):
        a0, a1 = 2 * math.pi * k / 16, 2 * math.pi * (k + 1) / 16
        sub.beam((0.4 * math.cos(a0), 0, 0.42 + 0.4 * math.sin(a0)), (0.4 * math.cos(a1), 0, 0.42 + 0.4 * math.sin(a1)),
                 0.06, WOOD, 0)
    for k in range(-5, 6):
        x = k * 0.07
        h = math.sqrt(max(0, 0.16 - x * x))
        sub.beam((x, 0, 0.42 - h), (x, 0, 0.42 + h), 0.008, IRON, 1)
        sub.beam((-h, 0, 0.42 + x), (h, 0, 0.42 + x), 0.008, IRON, 1)
    sub.beam((0.3, 0.1, 0.2), (0.4, -0.5, 0.0), 0.04, WOOD, 0)
    sub.beam((-0.3, 0.1, 0.2), (-0.4, -0.5, 0.0), 0.04, WOOD, 0)
    mb.merge(_rotz(sub, ang), offset=p)


def ashlar_block(mb, p, size, ang, rng, rough=False):
    mb.box(p, size, jitter_col(STONE, rng, 0.05), 2, rot=ang)
    if rough:
        mb.sphere((p[0], p[1], p[2] + size[2]), min(size[:2]) * 0.45, n=8, rings=4, col=STONE, mat=2,
                  sz=0.35, noise=0.25, rng=rng)


def tool_rack(mb, p, ang):
    """Picks, shovels and a mattock leaning on a trestle."""
    sub = MB()
    sub.beam((-1.0, 0, 0.9), (1.0, 0, 0.9), 0.08, WOOD, 0)
    for x in (-1.0, 1.0):
        sub.beam((x, -0.3, 0), (x, 0, 0.9), 0.07, WOOD, 0)
        sub.beam((x, 0.3, 0), (x, 0, 0.9), 0.07, WOOD, 0)
    for k in range(6):
        x = -0.8 + k * 0.32
        top = (x, 0.05, 0.98)
        bot = (x + 0.1, 0.65, 0.0)
        sub.beam(top, bot, 0.035, WOOD, 0)
        if k % 2 == 0:        # shovel blade at the bottom
            sub.box((bot[0], bot[1] - 0.05, 0.0), (0.22, 0.04, 0.28), IRON, 1, rot=0.0)
        else:                 # pick head at the top
            sub.beam((top[0] - 0.3, top[1], top[2] + 0.05), (top[0] + 0.3, top[1], top[2] + 0.02), 0.04, IRON, 1)
    mb.merge(_rotz(sub, ang), offset=p)


def sign_board(mb, p, ang):
    sub = MB()
    sub.beam((-1.4, 0, 0), (-1.4, 0, 2.6), 0.1, DARKW, 0)
    sub.beam((1.4, 0, 0), (1.4, 0, 2.6), 0.1, DARKW, 0)
    i = sub.add_verts([(-1.5, -0.06, 1.7), (1.5, -0.06, 1.7), (1.5, -0.06, 2.55), (-1.5, -0.06, 2.55)])
    sub.face((i, i + 1, i + 2, i + 3), (1, 1, 1), 5, [(0, 0), (1, 0), (1, 1), (0, 1)])
    mb.merge(_rotz(sub, ang), offset=p)
