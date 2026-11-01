"""The works on Goerlich's central platform, driven by a progress value p in
[0, 1]:

  0.00-0.20  excavation finished by hand, spoil carted away, floor slab poured
  0.10-0.42  brick retaining walls rise course by course, patio ring, piers
  0.15-0.64  timber derricks (cabrias) swing stone and panels on hemp ropes
  0.42-0.70  deck panels set from the wide end towards the narrow end
  0.58-0.80  stone cladding, stairs, balustrades
  0.66-0.86  paving laid, parterres planted, flowers bloom
  0.78-0.95  trees, palms, lamps, kiosks, fountain; hoarding comes down
"""
import math
import random

import numpy as np
from mathutils import Vector
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union

from . import plan
from .city import top_face, shp_prism, polys_of, largest, tree_mesh, lamp
from .common import MB, L, jitter_col, new_coll, obj_from_mesh, ramp, settle, smooth, clamp

ISL = plan.ISLAND
LEN = plan.X_MAX - plan.X_MIN
CRANES = [((60.0, -14.5), 22.0), ((12.0, 9.0), 21.0), ((-32.0, -3.5), 18.0)]
COMPOUND = box(89.0, -24.0, 99.0, 24.0)


def deck_front_x(p):
    return plan.X_MAX - LEN * clamp((p - 0.42) / 0.24)


def wall_front_x(p):
    return plan.X_MAX - LEN * clamp((p - 0.12) / 0.28)


def boundary_at(x, side):
    """Point on the island boundary at abscissa x on side +1 (N) / -1 (S)."""
    x = min(max(x, plan.X_MIN + 0.3), plan.X_MAX - 0.3)
    from shapely.geometry import LineString
    ln = LineString([(x, plan.axis_y(x)), (x, plan.axis_y(x) + 60 * side)])
    q = ISL.exterior.intersection(ln)
    if q.is_empty:
        return np.array([x, plan.axis_y(x)])
    if q.geom_type != "Point":
        q = list(q.geoms)[0]
    return np.array([q.x, q.y])


class Item:
    __slots__ = ("ob", "t0", "t1", "out0", "out1", "mode", "drop", "base_z",
                 "scale_axis", "base_scale")

    def __init__(self, ob, t0, t1, mode="fade", drop=0.0, out0=None, out1=None,
                 scale_axis=None):
        self.ob = ob
        self.t0, self.t1 = t0, t1
        self.out0, self.out1 = out0, out1
        self.mode = mode
        self.drop = drop
        self.base_z = ob.location.z
        self.scale_axis = scale_axis
        self.base_scale = tuple(ob.scale)


class Site:
    def __init__(self, M, rng, coll=None):
        self.M = M
        self.rng = rng
        self.coll = coll or new_coll("site")
        self.items = []
        self.dust_sources = []        # (x, y, z, t0, t1) for the post pass

    def add(self, mb, name, mats, t0, t1, mode="fade", drop=0.0, out=None, origin=None,
            scale_axis=None, local=False):
        if origin is not None and not local:
            mb.v = [(x - origin[0], y - origin[1], z - origin[2]) for x, y, z in mb.v]
        ob = mb.build(name, mats, self.coll, location=origin or (0, 0, 0))
        it = Item(ob, t0, t1, mode, drop, *(out or (None, None)), scale_axis=scale_axis)
        self.items.append(it)
        return it

    # ---------------------------------------------------------------- static
    def pit(self):
        mb = MB()
        c = list(ISL.exterior.coords)[:-1]
        u = 0
        for k in range(len(c)):
            a, b = c[k], c[(k + 1) % len(c)]
            u += mb.quad_wall(b, a, plan.PIT_Z, 0.0, L(150, 116, 80), 0, u)
        top_face(mb, ISL, plan.PIT_Z, L(150, 116, 80), 0)
        self.pit_obj = mb.build("pit", [self.M["earth"]], self.coll)

    # ---------------------------------------------------------------- phases
    def spoil_and_ramp(self):
        rng = self.rng
        for k, (x, y, r) in enumerate(((-20, 2, 7), (20, -6, 8), (40, 10, 6), (-45, -2, 5))):
            mb = MB()
            mb.sphere((0, 0, 0), r, n=14, rings=7, col=L(140, 106, 72), sz=0.38,
                      noise=0.12, rng=rng)
            it = self.add(mb, "spoil%d" % k, [self.M["earth"]], 0, 0, mode="shrink",
                          origin=(x, y, plan.PIT_Z), local=True)
            it.out0, it.out1 = 0.02 + 0.03 * k, 0.16 + 0.02 * k
        # earth ramp at the narrow end (removed before the last wall closes)
        mb = MB()
        x0 = plan.X_MIN + 0.5
        x1 = x0 + 20
        y0 = plan.C2[1]
        i = mb.add_verts([(x0, y0 - 2.4, 0.0), (x0, y0 + 2.4, 0.0),
                          (x1, y0 + 2.4, plan.PIT_Z), (x1, y0 - 2.4, plan.PIT_Z),
                          (x0, y0 - 2.4, plan.PIT_Z), (x0, y0 + 2.4, plan.PIT_Z)])
        mb.face((i, i + 1, i + 2, i + 3), L(150, 116, 80))
        mb.face((i, i + 3, i + 4), L(130, 100, 70))
        mb.face((i + 1, i + 5, i + 2), L(130, 100, 70))
        for k in range(6):   # plank runway
            t = k / 6
            mb.box((x0 + 2 + t * 16, y0, -t * 4.0 + 0.05 - 0.8 * t), (2.6, 1.0, 0.06),
                   L(160, 130, 90), 1, rot=0.0)
        self.add(mb, "ramp", [self.M["earth"], self.M["wood"]], -1, -1,
                 out=(0.36, 0.40))

    def floor_slab(self):
        rng = self.rng
        inner = ISL.buffer(-0.6)
        minx, miny, maxx, maxy = inner.bounds
        s = 5.0
        for ix in range(int((maxx - minx) / s) + 1):
            for iy in range(int((maxy - miny) / s) + 1):
                cell = box(minx + ix * s, miny + iy * s, minx + (ix + 1) * s, miny + (iy + 1) * s)
                g = largest(cell.intersection(inner))
                if g is None or g.area < 1:
                    continue
                mb = MB()
                shp_prism(mb, g, plan.PIT_Z - 0.2, plan.PIT_Z + 0.12,
                          jitter_col(L(160, 154, 140), rng, 0.05), 0, top_mat=0)
                cx = g.centroid.x
                t0 = 0.03 + 0.15 * (plan.X_MAX - cx) / LEN + rng.uniform(-0.01, 0.015)
                self.add(mb, "slab", [self.M["plain_fade"]], t0, t0 + 0.02, mode="rise",
                         drop=0.32, origin=(g.centroid.x, g.centroid.y, plan.PIT_Z))

    def walls(self):
        rng = self.rng
        ring = ISL.exterior
        n = int(ring.length / 3.0)
        H = 0.6 - plan.PIT_Z
        self.wall_segs = []
        for k in range(n):
            a = ring.interpolate(ring.length * k / n)
            b = ring.interpolate(ring.length * (k + 1) / n)
            pa, pb = np.array([a.x, a.y]), np.array([b.x, b.y])
            d = pb - pa
            d /= np.linalg.norm(d)
            inw = np.array([-d[1], d[0]])
            if not ISL.contains(Point(*(0.5 * (pa + pb) + inw * 1.0))):
                inw = -inw
            poly = [tuple(pa), tuple(pb), tuple(pb + inw * 0.6), tuple(pa + inw * 0.6)]
            mb = MB()
            mb.prism(poly, plan.PIT_Z, 0.6, (1, 1, 1), 0)
            xm = 0.5 * (pa[0] + pb[0])
            t0 = 0.10 + 0.28 * (plan.X_MAX - xm) / LEN + rng.uniform(-0.008, 0.012)
            t0 = max(t0, 0.40) if xm < plan.X_MIN + 4 else t0
            it = self.add(mb, "wall%d" % k, [self.M["brick"]], t0, t0 + 0.075, mode="courses",
                          drop=H, origin=(xm, 0.5 * (pa[1] + pb[1]), plan.PIT_Z))
            self.wall_segs.append((it, pa, pb, inw))
            # stone cladding of the visible plinth
            mb = MB()
            out = -inw
            cpoly = [tuple(pa + out * 0.35), tuple(pb + out * 0.35), tuple(pb), tuple(pa)]
            mb.prism(cpoly, -0.2, plan.DECK_Z + 0.15, L(226, 214, 190), 0)
            tc = 0.58 + 0.14 * (plan.X_MAX - xm) / LEN + rng.uniform(-0.006, 0.006)
            self.add(mb, "clad%d" % k, [self.M["ashlar"]], tc, tc + 0.03, mode="rise", drop=0.8,
                     origin=(xm, 0.5 * (pa[1] + pb[1]), 0))
        # patio ring wall
        c, r = plan.PATIO_C, plan.PATIO_R
        for k in range(20):
            a0, a1 = 2 * math.pi * k / 20, 2 * math.pi * (k + 1) / 20
            pts = [(c[0] + r * math.cos(a0), c[1] + r * math.sin(a0)),
                   (c[0] + r * math.cos(a1), c[1] + r * math.sin(a1)),
                   (c[0] + (r + 0.6) * math.cos(a1), c[1] + (r + 0.6) * math.sin(a1)),
                   (c[0] + (r + 0.6) * math.cos(a0), c[1] + (r + 0.6) * math.sin(a0))]
            mb = MB()
            mb.prism(pts, plan.PIT_Z, plan.DECK_Z + 0.1, (1, 1, 1), 0)
            t0 = 0.20 + 0.12 * k / 20 + rng.uniform(0, 0.01)
            self.add(mb, "patio%d" % k, [self.M["brick"]], t0, t0 + 0.07, mode="courses",
                     drop=plan.DECK_Z + 0.1 - plan.PIT_Z,
                     origin=(pts[0][0], pts[0][1], plan.PIT_Z))

    def piers(self):
        rng = self.rng
        inner = ISL.buffer(-3.0)
        minx, miny, maxx, maxy = inner.bounds
        s = 6.0
        pc = Point(plan.PATIO_C)
        for ix in range(int((maxx - minx) / s) + 1):
            for iy in range(int((maxy - miny) / s) + 1):
                p = Point(minx + 1 + ix * s, miny + 1 + iy * s)
                if not inner.contains(p) or p.distance(pc) < plan.PATIO_R + 2.0:
                    continue
                mb = MB()
                mb.box((0, 0, 0), (0.6, 0.6, 0.6 - plan.PIT_Z), (1, 1, 1), 0)
                t0 = 0.18 + 0.24 * (plan.X_MAX - p.x) / LEN + rng.uniform(-0.01, 0.01)
                self.add(mb, "pier", [self.M["brick"]], t0, t0 + 0.06, mode="courses",
                         drop=0.6 - plan.PIT_Z, origin=(p.x, p.y, plan.PIT_Z), local=True)
        # arcade of stone columns around the patio (market)
        c = plan.PATIO_C
        for k in range(16):
            a = 2 * math.pi * k / 16
            p = (c[0] + (plan.PATIO_R + 1.6) * math.cos(a), c[1] + (plan.PATIO_R + 1.6) * math.sin(a))
            mb = MB()
            mb.cyl((0, 0, 0), 0.32, 0.6 - plan.PIT_Z, 8, L(226, 214, 190), 0)
            t0 = 0.30 + 0.06 * k / 16
            self.add(mb, "col", [self.M["ashlar"]], t0, t0 + 0.03, mode="rise",
                     drop=2.0, origin=(p[0], p[1], plan.PIT_Z), local=True)

    def scaffolds(self):
        rng = self.rng
        segs = self.wall_segs
        self.scaffold_levels = [plan.PIT_Z + 1.9 * (k + 1) for k in range(3)]
        for k in range(0, len(segs), 3):
            group = segs[k:k + 3]
            mb = MB()
            for it, pa, pb, inw in group:
                d = pb - pa
                ln = np.linalg.norm(d)
                t = d / ln
                for j in range(int(ln / 1.5) + 1):
                    q = pa + t * min(j * 1.5, ln)
                    for off in (0.9, 2.1):
                        b = q + inw * off
                        top = plan.PIT_Z + 6.4 + rng.uniform(-0.3, 0.3)
                        mb.beam((b[0], b[1], plan.PIT_Z), (b[0] + rng.uniform(-0.05, 0.05),
                                                           b[1] + rng.uniform(-0.05, 0.05), top),
                                0.09, L(150, 120, 86), 0)
                for z in self.scaffold_levels:
                    for off in (0.9, 2.1):
                        a2 = pa + inw * off
                        b2 = pb + inw * off
                        mb.beam((a2[0], a2[1], z), (b2[0], b2[1], z + rng.uniform(-0.06, 0.06)),
                                0.08, L(140, 112, 80), 0)
                    a2 = pa + inw * 1.5
                    b2 = pb + inw * 1.5
                    # plank deck
                    mb.beam((a2[0], a2[1], z + 0.06), (b2[0], b2[1], z + 0.06), 1.1,
                            jitter_col(L(176, 146, 100), rng, 0.1), 0, h=0.05)
                # diagonal brace
                a3 = pa + inw * 2.1
                b3 = pb + inw * 2.1
                mb.beam((a3[0], a3[1], plan.PIT_Z), (b3[0], b3[1], plan.PIT_Z + 5.6), 0.07,
                        L(130, 104, 76), 0)
            it0 = group[0][0]
            xm = group[0][1][0]
            tout = 0.60 + 0.10 * (plan.X_MAX - xm) / LEN + rng.uniform(0, 0.02)
            self.add(mb, "scaffold", [self.M["wood"]], it0.t0 - 0.03, it0.t0 - 0.01,
                     out=(tout, tout + 0.02))
        # around the patio
        mb = MB()
        c = plan.PATIO_C
        for k in range(24):
            a = 2 * math.pi * k / 24
            for rr in (plan.PATIO_R - 0.9, plan.PATIO_R - 2.0):
                p = (c[0] + rr * math.cos(a), c[1] + rr * math.sin(a))
                mb.beam((p[0], p[1], plan.PIT_Z), (p[0], p[1], plan.PIT_Z + 6.5), 0.09,
                        L(150, 120, 86), 0)
        for z in self.scaffold_levels:
            for k in range(24):
                a0, a1 = 2 * math.pi * k / 24, 2 * math.pi * (k + 1) / 24
                rr = plan.PATIO_R - 1.45
                mb.beam((c[0] + rr * math.cos(a0), c[1] + rr * math.sin(a0), z + 0.06),
                        (c[0] + rr * math.cos(a1), c[1] + rr * math.sin(a1), z + 0.06), 1.1,
                        L(170, 140, 96), 0, h=0.05)
        self.add(mb, "scaffold_patio", [self.M["wood"]], 0.19, 0.21, out=(0.74, 0.77))

    def deck(self):
        rng = self.rng
        s = 4.0
        minx, miny, maxx, maxy = ISL.bounds
        hole = Point(plan.PATIO_C).buffer(plan.PATIO_R, 32)
        area = ISL.difference(hole)
        self.deck_cells = []
        for ix in range(int((maxx - minx) / s) + 1):
            for iy in range(int((maxy - miny) / s) + 1):
                cell = box(minx + ix * s, miny + iy * s, minx + (ix + 1) * s, miny + (iy + 1) * s)
                for g in polys_of(cell.intersection(area)):
                    if g.area < 0.5:
                        continue
                    mb = MB()
                    shp_prism(mb, g, plan.DECK_Z - plan.SLAB_T, plan.DECK_Z, (1, 1, 1), 0,
                              top_mat=0)
                    cx, cy = g.centroid.x, g.centroid.y
                    t0 = 0.42 + 0.24 * (plan.X_MAX - cx) / LEN + rng.uniform(-0.012, 0.012)
                    for (bx, by), _ in CRANES:
                        if math.hypot(cx - bx, cy - by) < 4.5:
                            t0 = max(t0, 0.69 + rng.uniform(0, 0.01))
                    it = self.add(mb, "deck", [self.M["deck"]], t0, t0 + 0.025, mode="drop",
                                  drop=3.0, origin=(cx, cy, plan.DECK_Z))
                    self.deck_cells.append(it)

    def stairs(self):
        for k, (pt, d) in enumerate(plan.STAIRS):
            mb = MB()
            d = np.array(d)
            t = np.array([-d[1], d[0]])
            n_steps = 6
            for j in range(n_steps):
                h = plan.DECK_Z * (n_steps - j) / n_steps
                c = np.array(pt) + d * (0.2 + 0.34 * j)
                ang = math.atan2(t[1], t[0])
                mb.box((c[0], c[1], 0), (7.0 - 0.0 * j, 0.36 + 0.34 * 0, h), L(220, 208, 184), 0,
                       rot=ang)
            for s_ in (-1, 1):   # cheek pedestals
                c = np.array(pt) + d * 1.0 + t * s_ * 3.8
                mb.box((c[0], c[1], 0), (0.7, 2.4, plan.DECK_Z + 1.0), L(226, 214, 190), 0,
                       rot=math.atan2(t[1], t[0]))
                mb.dome((c[0], c[1], plan.DECK_Z + 1.0), 0.32, n=8, rings=3,
                        col=L(226, 214, 190), mat=0)
            self.add(mb, "stair%d" % k, [self.M["ashlar"]], 0.62 + 0.02 * k, 0.65 + 0.02 * k,
                     mode="rise", drop=1.4)

    def balustrades(self):
        rng = self.rng
        ring = ISL.buffer(-0.25).exterior
        n = int(ring.length / 6.0)
        stair_pts = [Point(p) for p, _ in plan.STAIRS]
        for k in range(n):
            mb = MB()
            s0 = ring.length * k / n
            s1 = ring.length * (k + 1) / n
            m = 0
            pos = s0
            while pos < s1:
                q = ring.interpolate(pos)
                if min(q.distance(sp) for sp in stair_pts) > 3.9:
                    mb.box((q.x, q.y, plan.DECK_Z), (0.16, 0.16, 0.85), L(232, 222, 200), 0)
                    m += 1
                pos += 0.42
            # rails between kept balusters
            pts = [ring.interpolate(s0 + (s1 - s0) * j / 6) for j in range(7)]
            for a, b in zip(pts[:-1], pts[1:]):
                if min(min(a.distance(sp), b.distance(sp)) for sp in stair_pts) < 3.9:
                    continue
                for z, w, h in ((plan.DECK_Z + 0.85, 0.3, 0.14), (plan.DECK_Z, 0.32, 0.12)):
                    mb.beam((a.x, a.y, z + h / 2), (b.x, b.y, z + h / 2), w,
                            L(236, 228, 208), 0, h=h)
            if not mb.v:
                continue
            q = ring.interpolate(s0)
            t0 = 0.66 + 0.12 * (plan.X_MAX - q.x) / LEN + rng.uniform(-0.005, 0.005)
            self.add(mb, "balus", [self.M["ashlar"]], t0, t0 + 0.02, mode="rise", drop=1.0,
                     origin=(q.x, q.y, 0))
        # patio rim
        c, r = plan.PATIO_C, plan.PATIO_R - 0.3
        mb = MB()
        for k in range(150):
            a = 2 * math.pi * k / 150
            mb.box((c[0] + r * math.cos(a), c[1] + r * math.sin(a), plan.DECK_Z),
                   (0.16, 0.16, 0.85), L(232, 222, 200), 0)
        for k in range(36):
            a0, a1 = 2 * math.pi * k / 36, 2 * math.pi * (k + 1) / 36
            mb.beam((c[0] + r * math.cos(a0), c[1] + r * math.sin(a0), plan.DECK_Z + 0.92),
                    (c[0] + r * math.cos(a1), c[1] + r * math.sin(a1), plan.DECK_Z + 0.92),
                    0.3, L(236, 228, 208), 0, h=0.14)
        self.add(mb, "balus_patio", [self.M["ashlar"]], 0.70, 0.73, mode="rise", drop=1.0)

    def gardens(self):
        rng = self.rng
        for k, (c, a, b) in enumerate(plan.parterres()):
            outer = Polygon(plan.ellipse(c, a, b, 32))
            inner = Polygon(plan.ellipse(c, a - 0.5, b - 0.5, 32))
            mb = MB()
            z = plan.DECK_Z
            shp_prism(mb, outer.difference(inner), z, z + 0.55, L(56, 90, 40), 0, top_mat=0)
            mb2 = MB()
            shp_prism(mb2, inner, z, z + 0.3, L(70, 100, 50), 1, top_mat=1)
            t0 = 0.74 + 0.08 * (plan.X_MAX - c[0]) / LEN
            mb.merge(mb2)
            self.add(mb, "parterre%d" % k, [self.M["foliage"], self.M["flowers"]],
                     t0, t0 + 0.05, mode="grow", origin=(c[0], c[1], plan.DECK_Z),
                     scale_axis=2)
        # fountain on the narrow-end circle
        mb = MB()
        c = plan.C2
        mb.cyl((c[0] + 1.5, c[1], plan.DECK_Z), 3.4, 0.6, 24, L(226, 214, 190), 0)
        mb.cyl((c[0] + 1.5, c[1], plan.DECK_Z), 2.9, 0.5, 24, L(80, 110, 120), 1)
        mb.cyl((c[0] + 1.5, c[1], plan.DECK_Z), 0.5, 1.6, 8, L(226, 214, 190), 0)
        mb.cyl((c[0] + 1.5, c[1], plan.DECK_Z + 1.6), 1.1, 0.25, 12, L(226, 214, 190), 0)
        self.add(mb, "fountain", [self.M["ashlar"], self.M["water"]], 0.80, 0.83, mode="rise",
                 drop=1.2)
        # patio floor: flower market stalls under the open sky
        mb = MB()
        pc = plan.PATIO_C
        top_face(mb, Point(pc).buffer(plan.PATIO_R, 32), plan.PIT_Z + 0.14, L(214, 200, 170), 0)
        for k in range(10):
            a = 2 * math.pi * k / 10 + 0.2
            p = (pc[0] + 6.3 * math.cos(a), pc[1] + 6.3 * math.sin(a), plan.PIT_Z + 0.14)
            mb.box(p, (2.0, 1.2, 0.9), L(150, 110, 70), 2, rot=a + math.pi / 2)
            mb.box((p[0], p[1], p[2] + 0.9), (1.9, 1.1, 0.25), L(200, 200, 200), 3,
                   rot=a + math.pi / 2)
            mb.box((p[0], p[1], p[2] + 2.0), (2.4, 1.8, 0.05),
                   rng.choice([L(220, 210, 190), L(170, 60, 50), L(70, 100, 80)]), 2,
                   rot=a + math.pi / 2)
        mb.cyl((pc[0], pc[1], plan.PIT_Z + 0.14), 1.6, 0.5, 16, L(226, 214, 190), 0)
        mb.cyl((pc[0], pc[1], plan.PIT_Z + 0.14), 1.3, 0.45, 16, L(80, 110, 120), 1)
        self.add(mb, "patio_market", [self.M["ashlar"], self.M["water"], self.M["paint"],
                                      self.M["flowers"]], 0.73, 0.78)

    def trees_lamps_kiosks(self):
        rng = self.rng
        ring = ISL.buffer(-2.8).exterior
        stair_pts = [Point(p) for p, _ in plan.STAIRS]
        variants = []
        for k in range(4):
            me = tree_mesh(rng, "tree", k + 3).mesh("deck_tree%d" % k)
            me.materials.append(self.M["foliage"])
            me.materials.append(self.M["wood"])
            variants.append(me)
        palm = tree_mesh(rng, "palm").mesh("deck_palm")
        palm.materials.append(self.M["foliage"])
        palm.materials.append(self.M["wood"])
        n = int(ring.length / 9.0)
        self.deck_trees = []
        for k in range(n):
            q = ring.interpolate(ring.length * k / n)
            if min(q.distance(sp) for sp in stair_pts) < 5:
                continue
            ob = obj_from_mesh("dtree", rng.choice(variants), self.coll,
                               (q.x, q.y, plan.DECK_Z), rng.uniform(0, 6), 1.0)
            t0 = 0.80 + 0.10 * (plan.X_MAX - q.x) / LEN + rng.uniform(0, 0.02)
            it = Item(ob, t0, t0 + 0.06, mode="growtree")
            it.base_scale = (rng.uniform(0.85, 1.05),) * 3
            self.items.append(it)
        for k in range(4):   # palms around the patio
            a = math.pi / 4 + k * math.pi / 2
            p = (plan.PATIO_C[0] + 13.5 * math.cos(a), plan.PATIO_C[1] + 13.5 * math.sin(a))
            ob = obj_from_mesh("dpalm", palm, self.coll, (p[0], p[1], plan.DECK_Z), a, 1.0)
            it = Item(ob, 0.84 + 0.01 * k, 0.92 + 0.01 * k, mode="growtree")
            it.base_scale = (1.0, 1.0, 1.0)
            self.items.append(it)
        # lamps
        mb = MB()
        r2 = ISL.buffer(-1.0).exterior
        for k in range(int(r2.length / 14)):
            q = r2.interpolate(14 * k + 7)
            if min(q.distance(sp) for sp in stair_pts) < 4.2:
                continue
            lamp(mb, (q.x, q.y, plan.DECK_Z))
        self.add(mb, "deck_lamps", [self.M["metal"], self.M["paint"]], 0.80, 0.84, mode="rise",
                 drop=4.0)
        # flower kiosks
        for k, x in enumerate((-22.0, 14.0)):
            for s_ in (-1, 1):
                mb = MB()
                y = plan.axis_y(x) + s_ * 0.0 + s_ * 2.6
                mb.cyl((x, y, plan.DECK_Z), 1.3, 2.2, 8, L(60, 96, 70), 0)
                mb.cyl((x, y, plan.DECK_Z + 0.9), 1.65, 0.35, 8, (1, 1, 1), 2)
                mb.cyl((x, y, plan.DECK_Z + 2.2), 1.9, 0.12, 8, L(50, 54, 50), 1)
                mb.cone((x, y, plan.DECK_Z + 2.32), 1.9, 0.9, 8, L(60, 96, 70), 0)
                self.add(mb, "fkiosk", [self.M["paint"], self.M["metal"], self.M["flowers"]],
                         0.85 + 0.01 * k, 0.88 + 0.01 * k, mode="rise", drop=2.5)

    def hoarding(self):
        """Timber hoarding around the works and the materials yard."""
        rng = self.rng
        site_fence = ISL.buffer(1.8, join_style=1).exterior
        yard = COMPOUND.exterior
        gates = [Point(plan.X_MAX + 1.8, plan.C1[1] + 3.0), Point(89.0, 3.0)]
        for ring, out in ((site_fence, 0.88), (yard, 0.90)):
            n = int(ring.length / 2.4)
            k = 0
            while k < n:
                mb = MB()
                for j in range(k, min(n, k + 5)):
                    a = ring.interpolate(ring.length * j / n)
                    b = ring.interpolate(ring.length * (j + 1) / n)
                    if min(a.distance(g) for g in gates) < 3.5:
                        continue
                    h = 2.0 + rng.uniform(-0.08, 0.08)
                    col = jitter_col(L(150, 118, 82), rng, 0.15)
                    mb.quad_wall((a.x, a.y), (b.x, b.y), 0.0, h, (*col, rng.random()), 0)
                    mb.beam((a.x, a.y, 0), (a.x, a.y, h + 0.2), 0.14, L(110, 90, 66), 0)
                if mb.v:
                    a = ring.interpolate(ring.length * k / n)
                    t = out + 0.05 * rng.random()
                    self.add(mb, "fence", [self.M["wood"]], -1, -1, out=(t, t + 0.02))
                k += 5

    def yard(self):
        """Materials yard: brick stacks, ashlar, timber, sand, sacks, site hut."""
        rng = self.rng
        self.stacks = []
        x0, y0, x1, y1 = COMPOUND.bounds
        for k in range(8):
            mb = MB()
            mb.box((0, 0, 0), (1.3, 1.0, 1.0), (1, 1, 1), 0, rot=0.1 * rng.random())
            p = (x0 + 1.5 + (k % 2) * 1.8, y0 + 2 + (k // 2) * 1.6, 0)
            it = self.add(mb, "bricks", [self.M["brick"]], -1, -1, mode="consume",
                          origin=p, scale_axis=2, local=True)
            it.out0, it.out1 = 0.15 + 0.05 * k, 0.5 + 0.03 * k
        mb = MB()
        for k in range(10):
            mb.box((x0 + 6.5 + (k % 2) * 1.3, y0 + 2 + (k // 2) * 1.4, 0), (1.1, 0.9, 0.7),
                   L(226, 214, 190), 0, rot=rng.uniform(-0.1, 0.1))
        for k in range(6):
            mb.beam((x0 + 1.0, y1 - 8 + k * 0.25, 0.15 + (k % 3) * 0.2),
                    (x0 + 8.5, y1 - 8 + k * 0.25, 0.15 + (k % 3) * 0.2), 0.22,
                    L(170, 140, 96), 1)
        mb.cone((x1 - 3, 0, 0), 2.4, 1.6, 12, L(214, 190, 140), 2)
        mb.cone((x1 - 3.5, -5, 0), 1.8, 1.2, 12, L(150, 146, 140), 2)
        for k in range(12):   # cement sacks
            mb.box((x0 + 2 + (k % 4) * 0.7, 6 + (k // 4) * 0.5, (k // 8) * 0.25),
                   (0.6, 0.42, 0.24), L(220, 214, 196), 3)
        # site hut with tiled roof
        hx, hy = x0 + 5.0, y1 - 3.0
        mb.box((hx, hy, 0), (6.0, 3.6, 2.6), L(170, 140, 100), 1)
        i = mb.add_verts([(hx - 3.3, hy - 2.1, 2.6), (hx + 3.3, hy - 2.1, 2.6),
                          (hx + 3.3, hy, 3.6), (hx - 3.3, hy, 3.6),
                          (hx + 3.3, hy + 2.1, 2.6), (hx - 3.3, hy + 2.1, 2.6)])
        mb.face((i, i + 1, i + 2, i + 3), L(176, 92, 60), 4, [(0, 0), (6.6, 0), (6.6, 2.3), (0, 2.3)])
        mb.face((i + 3, i + 2, i + 4, i + 5), L(176, 92, 60), 4, [(0, 0), (6.6, 0), (6.6, 2.3), (0, 2.3)])
        mb.box((hx + 1.5, hy - 1.85, 0), (0.9, 0.1, 2.0), L(70, 90, 70), 3)
        self.add(mb, "yard", [self.M["ashlar"], self.M["wood"], self.M["earth"],
                              self.M["plain_fade"], self.M["rooftile"]], -1, -1,
                 out=(0.90, 0.93))

    # ----------------------------------------------------------- derricks
    def derricks(self):
        self.cranes = []
        for k, ((bx, by), LB) in enumerate(CRANES):
            z0 = plan.PIT_Z
            mast_h = 15.0
            mb = MB()
            mb.box((bx, by, z0), (2.4, 2.4, 0.4), L(130, 104, 76), 0)
            mb.beam((bx, by, z0 + 0.4), (bx, by, z0 + mast_h), 0.4, L(150, 120, 86), 0)
            mb.box((bx, by, z0 + mast_h), (0.7, 0.7, 0.5), L(110, 90, 66), 0)
            for j in range(4):   # guy ropes to anchors outside the pit
                a = math.pi / 4 + j * math.pi / 2 + 0.3 * k
                ax, ay = bx + 26 * math.cos(a), by + 26 * math.sin(a)
                az = 0.0 if not ISL.contains(Point(ax, ay)) else z0
                mb.beam((bx, by, z0 + mast_h + 0.3), (ax, ay, az), 0.05, L(190, 170, 130), 0)
                mb.box((ax, ay, az), (0.6, 0.6, 0.5), L(120, 110, 100), 0)
            # hand winch
            mb.box((bx + 1.6, by - 0.8, z0), (1.2, 0.8, 0.9), L(120, 96, 70), 0)
            mast = self.add(mb, "mast%d" % k, [self.M["wood"]], 0.12 + 0.02 * k,
                            0.15 + 0.02 * k, out=(0.64 + 0.01 * k, 0.68 + 0.01 * k))
            # boom (local +x)
            mb = MB()
            for s_ in (-1, 1):
                mb.beam((0, s_ * 0.45, 0), (LB, s_ * 0.06, 0), 0.22, L(156, 126, 88), 0)
            for j in range(1, 8):
                x = LB * j / 8
                w = 0.45 * (1 - j / 8) + 0.06
                mb.beam((x, -w, 0), (x + LB / 8, w, 0), 0.06, L(140, 112, 80), 0)
            mb.cyl((LB, 0, -0.3), 0.25, 0.6, 8, L(80, 70, 60), 0)
            boom = mb.build("boom%d" % k, [self.M["wood"]], self.coll,
                            location=(bx, by, z0 + 1.6))
            # rope (unit length along +z) and load
            mb = MB()
            mb.beam((0, 0, 0), (0, 0, 1), 0.06, L(200, 180, 140), 0)
            rope = mb.build("rope%d" % k, [self.M["wood"]], self.coll)
            rope.rotation_mode = "QUATERNION"
            mb = MB()
            mb.box((0, 0, -0.9), (1.6, 1.0, 0.8), L(226, 214, 190), 0)
            mb.beam((0, 0, -0.1), (-0.7, -0.4, -0.5), 0.04, L(200, 180, 140), 1)
            mb.beam((0, 0, -0.1), (0.7, 0.4, -0.5), 0.04, L(200, 180, 140), 1)
            mb.cyl((0, 0, -0.15), 0.18, 0.25, 6, L(60, 60, 60), 2)
            load = mb.build("load%d" % k, [self.M["ashlar"], self.M["wood"], self.M["metal"]],
                            self.coll)
            self.cranes.append(dict(base=np.array([bx, by]), z0=z0 + 1.6, LB=LB, mast=mast,
                                    boom=boom, rope=rope, load=load, k=k))

    def plan_cranes(self, times, pfun, fps):
        """Precompute slew/luff/hook trajectories with a damped pendulum so
        the loads swing and settle like real hemp-rope derricks."""
        rng = random.Random(11)
        n = len(times)
        for cr in self.cranes:
            b, LB, z0 = cr["base"], cr["LB"], cr["z0"]
            slew = np.zeros(n)
            luff = np.zeros(n)
            hookz = np.zeros(n)
            att = np.zeros(n, bool)
            # cycles
            period = 6.5 + cr["k"] * 0.6
            t_start = times[0]
            cyc = []
            t = t_start + cr["k"] * 1.7
            while t < times[-1] + period:
                p = pfun(t)
                # pick point: just outside the hoarding on the crane's side
                ang = math.atan2(b[1] - plan.axis_y(b[0]), 0.0001) if abs(b[1]) > 1 else math.pi
                q = ISL.exterior.interpolate(ISL.exterior.project(Point(*b)))
                v = np.array([q.x, q.y]) - b
                v /= np.linalg.norm(v) + 1e-9
                P = np.array([q.x, q.y]) + v * 4.0 + rng.uniform(-2, 2) * np.array([-v[1], v[0]])
                if p < 0.42:
                    fx = wall_front_x(p)
                    D = boundary_at(fx + rng.uniform(-6, 6), 1 if rng.random() < 0.5 else -1)
                    D = D + (b - D) * 0.08
                    Dz = max(plan.PIT_Z, plan.PIT_Z + (0.6 - plan.PIT_Z) * 0.6)
                else:
                    fx = deck_front_x(p) + rng.uniform(0, 6)
                    D = np.array([fx, plan.axis_y(fx) + rng.uniform(-0.6, 0.6) * plan.half_width(fx)])
                    Dz = plan.DECK_Z + 0.2
                for X in ("P", "D"):
                    Q = P if X == "P" else D
                    d = Q - b
                    r = np.linalg.norm(d)
                    r = min(max(r, 6.0), LB * 0.93)
                    Q2 = b + d / (np.linalg.norm(d) + 1e-9) * r
                    if X == "P":
                        P = Q2
                    else:
                        D = Q2
                cyc.append((t, period * rng.uniform(0.9, 1.1), P, D, Dz))
                t += cyc[-1][1]

            def angles(Q):
                d = Q - b
                r = np.linalg.norm(d)
                return math.atan2(d[1], d[0]), math.acos(min(r / LB, 0.99))

            prev = None
            for i, tt in enumerate(times):
                c = None
                for cc in cyc:
                    if cc[0] <= tt < cc[0] + cc[1]:
                        c = cc
                        break
                if c is None:
                    slew[i], luff[i], hookz[i] = (prev or (0, 0.9, 5))
                    continue
                t0, per, P, D, Dz = c
                ph = (tt - t0) / per
                aP, lP = angles(P)
                aD, lD = angles(D)
                da = (aD - aP + math.pi) % (2 * math.pi) - math.pi
                go = smooth((ph - 0.2) / 0.32)
                back = smooth((ph - 0.74) / 0.24)
                k_ = go * (1 - back)
                slew[i] = aP + da * k_
                luff[i] = lP + (lD - lP) * k_
                tipz = z0 + LB * math.sin(luff[i])
                high = tipz - 3.5
                lowP = 0.0 + 1.2
                lowD = Dz + 1.0
                if ph < 0.1:
                    hz = lowP + (high - lowP) * (1 - smooth(ph / 0.1)) * 0 + lowP * 0
                    hz = high + (lowP - high) * smooth(ph / 0.1)
                elif ph < 0.25:
                    hz = lowP + (high - lowP) * smooth((ph - 0.1) / 0.15)
                elif ph < 0.52:
                    hz = high
                elif ph < 0.66:
                    hz = high + (lowD - high) * smooth((ph - 0.52) / 0.14)
                elif ph < 0.74:
                    hz = lowD
                else:
                    hz = lowD + (high - lowD) * smooth((ph - 0.74) / 0.12)
                hookz[i] = hz
                att[i] = 0.08 < ph < 0.70
                prev = (slew[i], luff[i], hookz[i])
            # pendulum sway of the hook under the tip
            tip = np.stack([b[0] + LB * np.cos(luff) * np.cos(slew),
                            b[1] + LB * np.cos(luff) * np.sin(slew),
                            z0 + LB * np.sin(luff)], 1)
            hook = np.zeros((n, 3))
            pos = tip[0, :2].copy()
            vel = np.zeros(2)
            dt = 1.0 / fps / 8
            for i in range(n):
                ln = max(1.0, tip[i, 2] - hookz[i])
                w2 = 9.81 / ln
                for _ in range(8):
                    acc = w2 * (tip[i, :2] - pos) - 0.35 * vel
                    vel += acc * dt
                    pos += vel * dt
                hook[i, :2] = pos
                hook[i, 2] = hookz[i]
            cr.update(slew=slew, luff=luff, tip=tip, hook=hook, att=att)

    # ----------------------------------------------------------- per frame
    def update(self, p, fi, LIVE):
        for it in self.items:
            self._apply(it, p)
        LIVE["deck_reveal"].default_value = -0.12 + 1.35 * ramp(p, 0.66, 0.84)
        LIVE["bloom"].default_value = ramp(p, 0.80, 0.92)
        for cr in self.cranes:
            a_in = cr["mast"].ob.color[3]
            vis = not cr["mast"].ob.hide_render
            for key in ("boom", "rope", "load"):
                cr[key].hide_render = not vis
                cr[key].color[3] = a_in
            if not vis:
                continue
            cr["boom"].rotation_euler = (0, -cr["luff"][fi], cr["slew"][fi])
            tip = Vector(cr["tip"][fi])
            hook = Vector(cr["hook"][fi])
            d = hook - tip
            cr["rope"].location = tip
            cr["rope"].rotation_quaternion = d.normalized().to_track_quat("Z", "Y")
            cr["rope"].scale = (1, 1, d.length)
            cr["load"].location = hook
            sw = (hook - Vector((tip.x, tip.y, hook.z)))
            cr["load"].rotation_euler = (sw.y * 0.05, -sw.x * 0.05, cr["slew"][fi])
            cr["load"].hide_render = bool((not cr["att"][fi]) or not vis)

    def _apply(self, it, p):
        ob = it.ob
        a_in = 1.0 if it.t0 < 0 else ramp(p, it.t0, it.t1)
        a_out = 0.0 if it.out0 is None else ramp(p, it.out0, it.out1)
        z = it.base_z
        alpha = a_in * (1 - a_out)
        sc = list(it.base_scale)
        m = it.mode
        if m == "fade":
            pass
        elif m in ("rise", "drop"):
            k = settle(a_in) if m == "drop" else smooth(a_in)
            z = it.base_z + (it.drop if m == "drop" else -it.drop) * (1 - k)
            alpha = min(1.0, a_in * 3.0) * (1 - a_out)
        elif m == "courses":
            # laid course by course (17 cm), slightly uneven along the run
            h = it.drop * a_in
            h = math.floor(h / 0.17) * 0.17
            z = it.base_z - it.drop + h
            alpha = 1.0 if a_in > 0.001 else 0.0
        elif m == "grow":
            k = settle(a_in)
            sc[it.scale_axis] = max(0.02, k) * it.base_scale[it.scale_axis]
            alpha = 1.0 if a_in > 0.001 else 0.0
        elif m == "growtree":
            k = settle(a_in)
            s = max(0.05, k) * it.base_scale[0]
            sc = [s, s, s]
            alpha = min(1.0, a_in * 4)
        elif m == "shrink":
            k = 1 - smooth(clamp((p - it.out0) / (it.out1 - it.out0)))
            sc = [it.base_scale[0] * (0.6 + 0.4 * k), it.base_scale[1] * (0.6 + 0.4 * k),
                  it.base_scale[2] * max(k, 0.01)]
            alpha = 1.0 if k > 0.02 else 0.0
        elif m == "consume":
            k = 1 - smooth(clamp((p - it.out0) / (it.out1 - it.out0)))
            sc[2] = max(0.02, math.ceil(k * 5) / 5)
            alpha = 1.0 if k > 0.01 else 0.0
        ob.location.z = z
        ob.scale = sc
        ob.color[3] = float(alpha)
        ob.hide_render = bool(alpha < 0.01)

    def build(self):
        self.pit()
        self.spoil_and_ramp()
        self.floor_slab()
        self.walls()
        self.piers()
        self.scaffolds()
        self.deck()
        self.stairs()
        self.balustrades()
        self.gardens()
        self.trees_lamps_kiosks()
        self.hoarding()
        self.yard()
        self.derricks()
