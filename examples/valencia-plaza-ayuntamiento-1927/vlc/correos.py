"""Palacio de Comunicaciones (Correos y Telégrafos), València.
Miguel Ángel Navarro Pérez, built 1915-1922, opened 1923.

Final building (simplified, recognisable): four wings ringing the public
hall (patio de operaciones) whose stained-glass dome rises over the roofs;
French-eclectic corner pavilions with slate mansards; a projecting central
body on the plaza with a great glazed arch, crowning sculpture group and the
slender metal tower topped by an armillary sphere.

Construction, driven by progress p in [0, 1]:

  0.00-0.12  setting out, hand excavation of the basement, spoil carted off
  0.08-0.22  foundation and basement walls, ground-floor slab
  0.20-0.62  six storeys rise in turn: riveted steel columns set by derrick,
             stone-faced load-bearing walls laid course by course in a wave
             around the plan, floor bays set; timber scaffolding climbs with
             the walls; a timber material hoist runs up and down
  0.40-0.52  hall columns and roof of the patio de operaciones
  0.58-0.72  attic, corner pavilions and mansards, central body
  0.62-0.80  iron ribs of the dome, then the glass; lattice tower, sphere
  0.72-0.80  great arch glazing, sculptures
  0.74-0.88  scaffolding struck from the top down, hoarding removed
"""
import math
import random

import numpy as np
from mathutils import Vector
from shapely.geometry import LineString, Point, Polygon, box
from shapely.geometry.polygon import orient

from . import plan
from .city import top_face, shp_prism, polys_of, largest, tree_mesh, lamp, kiosk
from .common import MB, L, jitter_col, new_coll, obj_from_mesh, ramp, settle, smooth, clamp
from .site import Site, Item

STONE = L(232, 220, 204)
TRIM = L(244, 238, 226)
SLATE = L(70, 76, 84)
IRON = L(56, 70, 66)
G0, HF, NB = 4.6, 3.3, 6          # ground storey, upper storey, number of storeys
H_TOP = G0 + HF * (NB - 1)        # 17.8 m main cornice... + attic below
PIT_Z = -3.2
WING = 13.0


def band_z(k):
    return (0.0, G0) if k == 0 else (G0 + HF * (k - 1), G0 + HF * k)


def t_band(k):
    return 0.20 + 0.068 * k


class Correos(Site):
    def __init__(self, M, rng):
        super().__init__(M, rng, new_coll("correos"))
        blk = plan.block_at((70, 90))
        F = orient(plan.footprint(blk).simplify(0.2), 1.0)
        # chamfer the two plaza corners like the corner pavilions
        self.F = F
        self.hall = largest(F.buffer(-WING, join_style=2))
        c = list(F.exterior.coords)[:-1]
        # front edge = the one closest to the plaza frontage
        best = None
        for i in range(len(c)):
            a, b = np.array(c[i]), np.array(c[(i + 1) % len(c)])
            m = (a + b) / 2
            d = abs(m[1] - plan.y_n(m[0]))
            if best is None or d < best[0]:
                best = (d, a, b)
        _, A, B = best
        self.A, self.B = A, B
        t = (B - A) / np.linalg.norm(B - A)
        self.t = t
        self.out = np.array([t[1], -t[0]])          # towards the plaza
        if self.out[1] > 0:
            self.out = -self.out
        self.front_mid = (A + B) / 2
        self.ang = math.atan2(t[1], t[0])
        hc = self.hall.centroid
        self.dome_c = np.array([hc.x, hc.y])
        self.ring = F.difference(self.hall)
        self.LEN = F.exterior.length
        self.yard = box(100.2, 72.0, 108.5, 124.0)
        self.work_zone = F.buffer(8.0, join_style=2)
        self.cranes_def = [(self.dome_c + np.array([0.0, 0.0]), 27.0, 0.0),
                           (self.front_mid + self.out * 6.5 - self.t * 14.0, 25.0, 0.0)]
        self.hoist_p = None

    # ------------------------------------------------------------- helpers
    def s_of(self, x, y):
        """Normalised position of a point along the perimeter (0 at front-left)."""
        ext = self.F.exterior
        s0 = ext.project(Point(*self.A))
        return ((ext.project(Point(x, y)) - s0) % ext.length) / ext.length

    # -------------------------------------------------------------- static
    def pit(self):
        mb = MB()
        c = list(self.F.exterior.coords)[:-1]
        u = 0
        for k in range(len(c)):
            a, b = c[k], c[(k + 1) % len(c)]
            u += mb.quad_wall(b, a, PIT_Z, 0.0, L(150, 116, 80), 0, u)
        top_face(mb, self.F, PIT_Z, L(150, 116, 80), 0)
        self.pit_obj = mb.build("pit", [self.M["earth"]], self.coll)
        # the lot as found: levelled earth at street level, dug away early
        mb = MB()
        top_face(mb, self.F, 0.0, L(160, 126, 88), 0)
        it = self.add(mb, "lot", [self.M["earth"]], -1, -1, mode="fade")
        it.out0, it.out1 = 0.035, 0.07
        # setting-out stakes and strings
        mb = MB()
        for x, y in c:
            mb.beam((x, y, 0), (x, y, 1.2), 0.08, L(170, 140, 96), 0)
        for k in range(len(c)):
            a, b = c[k], c[(k + 1) % len(c)]
            mb.beam((a[0], a[1], 0.9), (b[0], b[1], 0.9), 0.025, L(230, 225, 210), 0)
        self.add(mb, "stakes", [self.M["wood"]], -1, -1, out=(0.06, 0.08))

    def spoil(self):
        rng = self.rng
        cen = np.array(self.F.centroid.coords[0])
        for k in range(4):
            p = cen + np.array([rng.uniform(-18, 18), rng.uniform(-18, 18)])
            mb = MB()
            mb.sphere((p[0], p[1], PIT_Z), rng.uniform(5, 7), n=14, rings=7,
                      col=L(140, 106, 72), sz=0.45, noise=0.12, rng=rng)
            it = self.add(mb, "spoil", [self.M["earth"]], 0.05 + 0.01 * k, 0.07 + 0.01 * k,
                          mode="grow_shrink", origin=(p[0], p[1], PIT_Z), local=True)
            it.out0, it.out1 = 0.12 + 0.01 * k, 0.20 + 0.01 * k

    def foundations(self):
        rng = self.rng
        # basement walls along the perimeter and the hall ring
        for ring, inward in ((self.F.exterior, True), (self.hall.exterior, False)):
            n = int(ring.length / 4.0)
            for k in range(n):
                a = ring.interpolate(ring.length * k / n)
                b = ring.interpolate(ring.length * (k + 1) / n)
                pa, pb = np.array([a.x, a.y]), np.array([b.x, b.y])
                d = (pb - pa) / np.linalg.norm(pb - pa)
                inw = np.array([-d[1], d[0]])
                mid = (pa + pb) / 2
                if not self.ring.contains(Point(*(mid + inw * 0.8))):
                    inw = -inw
                poly = [tuple(pa), tuple(pb), tuple(pb + inw * 0.8), tuple(pa + inw * 0.8)]
                mb = MB()
                mb.prism(poly, PIT_Z, 0.0, (1, 1, 1), 0)
                t0 = 0.09 + 0.07 * self.s_of(*mid) + rng.uniform(0, 0.01)
                self.add(mb, "found", [self.M["brick"]], t0, t0 + 0.04, mode="courses",
                         drop=-PIT_Z, origin=(mid[0], mid[1], PIT_Z))

    def storeys(self):
        """Stone-faced walls, steel columns and floor bays, storey by storey."""
        rng = self.rng
        self.wall_items = []
        for k in range(NB):
            z0, z1 = band_z(k)
            tk = t_band(k)
            ins = 0.03 * k
            for ring, is_out in ((self.F.exterior, True), (self.hall.exterior, False)):
                n = int(ring.length / 4.0)
                for j in range(n):
                    a = ring.interpolate(ring.length * j / n)
                    b = ring.interpolate(ring.length * (j + 1) / n)
                    pa, pb = np.array([a.x, a.y]), np.array([b.x, b.y])
                    d = (pb - pa) / np.linalg.norm(pb - pa)
                    inw = np.array([-d[1], d[0]])
                    mid = (pa + pb) / 2
                    if not self.ring.contains(Point(*(mid + inw * 0.8))):
                        inw = -inw
                    th = 0.7 - 2 * ins
                    pa2, pb2 = pa + inw * ins, pb + inw * ins
                    poly = [tuple(pa2), tuple(pb2), tuple(pb2 + inw * th), tuple(pa2 + inw * th)]
                    mb = MB()
                    # facade face with window rhythm on the outer side; UV from
                    # the true perimeter so windows line up storey to storey
                    u0 = ring.project(a)
                    i0 = mb.add_verts([(*poly[0], z0), (*poly[1], z0), (*poly[1], z1), (*poly[0], z1)])
                    dl = np.linalg.norm(pb - pa)
                    mb.face((i0, i0 + 1, i0 + 2, i0 + 3), (*STONE, 0.52), 0,
                            [(u0, z0), (u0 + dl, z0), (u0 + dl, z1), (u0, z1)])
                    i1 = mb.add_verts([(*poly[3], z0), (*poly[2], z0), (*poly[2], z1), (*poly[3], z1)])
                    mb.face((i1 + 1, i1, i1 + 3, i1 + 2), L(196, 120, 84), 1)
                    i2 = mb.add_verts([(*p, z1) for p in poly])
                    mb.face((i2, i2 + 1, i2 + 2, i2 + 3), L(190, 110, 80), 1)
                    for e0, e1 in ((poly[0], poly[3]), (poly[1], poly[2])):
                        mb.quad_wall(e0, e1, z0, z1, L(190, 110, 80), 1)
                    s = self.s_of(*mid)
                    t0 = tk + 0.045 * (s if is_out else (s * 0.6 + 0.2)) + rng.uniform(-0.003, 0.004)
                    it = self.add(mb, "wall%d" % k, [self.M["facade"], self.M["brick"]], t0, t0 + 0.04,
                                  mode="courses", drop=z1 - z0, origin=(mid[0], mid[1], z0))
                    self.wall_items.append((k, it, mid, inw, is_out))
            # steel columns on a 6 m grid inside the wings
            minx, miny, maxx, maxy = self.ring.bounds
            inner = self.ring.buffer(-1.6)
            for ix in range(int((maxx - minx) / 6) + 1):
                for iy in range(int((maxy - miny) / 6) + 1):
                    p = Point(minx + 2 + ix * 6, miny + 2 + iy * 6)
                    if not inner.contains(p):
                        continue
                    mb = MB()
                    mb.box((0, 0, 0), (0.32, 0.32, z1 - z0 - 0.3), IRON, 0)
                    mb.box((0, 0, z1 - z0 - 0.3), (0.6, 0.6, 0.3), IRON, 0)
                    t0 = tk - 0.012 + 0.03 * self.s_of(p.x, p.y) + rng.uniform(0, 0.006)
                    self.add(mb, "col%d" % k, [self.M["metal"]], t0, t0 + 0.012, mode="drop",
                             drop=4.0, origin=(p.x, p.y, z0), local=True)
            # floor bays (steel joists + brick vaults + screed)
            for g in self.grid_cells(self.ring, 6.0):
                mb = MB()
                shp_prism(mb, g, z1 - 0.35, z1, L(176, 168, 152), 0, top_mat=0)
                cx, cy = g.centroid.x, g.centroid.y
                t0 = tk + 0.035 + 0.035 * self.s_of(cx, cy) + rng.uniform(0, 0.006)
                self.add(mb, "slab%d" % k, [self.M["plain_fade"]], t0, t0 + 0.012, mode="drop",
                         drop=2.5, origin=(cx, cy, z1))
        # ground-floor slab over the basement (whole footprint)
        for g in self.grid_cells(self.F, 6.0):
            mb = MB()
            shp_prism(mb, g, -0.35, 0.0, L(176, 168, 152), 0, top_mat=0)
            cx, cy = g.centroid.x, g.centroid.y
            t0 = 0.17 + 0.04 * self.s_of(cx, cy) + self.rng.uniform(0, 0.008)
            self.add(mb, "gslab", [self.M["plain_fade"]], t0, t0 + 0.012, mode="drop", drop=2.0,
                     origin=(cx, cy, 0))

    def grid_cells(self, region, s):
        minx, miny, maxx, maxy = region.bounds
        out = []
        for ix in range(int((maxx - minx) / s) + 1):
            for iy in range(int((maxy - miny) / s) + 1):
                cell = box(minx + ix * s, miny + iy * s, minx + (ix + 1) * s, miny + (iy + 1) * s)
                for g in polys_of(cell.intersection(region)):
                    if g.area > 0.6:
                        out.append(g)
        return out

    def hall_and_dome(self):
        rng = self.rng
        hz = G0 + HF * 2 + 0.6          # hall roof level (two storeys high)
        dc, R = self.dome_c, 9.5
        hole = Point(*dc).buffer(R + 0.6, 40)
        for g in self.grid_cells(self.hall, 6.0):
            # hall columns
            p = g.centroid
            if p.distance(Point(*dc)) > R + 2.5:
                mb = MB()
                mb.cyl((0, 0, 0), 0.3, hz - 0.4, 8, IRON, 0)
                t0 = 0.40 + 0.03 * rng.random()
                self.add(mb, "hallcol", [self.M["metal"]], t0, t0 + 0.012, mode="drop", drop=4,
                         origin=(p.x, p.y, 0), local=True)
        for g in self.grid_cells(self.hall.difference(hole), 6.0):
            mb = MB()
            shp_prism(mb, g, hz - 0.4, hz, L(186, 176, 160), 0, top_mat=0)
            t0 = 0.45 + 0.05 * rng.random()
            self.add(mb, "hallroof", [self.M["plain_fade"]], t0, t0 + 0.012, mode="drop", drop=2.5,
                     origin=(g.centroid.x, g.centroid.y, hz))
        # drum
        mb = MB()
        mb.cyl((dc[0], dc[1], hz), R + 0.6, 1.6, 32, STONE, 0)
        self.add(mb, "drum", [self.M["ashlar"]], 0.60, 0.63, mode="rise", drop=1.6)
        # iron ribs, one by one
        dz = hz + 1.6
        for k in range(16):
            a = 2 * math.pi * k / 16
            mb = MB()
            pts = []
            for j in range(9):
                ph = (math.pi / 2) * j / 8
                pts.append((dc[0] + R * math.cos(ph) * math.cos(a), dc[1] + R * math.cos(ph) * math.sin(a),
                            dz + R * 0.62 * math.sin(ph)))
            for p0, p1 in zip(pts[:-1], pts[1:]):
                mb.beam(p0, p1, 0.26, IRON, 0)
            t0 = 0.62 + 0.05 * k / 16
            self.add(mb, "rib", [self.M["metal"]], t0, t0 + 0.012, mode="fade")
        for j in (3, 6):    # ring purlins
            ph = (math.pi / 2) * j / 8
            mb = MB()
            r = R * math.cos(ph)
            for k in range(32):
                a0, a1 = 2 * math.pi * k / 32, 2 * math.pi * (k + 1) / 32
                mb.beam((dc[0] + r * math.cos(a0), dc[1] + r * math.sin(a0), dz + R * 0.62 * math.sin(ph)),
                        (dc[0] + r * math.cos(a1), dc[1] + r * math.sin(a1), dz + R * 0.62 * math.sin(ph)),
                        0.18, IRON, 0)
            self.add(mb, "purlin", [self.M["metal"]], 0.67, 0.68, mode="fade")
        # glass, gore by gore
        for k in range(16):
            mb = MB()
            a0, a1 = 2 * math.pi * k / 16, 2 * math.pi * (k + 1) / 16
            rows = []
            for j in range(7):
                ph = (math.pi / 2) * j / 6 * 0.97
                r = (R - 0.05) * math.cos(ph)
                z = dz + (R - 0.05) * 0.62 * math.sin(ph)
                rows.append([(dc[0] + r * math.cos(a0), dc[1] + r * math.sin(a0), z),
                             (dc[0] + r * math.cos(a1), dc[1] + r * math.sin(a1), z)])
            for j in range(6):
                i = mb.add_verts([rows[j][0], rows[j][1], rows[j + 1][1], rows[j + 1][0]])
                mb.face((i, i + 1, i + 2, i + 3), jitter_col(L(130, 160, 158), rng, 0.08), 0)
            t0 = 0.69 + 0.06 * k / 16
            self.add(mb, "glass", [self.M["glass"]], t0, t0 + 0.015, mode="fade")
        mb = MB()
        mb.cyl((dc[0], dc[1], dz + R * 0.62 - 0.2), 1.4, 1.2, 12, IRON, 0)
        mb.dome((dc[0], dc[1], dz + R * 0.62 + 1.0), 1.4, n=12, rings=3, col=IRON, mat=0)
        self.add(mb, "lantern", [self.M["metal"]], 0.75, 0.77, mode="drop", drop=3)

    def crowning(self):
        """Attic, corner pavilions with mansards, central body, tower."""
        rng = self.rng
        A, B, t, out, ang = self.A, self.B, self.t, self.out, self.ang
        zt = band_z(NB - 1)[1]
        # cornice + attic balustrade along the whole perimeter
        ext = self.F.exterior
        n = int(ext.length / 8)
        for k in range(n):
            a = ext.interpolate(ext.length * k / n)
            b = ext.interpolate(ext.length * (k + 1) / n)
            mb = MB()
            d = np.array([b.x - a.x, b.y - a.y])
            ln = np.linalg.norm(d)
            dd = d / ln
            o = np.array([dd[1], -dd[0]])
            if self.F.contains(Point(*((np.array([a.x, a.y]) + np.array([b.x, b.y])) / 2 + o * 0.8))):
                o = -o
            mid = (np.array([a.x, a.y]) + np.array([b.x, b.y])) / 2 + o * 0.35
            aa = math.atan2(dd[1], dd[0])
            mb.box((mid[0], mid[1], zt), (ln + 0.7, 0.9, 0.55), TRIM, 0, rot=aa)
            m2 = mid - o * 0.25
            mb.box((m2[0], m2[1], zt + 0.55), (ln, 0.35, 0.18), TRIM, 0, rot=aa)
            for j in range(int(ln / 0.45)):
                q = np.array([a.x, a.y]) + dd * (j + 0.5) * 0.45 - o * 0.0
                mb.box((q[0], q[1], zt + 0.73), (0.15, 0.15, 0.75), TRIM, 0)
            mb.box((m2[0], m2[1], zt + 1.48), (ln, 0.4, 0.18), TRIM, 0, rot=aa)
            s = self.s_of(*mid)
            t0 = 0.60 + 0.05 * s
            self.add(mb, "attic", [self.M["ashlar"]], t0, t0 + 0.015, mode="rise", drop=1.6)
        # corner pavilions on the plaza with slate mansards and lanterns
        for s_, P in ((-1, A), (1, B)):
            c = P - t * s_ * 5.6 - out * 5.6
            mb = MB()
            mb.box((c[0], c[1], zt), (11.2, 11.2, 4.2), (*STONE, 0.27), 0, rot=ang)
            mb.box((c[0], c[1], zt + 4.2), (12.0, 12.0, 0.5), TRIM, 1, rot=ang)
            self.add(mb, "pav%d" % s_, [self.M["facade"], self.M["ashlar"]], 0.61, 0.64,
                     mode="courses", drop=4.7, origin=(c[0], c[1], zt))
            mb = MB()
            base = [(-5.6, -5.6), (5.6, -5.6), (5.6, 5.6), (-5.6, 5.6)]
            topp = [(-3.2, -3.2), (3.2, -3.2), (3.2, 3.2), (-3.2, 3.2)]
            ca_, sa_ = math.cos(ang), math.sin(ang)
            R2 = lambda q, z: (c[0] + q[0] * ca_ - q[1] * sa_, c[1] + q[0] * sa_ + q[1] * ca_, z)
            z0 = zt + 4.7
            for k in range(4):
                i = mb.add_verts([R2(base[k], z0), R2(base[(k + 1) % 4], z0),
                                  R2(topp[(k + 1) % 4], z0 + 4.2), R2(topp[k], z0 + 4.2)])
                mb.face((i, i + 1, i + 2, i + 3), SLATE, 0, [(0, 0), (11.2, 0), (6.4, 4.4), (0, 4.4)])
                # dormer
                q = ((base[k][0] + base[(k + 1) % 4][0]) / 2 * 0.86, (base[k][1] + base[(k + 1) % 4][1]) / 2 * 0.86)
                mb.box(R2(q, z0 + 0.6), (1.5, 1.5, 1.9), TRIM, 1)
            i = mb.add_verts([R2(q, z0 + 4.2) for q in topp])
            mb.face((i, i + 1, i + 2, i + 3), SLATE, 0)
            mb.cyl(R2((0, 0), z0 + 4.2), 1.3, 1.6, 10, TRIM, 1)
            mb.dome(R2((0, 0), z0 + 5.8), 1.3, n=10, rings=3, col=SLATE, mat=0)
            mb.beam(R2((0, 0), z0 + 7.0), R2((0, 0), z0 + 9.0), 0.1, IRON, 2)
            self.add(mb, "mansard%d" % s_, [self.M["rooftile"], self.M["ashlar"], self.M["metal"]],
                     0.65, 0.69, mode="rise", drop=3.0)
        # central body: projecting frontispiece with the great arch
        c = self.front_mid + out * 1.2
        mb = MB()
        mb.box((c[0], c[1], 0), (19.0, 2.4, zt + 6.0), (*STONE, 0.33), 0, rot=ang)
        self.add(mb, "frontis", [self.M["facade"]], 0.36, 0.60, mode="courses", drop=zt + 6.0,
                 origin=(c[0], c[1], 0))
        mb = MB()
        f = c + out * 1.25
        mb.box((f[0], f[1], 1.0), (9.4, 0.3, 13.0), L(38, 46, 54), 0, rot=ang)       # glazing
        for j in range(5):
            q = f + t * (j - 2) * 2.1 + out * 0.1
            mb.box((q[0], q[1], 1.0), (0.18, 0.3, 13.0), IRON, 1, rot=ang)
        for j in range(4):
            q = f + out * 0.1
            mb.box((q[0], q[1], 3.5 + j * 3.0), (9.4, 0.3, 0.16), IRON, 1, rot=ang)
        for j in range(13):    # arch voussoirs
            a = math.pi * j / 12
            q = f + t * 5.3 * math.cos(a) + out * 0.2
            mb.box((q[0], q[1], 14.0 + 5.3 * math.sin(a) * 0.6), (1.2, 0.5, 1.0), TRIM, 2, rot=ang + a)
        for s_ in (-1, 1):
            for j in range(2):
                q = f + t * s_ * (6.0 + 1.4 * j) + out * 0.4
                mb.cyl((q[0], q[1], 1.0), 0.45, 15.0, 10, TRIM, 2)
        self.add(mb, "arch", [self.M["glass"], self.M["metal"], self.M["ashlar"]], 0.72, 0.75)
        # crowning group of sculptures (allegories of the Posts and Telegraphs)
        mb = MB()
        top = zt + 6.0
        mb.box((c[0], c[1], top), (10.0, 3.0, 1.4), TRIM, 0, rot=ang)
        for j, (dx, h) in enumerate(((-3.2, 3.2), (-1.2, 3.8), (1.2, 3.8), (3.2, 3.2))):
            q = c + t * dx
            mb.cyl((q[0], q[1], top + 1.4), 0.55, h * 0.55, 8, L(214, 206, 190), 0, r2=0.42)
            mb.sphere((q[0], q[1], top + 1.4 + h * 0.62), 0.32, n=8, rings=5, col=L(214, 206, 190))
            mb.beam((q[0], q[1], top + 1.4 + h * 0.45), (q[0] + 0.6 * out[0] + 0.5 * dx / 3,
                    q[1] + 0.6 * out[1], top + 1.4 + h * (0.75 + 0.05 * j)), 0.16, L(214, 206, 190), 0)
        self.add(mb, "sculpt", [self.M["ashlar"]], 0.76, 0.79, mode="drop", drop=3.0)
        # slender metal tower with the armillary sphere
        tw = c - out * 0.6
        z0 = top + 1.4
        Ht = 21.0
        segs = 7
        for k in range(segs):
            mb = MB()
            za, zb = z0 + Ht * k / segs, z0 + Ht * (k + 1) / segs
            wa, wb = 2.4 * (1 - 0.75 * k / segs), 2.4 * (1 - 0.75 * (k + 1) / segs)
            cs = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
            for (sx, sy) in cs:
                mb.beam((tw[0] + sx * wa, tw[1] + sy * wa, za), (tw[0] + sx * wb, tw[1] + sy * wb, zb), 0.16, IRON, 0)
            for i in range(4):
                (sx, sy), (sx2, sy2) = cs[i], cs[(i + 1) % 4]
                mb.beam((tw[0] + sx * wa, tw[1] + sy * wa, za), (tw[0] + sx2 * wb, tw[1] + sy2 * wb, zb), 0.07, IRON, 0)
                mb.beam((tw[0] + sx * wb, tw[1] + sy * wb, zb), (tw[0] + sx2 * wb, tw[1] + sy2 * wb, zb), 0.09, IRON, 0)
            t0 = 0.66 + 0.10 * k / segs
            self.add(mb, "tower%d" % k, [self.M["metal"]], t0, t0 + 0.012, mode="drop", drop=5.0)
        mb = MB()
        sz = z0 + Ht + 2.4
        for rot in (0.0, math.pi / 2, math.pi / 4):
            pts = []
            for k in range(25):
                a = 2 * math.pi * k / 24
                v = np.array([2.2 * math.cos(a), 0.0, 2.2 * math.sin(a)])
                v = np.array([v[0] * math.cos(rot) - v[1] * math.sin(rot), v[0] * math.sin(rot) + v[1] * math.cos(rot), v[2]])
                pts.append((tw[0] + v[0], tw[1] + v[1], sz + v[2]))
            for p0, p1 in zip(pts[:-1], pts[1:]):
                mb.beam(p0, p1, 0.12, L(176, 150, 80), 0)
        pts = [(tw[0] + 2.2 * math.cos(2 * math.pi * k / 24), tw[1] + 2.2 * math.sin(2 * math.pi * k / 24), sz)
               for k in range(25)]
        for p0, p1 in zip(pts[:-1], pts[1:]):
            mb.beam(p0, p1, 0.12, L(176, 150, 80), 0)
        mb.beam((tw[0], tw[1], sz - 3.0), (tw[0], tw[1], sz + 3.2), 0.1, L(176, 150, 80), 0)
        mb.sphere((tw[0], tw[1], sz), 0.45, n=8, rings=5, col=L(196, 170, 90))
        self.add(mb, "armillary", [self.M["metal"]], 0.78, 0.80, mode="drop", drop=6.0)
        self.tower_top = (tw[0], tw[1], sz)
        # flags for the opening (red-gold-red of 1923)
        self.flag_spots = [((c + t * dx + out * 1.3)[0], (c + t * dx + out * 1.3)[1], 6.0)
                           for dx in (-7.0, 7.0)]

    def ornament(self):
        """Rusticated base, giant pilasters, string courses, roof finish."""
        rng = self.rng
        zt = band_z(NB - 1)[1]
        ext = self.F.exterior
        c = list(self.F.exterior.coords)[:-1]
        for i in range(len(c)):
            a, b = np.array(c[i]), np.array(c[(i + 1) % len(c)])
            d = b - a
            ln = np.linalg.norm(d)
            d /= ln
            o = np.array([d[1], -d[0]])
            mid = (a + b) / 2
            if self.F.contains(Point(*(mid + o * 0.8))):
                o = -o
            ang = math.atan2(d[1], d[0])
            mb = MB()
            mb.box((*(mid + o * 0.22), 0), (ln, 0.45, G0 - 0.3), (*STONE, 0.9), 1, rot=ang)
            mb.box((*(mid + o * 0.3), G0 - 0.3), (ln + 0.3, 0.6, 0.35), TRIM, 1, rot=ang)
            for z in (G0 + HF * 1 - 0.1, G0 + HF * 3 - 0.1):
                mb.box((*(mid + o * 0.3), z), (ln, 0.5, 0.22), TRIM, 1, rot=ang)
            n_p = max(2, int(ln / 7.2))
            for k in range(n_p + 1):
                q = a + d * ln * k / n_p + o * 0.2
                mb.box((*q, G0 + 0.05), (0.9, 0.42, zt - G0 - 0.1), TRIM, 1, rot=ang)
            s = self.s_of(*mid)
            t0 = 0.70 + 0.05 * s
            self.add(mb, "ornament", [self.M["facade"], self.M["ashlar"]], t0, t0 + 0.02, mode="fade")
        mb = MB()
        inner = self.ring.buffer(-0.6, join_style=2)
        for g in polys_of(inner):
            top_face(mb, g, zt + 0.03, L(200, 160, 128), 0)
        self.add(mb, "roof_finish", [self.M["terrace"]], 0.66, 0.70, mode="fade")

    def scaffolds(self):
        """Timber scaffolding: one lift per storey per facade run, climbing
        with the walls and struck from the top down."""
        rng = self.rng
        ext = self.F.exterior
        n = int(ext.length / 8.0)
        for k in range(NB):
            z0, z1 = band_z(k)
            for j in range(n):
                a = ext.interpolate(ext.length * j / n)
                b = ext.interpolate(ext.length * (j + 1) / n)
                pa, pb = np.array([a.x, a.y]), np.array([b.x, b.y])
                d = (pb - pa) / np.linalg.norm(pb - pa)
                o = np.array([d[1], -d[0]])
                mid = (pa + pb) / 2
                if self.F.contains(Point(*(mid + o * 0.8))):
                    o = -o
                mb = MB()
                ln = np.linalg.norm(pb - pa)
                for i in range(int(ln / 2.0) + 1):
                    q = pa + d * min(i * 2.0, ln)
                    for off in (0.9, 2.1):
                        bq = q + o * off
                        mb.beam((bq[0], bq[1], z0), (bq[0] + rng.uniform(-0.04, 0.04), bq[1], z1 + 0.15),
                                0.1, L(150, 120, 86), 0)
                for off in (0.9, 2.1):
                    a2, b2 = pa + o * off, pb + o * off
                    mb.beam((a2[0], a2[1], z1 - 0.05), (b2[0], b2[1], z1 - 0.05 + rng.uniform(-0.05, 0.05)),
                            0.08, L(140, 112, 80), 0)
                a2, b2 = pa + o * 1.5, pb + o * 1.5
                mb.beam((a2[0], a2[1], z1 + 0.02), (b2[0], b2[1], z1 + 0.02), 1.1,
                        jitter_col(L(176, 146, 100), rng, 0.1), 0, h=0.05)
                a3, b3 = pa + o * 2.1, pb + o * 2.1
                mb.beam((a3[0], a3[1], z0), (b3[0], b3[1], z1), 0.07, L(130, 104, 76), 0)
                s = self.s_of(*mid)
                t_in = t_band(k) + 0.045 * s - 0.006
                t_out = 0.74 + 0.11 * (NB - 1 - k) / NB + 0.015 * s
                self.add(mb, "scaf%d" % k, [self.M["wood"]], t_in, t_in + 0.008, out=(t_out, t_out + 0.01))

    def hoarding_and_yard(self):
        rng = self.rng
        ring = self.work_zone.exterior
        n = int(ring.length / 2.4)
        for k in range(0, n, 5):
            mb = MB()
            for j in range(k, min(n, k + 5)):
                a = ring.interpolate(ring.length * j / n)
                b = ring.interpolate(ring.length * (j + 1) / n)
                h = 2.2 + rng.uniform(-0.08, 0.08)
                col = jitter_col(L(150, 118, 82), rng, 0.15)
                mb.quad_wall((a.x, a.y), (b.x, b.y), 0.0, h, (*col, rng.random()), 0)
                mb.beam((a.x, a.y, 0), (a.x, a.y, h + 0.2), 0.14, L(110, 90, 66), 0)
            t = 0.86 + 0.04 * rng.random()
            self.add(mb, "fence", [self.M["wood"]], -1, -1, out=(t, t + 0.015))
        # stone yard on the cross street: ashlar blocks, timber, sand, hut
        x0, y0, x1, y1 = self.yard.bounds
        for k in range(10):
            mb = MB()
            mb.box((0, 0, 0), (1.5, 1.0, 0.9), (1, 1, 1), 0, rot=rng.uniform(-0.1, 0.1))
            p = (x0 + 1.4 + (k % 3) * 2.2, y0 + 3 + (k // 3) * 3.0, 0)
            it = self.add(mb, "ashlar_stack", [self.M["ashlar"]], -1, -1, mode="consume",
                          origin=p, scale_axis=2, local=True)
            it.out0, it.out1 = 0.20 + 0.04 * k, 0.55 + 0.03 * k
        mb = MB()
        for k in range(6):
            mb.beam((x0 + 1.0, y1 - 14 + k * 0.25, 0.15 + (k % 3) * 0.2),
                    (x0 + 7.5, y1 - 14 + k * 0.25, 0.15 + (k % 3) * 0.2), 0.22, L(170, 140, 96), 1)
        mb.cone((x1 - 2.5, y1 - 22, 0), 2.2, 1.5, 12, L(214, 190, 140), 2)
        for k in range(8):   # steel joists
            mb.beam((x0 + 1.0, y1 - 18 + k * 0.3, 0.2), (x0 + 7.6, y1 - 18 + k * 0.3, 0.2), 0.18, IRON, 3)
        hx, hy = x0 + 4.2, y1 - 4.0
        mb.box((hx, hy, 0), (6.0, 3.6, 2.6), L(170, 140, 100), 1)
        i = mb.add_verts([(hx - 3.3, hy - 2.1, 2.6), (hx + 3.3, hy - 2.1, 2.6), (hx + 3.3, hy, 3.6),
                          (hx - 3.3, hy, 3.6), (hx + 3.3, hy + 2.1, 2.6), (hx - 3.3, hy + 2.1, 2.6)])
        mb.face((i, i + 1, i + 2, i + 3), L(176, 92, 60), 4, [(0, 0), (6.6, 0), (6.6, 2.3), (0, 2.3)])
        mb.face((i + 3, i + 2, i + 4, i + 5), L(176, 92, 60), 4, [(0, 0), (6.6, 0), (6.6, 2.3), (0, 2.3)])
        self.add(mb, "yard", [self.M["ashlar"], self.M["wood"], self.M["earth"], self.M["metal"],
                              self.M["rooftile"]], -1, -1, out=(0.88, 0.91))
        # banker stones for the stone cutters
        self.bankers = [(x0 + 2.0 + 2.6 * (k % 3), y0 + 16 + 2.6 * (k // 3)) for k in range(6)]
        mb = MB()
        for bx, by in self.bankers:
            mb.box((bx, by, 0), (1.2, 0.8, 0.8), L(226, 214, 190), 0)
        self.add(mb, "bankers", [self.M["ashlar"]], -1, -1, out=(0.80, 0.83))

    def hoist(self):
        """Timber material hoist on the side street, cage running up and down."""
        F = self.F
        ext = F.exterior
        # midpoint of the side facade facing the cross street (largest x)
        c = list(F.exterior.coords)[:-1]
        best = max(range(len(c)), key=lambda i: (c[i][0] + c[(i + 1) % len(c)][0]))
        a, b = np.array(c[best]), np.array(c[(best + 1) % len(c)])
        mid = (a + b) / 2
        d = (b - a) / np.linalg.norm(b - a)
        o = np.array([d[1], -d[0]])
        if F.contains(Point(*(mid + o))):
            o = -o
        p = mid + o * 4.0 + d * 6.0
        self.hoist_p = p
        mb = MB()
        Ht = band_z(NB - 1)[1] + 4
        for sx in (-1, 1):
            for sy in (-1, 1):
                mb.beam((p[0] + sx * 1.1, p[1] + sy * 1.1, 0), (p[0] + sx * 1.0, p[1] + sy * 1.0, Ht), 0.18,
                        L(150, 120, 86), 0)
        for z in np.arange(2.0, Ht, 2.4):
            for (sx, sy), (sx2, sy2) in (((-1, -1), (1, -1)), ((1, -1), (1, 1)), ((1, 1), (-1, 1)), ((-1, 1), (-1, -1))):
                mb.beam((p[0] + sx, p[1] + sy, z), (p[0] + sx2, p[1] + sy2, z + 1.2), 0.07, L(140, 112, 80), 0)
        mb.box((p[0], p[1], Ht), (2.6, 2.6, 0.4), L(120, 96, 70), 0)
        mb.cyl((p[0], p[1] + 1.6, Ht + 0.4), 0.35, 0.6, 8, L(80, 70, 60), 0)
        mb.box((p[0] + 2.0, p[1] - 1.5, 0), (1.4, 1.0, 1.0), L(110, 90, 66), 0)  # winch
        self.hoist_tower = self.add(mb, "hoist", [self.M["wood"]], 0.20, 0.23, out=(0.80, 0.83))
        mb = MB()
        mb.box((0, 0, 0), (1.8, 1.8, 0.15), L(150, 120, 86), 0)
        mb.box((0, 0.3, 0.15), (1.2, 0.8, 0.6), L(176, 88, 58), 1)
        mb.beam((0, 0, 0.15), (0, 0, 2.0), 0.04, L(200, 180, 140), 0)
        self.cage = mb.build("cage", [self.M["wood"], self.M["brick"]], self.coll)
        self.hoist_H = Ht

    def plaza_1915(self):
        """The plaza as it was while the Palacio rose: the island of the
        Urbanización de la plaza de Emilio Castelar laid out as gardens at
        grade, with parterres, trees, a fountain and flower stalls."""
        rng = self.rng
        isl = plan.ISLAND
        mb = MB()
        shp_prism(mb, isl, -0.3, 0.18, L(214, 200, 176), 0, top_mat=1, top_col=L(206, 190, 160))
        for c, a, b in plan.parterres():
            outer = Polygon(plan.ellipse(c, a, b, 32))
            inner = Polygon(plan.ellipse(c, a - 0.5, b - 0.5, 32))
            shp_prism(mb, outer.difference(inner), 0.18, 0.75, L(56, 90, 40), 2, top_mat=2)
            shp_prism(mb, inner, 0.18, 0.45, L(70, 100, 50), 3, top_mat=3)
        pc = plan.PATIO_C
        mb.cyl((pc[0], pc[1], 0.18), 5.0, 0.7, 32, L(226, 214, 190), 4)
        mb.cyl((pc[0], pc[1], 0.18), 4.5, 0.6, 32, L(80, 110, 120), 5)
        mb.cyl((pc[0], pc[1], 0.18), 0.6, 2.4, 10, L(226, 214, 190), 4)
        mb.cyl((pc[0], pc[1], 2.6), 1.5, 0.3, 14, L(226, 214, 190), 4)
        self.island_obj = mb.build("island1915", [self.M["ashlar"], self.M["sidewalk"], self.M["foliage"],
                                                  self.M["flowers"], self.M["ashlar"], self.M["water"]],
                                   self.coll)
        mb = MB()
        r2 = isl.buffer(-1.0).exterior
        for k in range(int(r2.length / 15)):
            q = r2.interpolate(15 * k + 7)
            lamp(mb, (q.x, q.y, 0.18))
        for x in (-22.0, 14.0, 34.0):
            for s_ in (-1, 1):
                kiosk(mb, (x, plan.axis_y(x) + s_ * 3.2, 0.18), rng)
        mb.build("island_furniture", [self.M["metal"], self.M["paint"], self.M["plain"]], self.coll)
        variants = []
        for k in range(4):
            me = tree_mesh(rng, "tree", k + 3).mesh("itree%d" % k)
            me.materials.append(self.M["foliage"])
            me.materials.append(self.M["wood"])
            variants.append(me)
        ring = isl.buffer(-2.8).exterior
        for k in range(int(ring.length / 8.5)):
            q = ring.interpolate(ring.length * k / int(ring.length / 8.5))
            obj_from_mesh("itree", rng.choice(variants), self.coll, (q.x, q.y, 0.18),
                          rng.uniform(0, 6), rng.uniform(0.85, 1.1))

    # ----------------------------------------------------------- derricks
    def derricks(self):
        self.cranes = []
        for k, (b, LB, z0) in enumerate(self.cranes_def):
            mast_h = 30.0
            mb = MB()
            mb.box((b[0], b[1], z0), (2.6, 2.6, 0.5), L(130, 104, 76), 0)
            mb.beam((b[0], b[1], z0 + 0.5), (b[0], b[1], z0 + mast_h), 0.5, L(150, 120, 86), 0)
            mb.box((b[0], b[1], z0 + mast_h), (0.8, 0.8, 0.6), L(110, 90, 66), 0)
            for j in range(4):
                a = math.pi / 4 + j * math.pi / 2 + 0.25 * k
                rr = 10.5 if k == 0 else 30.0
                ax, ay = b[0] + rr * math.cos(a), b[1] + rr * math.sin(a)
                mb.beam((b[0], b[1], z0 + mast_h + 0.3), (ax, ay, 0.0), 0.05, L(190, 170, 130), 0)
                mb.box((ax, ay, 0.0), (0.6, 0.6, 0.5), L(120, 110, 100), 0)
            mb.box((b[0] + 1.8, b[1] - 0.9, z0), (1.3, 0.9, 1.0), L(120, 96, 70), 0)
            t_out = 0.58 if k == 0 else 0.80
            mast = self.add(mb, "mast%d" % k, [self.M["wood"]], 0.19, 0.21, out=(t_out, t_out + 0.025))
            mb = MB()
            for s_ in (-1, 1):
                mb.beam((0, s_ * 0.5, 0), (LB, s_ * 0.07, 0), 0.26, L(156, 126, 88), 0)
            for j in range(1, 9):
                x = LB * j / 9
                w = 0.5 * (1 - j / 9) + 0.07
                mb.beam((x, -w, 0), (x + LB / 9, w, 0), 0.07, L(140, 112, 80), 0)
            mb.cyl((LB, 0, -0.3), 0.28, 0.6, 8, L(80, 70, 60), 0)
            boom = mb.build("boom%d" % k, [self.M["wood"]], self.coll, location=(b[0], b[1], z0 + 2.0))
            mb = MB()
            mb.beam((0, 0, 0), (0, 0, 1), 0.07, L(200, 180, 140), 0)
            rope = mb.build("rope%d" % k, [self.M["wood"]], self.coll)
            rope.rotation_mode = "QUATERNION"
            mb = MB()
            mb.box((0, 0, -1.0), (1.8, 1.0, 0.9), L(226, 214, 190), 0)
            mb.beam((0, 0, -0.1), (-0.8, -0.45, -0.55), 0.04, L(200, 180, 140), 1)
            mb.beam((0, 0, -0.1), (0.8, 0.45, -0.55), 0.04, L(200, 180, 140), 1)
            mb.cyl((0, 0, -0.15), 0.2, 0.25, 6, L(60, 60, 60), 2)
            load = mb.build("load%d" % k, [self.M["ashlar"], self.M["wood"], self.M["metal"]], self.coll)
            mb = MB()
            mb.box((0, 0, -0.4), (0.4, 6.0, 0.4), IRON, 0)
            girder = mb.build("girder%d" % k, [self.M["metal"]], self.coll)
            self.cranes.append(dict(base=np.array(b), z0=z0 + 2.0, LB=LB, mast=mast, boom=boom,
                                    rope=rope, load=load, girder=girder, k=k))

    def work_level(self, p):
        """Top of the storey currently being built."""
        for k in range(NB):
            if p < t_band(k) + 0.09:
                return band_z(k)[1]
        return band_z(NB - 1)[1] + 4

    def plan_cranes(self, times, pfun, fps):
        rng = random.Random(11)
        n = len(times)
        x0, y0, x1, y1 = self.yard.bounds
        for cr in self.cranes:
            b, LB, z0 = cr["base"], cr["LB"], cr["z0"]
            slew, luff, hookz = np.zeros(n), np.zeros(n), np.zeros(n)
            att = np.zeros(n, bool)
            kind = np.zeros(n, int)
            period = 7.0 + cr["k"] * 0.8
            cyc = []
            t = times[0] + cr["k"] * 2.3
            while t < times[-1] + period:
                p = pfun(t)
                # pick from the yard side of the hoarding, drop on the working storey
                if cr["k"] == 0:
                    P = np.array([b[0] + rng.uniform(-6, 6), b[1] + rng.uniform(-6, 6)])
                    Pz = 0.0
                else:
                    P = b + np.array([rng.uniform(-3, 3), -rng.uniform(5, 9)])
                    Pz = 0.0
                ring = self.ring.buffer(-1.0)
                for _ in range(20):
                    ang = rng.uniform(0, 2 * math.pi)
                    r = rng.uniform(8, LB * 0.9)
                    D = b + r * np.array([math.cos(ang), math.sin(ang)])
                    if ring.contains(Point(*D)):
                        break
                Dz = self.work_level(p)
                if p > 0.6 and cr["k"] == 1:     # lifts for the crowning works
                    D = self.front_mid - self.out * 1.0 + self.t * rng.uniform(-8, 8)
                    Dz = band_z(NB - 1)[1] + 6.0
                cyc.append((t, period * rng.uniform(0.9, 1.1), P, Pz, D, Dz, rng.random() < 0.4))
                t += cyc[-1][1]

            def angles(Q):
                d = Q - b
                r = min(max(np.linalg.norm(d), 4.0), LB * 0.95)
                return math.atan2(d[1], d[0]), math.acos(min(r / LB, 0.99))

            prev = None
            for i, tt in enumerate(times):
                c = next((cc for cc in cyc if cc[0] <= tt < cc[0] + cc[1]), None)
                if c is None:
                    slew[i], luff[i], hookz[i] = prev or (0, 1.0, 10)
                    continue
                t0, per, P, Pz, D, Dz, gird = c
                ph = (tt - t0) / per
                aP, lP = angles(P)
                aD, lD = angles(D)
                da = (aD - aP + math.pi) % (2 * math.pi) - math.pi
                k_ = smooth((ph - 0.2) / 0.32) * (1 - smooth((ph - 0.74) / 0.24))
                slew[i] = aP + da * k_
                luff[i] = lP + (lD - lP) * k_
                tipz = z0 + LB * math.sin(luff[i])
                high = tipz - 3.5
                lowP, lowD = Pz + 1.3, Dz + 1.2
                if ph < 0.1:
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
                hookz[i] = min(hz, tipz - 1.0)
                att[i] = 0.08 < ph < 0.70
                kind[i] = 1 if gird else 0
                prev = (slew[i], luff[i], hookz[i])
            tip = np.stack([b[0] + LB * np.cos(luff) * np.cos(slew), b[1] + LB * np.cos(luff) * np.sin(slew),
                            z0 + LB * np.sin(luff)], 1)
            hook = np.zeros((n, 3))
            pos = tip[0, :2].copy()
            vel = np.zeros(2)
            dt = 1.0 / fps / 8
            for i in range(n):
                ln = max(1.0, tip[i, 2] - hookz[i])
                for _ in range(8):
                    acc = 9.81 / ln * (tip[i, :2] - pos) - 0.35 * vel
                    vel += acc * dt
                    pos += vel * dt
                hook[i, :2] = pos
                hook[i, 2] = hookz[i]
            cr.update(slew=slew, luff=luff, tip=tip, hook=hook, att=att, kind=kind)

    def update(self, p, fi, LIVE):
        for it in self.items:
            self._apply(it, p)
        LIVE["deck_reveal"].default_value = 1.2
        LIVE["bloom"].default_value = 1.0
        for cr in self.cranes:
            vis = not cr["mast"].ob.hide_render
            a_in = cr["mast"].ob.color[3]
            for key in ("boom", "rope", "load", "girder"):
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
            sw = hook - Vector((tip.x, tip.y, hook.z))
            for key, want in (("load", 0), ("girder", 1)):
                ob = cr[key]
                ob.location = hook
                ob.rotation_euler = (sw.y * 0.05, -sw.x * 0.05, cr["slew"][fi] + 0.3 * math.sin(fi / 30))
                ob.hide_render = bool((not cr["att"][fi]) or cr["kind"][fi] != want)
        # material hoist cage: up and down continuously below the working level
        if self.hoist_p is not None:
            vis = not self.hoist_tower.ob.hide_render
            self.cage.hide_render = not vis
            top = max(2.0, min(self.work_level(p) - 0.2, self.hoist_H - 1))
            cyc = (fi / 24.0) % 9.0
            k = smooth(cyc / 3.5) if cyc < 4.5 else 1 - smooth((cyc - 4.5) / 3.5)
            self.cage.location = (self.hoist_p[0], self.hoist_p[1], 0.3 + (top - 0.3) * k)

    def _apply(self, it, p):
        if it.mode == "grow_shrink":
            a_in = ramp(p, it.t0, it.t1)
            k = 1 - smooth(clamp((p - it.out0) / (it.out1 - it.out0)))
            s = settle(a_in) * k
            it.ob.scale = (it.base_scale[0] * (0.5 + 0.5 * s), it.base_scale[1] * (0.5 + 0.5 * s),
                           max(0.01, s) * it.base_scale[2])
            it.ob.color[3] = 1.0
            it.ob.hide_render = bool(s < 0.02)
            return
        super()._apply(it, p)

    def build(self):
        self.pit()
        self.spoil()
        self.foundations()
        self.storeys()
        self.hall_and_dome()
        self.crowning()
        self.ornament()
        self.scaffolds()
        self.hoarding_and_yard()
        self.hoist()
        self.plaza_1915()
        self.derricks()
