"""The living city around the works: parcelled blocks of 1900-1930 eclectic
housing with roof terraces, the Casas Consistoriales, the Palacio de
Comunicaciones, the tall corner-cupola blocks of the late 1920s, the brick
bell tower of San Agustín, trees, lamps, tram tracks and overhead wires."""
import math
import random

import mapbox_earcut as earcut
import numpy as np
from shapely.geometry import LineString, Point, Polygon, MultiPolygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union
from shapely import make_valid, set_precision

from . import plan
from .common import MB, L, jitter_col, new_coll, obj_from_mesh

FACADE_COLS = [L(232, 220, 196), L(214, 182, 134), L(236, 214, 162), L(222, 172, 142),
               L(228, 194, 178), L(240, 234, 222), L(204, 196, 182), L(208, 146, 108),
               L(198, 204, 204), L(226, 206, 170), L(190, 160, 120)]
TERRACE_COLS = [L(196, 150, 120), L(214, 190, 160), L(186, 140, 110), L(222, 214, 200)]

VIEW_C = (8.0, 0.0)


# ---------------------------------------------------------------------------
# shapely -> mesh
# ---------------------------------------------------------------------------
def largest(g):
    if g.is_empty:
        return None
    if isinstance(g, Polygon):
        return g
    polys = [p for p in getattr(g, "geoms", []) if isinstance(p, Polygon)]
    return max(polys, key=lambda p: p.area) if polys else None


def polys_of(g):
    if g.is_empty:
        return []
    if isinstance(g, Polygon):
        return [g]
    return [p for p in getattr(g, "geoms", []) if isinstance(p, Polygon)]


def top_face(mb, poly, z, col, mat):
    poly = orient(poly, 1.0)
    rings = [np.array(poly.exterior.coords)[:-1]]
    rings += [np.array(r.coords)[:-1] for r in poly.interiors]
    verts = np.concatenate(rings).astype(np.float64)
    ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
    tri = earcut.triangulate_float64(verts, ends).reshape(-1, 3)
    b = mb.add_verts([(x, y, z) for x, y in verts])
    for t in tri:
        mb.face((b + int(t[0]), b + int(t[1]), b + int(t[2])), col, mat)


def shp_prism(mb, poly, z0, z1, col, wall_mat, top_mat=None, top_col=None,
              wall_fn=None):
    """Extrude a shapely polygon (holes allowed).  wall_fn(a, b) -> (mat, col)
    lets callers pick facade vs. party-wall per edge."""
    poly = orient(poly, 1.0)
    for ring in [poly.exterior] + list(poly.interiors):
        c = list(ring.coords)[:-1]
        u = 0.0
        for k in range(len(c)):
            a, b = c[k], c[(k + 1) % len(c)]
            m, cc = (wall_mat, col) if wall_fn is None else wall_fn(a, b)
            u += mb.quad_wall(a, b, z0, z1, cc, m, u)
    if top_mat is not None:
        top_face(mb, poly, z1, top_col or col, top_mat)


# ---------------------------------------------------------------------------
class City:
    def __init__(self, M, rng):
        self.M = M
        self.rng = rng
        self.coll = new_coll("city")
        self.paths_walk = []      # closed sidewalk loops for pedestrians
        self.trees = []           # (x, y, z, scale, kind)
        self.flags = []
        self.roof_life = []       # laundry etc.
        self.bounds_r = 270.0

    # -- ground -------------------------------------------------------------
    def ground(self):
        mb = MB()
        g = Polygon([(-plan.EXT, -plan.EXT), (plan.EXT, -plan.EXT),
                     (plan.EXT, plan.EXT), (-plan.EXT, plan.EXT)])
        g = g.difference(plan.ISLAND)
        top_face(mb, g, 0.0, (1, 1, 1), 0)
        self.ground_obj = mb.build("ground", [self.M["cobble"]], self.coll)

    def sidewalks(self):
        mb = MB()
        curb = L(170, 162, 148)
        for blk in self.blocks:
            inner = plan.footprint(blk)
            ring = blk.difference(inner) if inner is not None else blk
            for p in polys_of(ring):
                shp_prism(mb, p, 0.0, 0.16, curb, 1, top_mat=0,
                          top_col=jitter_col(L(214, 204, 186), self.rng, 0.05))
            loop = blk.buffer(-1.8, join_style=2)
            if not loop.is_empty and isinstance(loop, Polygon):
                self.paths_walk.append(list(loop.exterior.coords))
        mb.build("sidewalks", [self.M["sidewalk"], self.M["plain"]], self.coll)

    # -- blocks and lots ----------------------------------------------------
    def make_lots(self, F, depth):
        rng = self.rng
        F = orient(F.simplify(0.4), 1.0)
        inner = F.buffer(-depth, join_style=2)
        inner = largest(inner) if not inner.is_empty else None
        if inner is not None and inner.area < 120:
            inner = None
        ring = F.difference(inner) if inner is not None else F
        coords = list(F.exterior.coords)[:-1]
        n = len(coords)
        taken = None
        lots = []
        for i in range(n):
            a = np.array(coords[i])
            b = np.array(coords[(i + 1) % n])
            d = b - a
            ln = np.linalg.norm(d)
            if ln < 2:
                continue
            t = d / ln
            inw = np.array([-t[1], t[0]])
            k = max(1, int(round(ln / rng.uniform(9, 17))))
            cuts = [0.0] + sorted(min(max((j + rng.uniform(-0.25, 0.25)) / k, 0.05), 0.95)
                                  for j in range(1, k)) + [1.0]
            for j in range(len(cuts) - 1):
                p0 = a + d * cuts[j]
                p1 = a + d * cuts[j + 1]
                q = Polygon([tuple(p0), tuple(p1), tuple(p1 + inw * (depth + 40)),
                             tuple(p0 + inw * (depth + 40))])
                if not q.is_valid:
                    continue
                lot = set_precision(q.intersection(ring), 0.01)
                if taken is not None:
                    lot = set_precision(make_valid(lot).difference(taken), 0.01)
                lot = largest(make_valid(lot))
                if lot is None or lot.area < 12:
                    continue
                taken = lot if taken is None else set_precision(
                    make_valid(taken.union(lot)), 0.01)
                lots.append((lot, tuple((p0 + p1) / 2)))
        rest = make_valid(ring.difference(taken)) if taken is not None else ring
        for p in polys_of(rest):
            if p.area > 12:
                lots.append((p, tuple(np.array(p.exterior.coords[0]))))
        return F, inner, lots

    def lot_building(self, mb_f, mb_r, lot, F, inner, H, col, seed, roof="terrace",
                     cornice=True):
        """One parcel: facade walls on street/courtyard edges, blank party
        walls elsewhere, cornice, parapet and roof clutter."""
        rng = self.rng
        ext = F.exterior
        inn = inner.exterior if inner is not None else None
        fcol = (*col, seed)
        party = jitter_col(col, rng, 0.12)
        lot = orient(lot.simplify(0.05), 1.0)

        def wall_fn(a, b):
            m = Point((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            if ext.distance(m) < 0.15 or (inn is not None and inn.distance(m) < 0.15):
                return 0, fcol
            return 1, (*party, 1.0)

        shp_prism(mb_f, lot, 0.0, H, fcol, 0, wall_fn=wall_fn)
        c = list(lot.exterior.coords)[:-1]
        # parapet (thin ring) and roof
        inner_p = lot.buffer(-0.3, join_style=2)
        tcol = TERRACE_COLS[int(seed * 997) % len(TERRACE_COLS)]
        if roof == "tiles" and len(c) <= 6:
            cen = np.array(lot.centroid.coords[0])
            top = [tuple(cen + (np.array(p) - cen) * 0.55) for p in c]
            rcol = jitter_col(L(176, 92, 60), rng, 0.1)
            for k in range(len(c)):
                a, b = c[k], c[(k + 1) % len(c)]
                ta, tb = top[k], top[(k + 1) % len(c)]
                i = mb_r.add_verts([(*a, H), (*b, H), (*tb, H + 3.2), (*ta, H + 3.2)])
                d = math.dist(a, b)
                mb_r.face((i, i + 1, i + 2, i + 3), rcol, 1,
                          [(0, 0), (d, 0), (d, 3.2), (0, 3.2)])
            i = mb_r.add_verts([(*p, H + 3.2) for p in top])
            mb_r.face(tuple(range(i, i + len(top))), rcol, 1)
        else:
            if isinstance(inner_p, Polygon) and not inner_p.is_empty:
                ring = lot.difference(inner_p)
                for p in polys_of(ring):
                    shp_prism(mb_r, p, H, H + 1.0, (*party, 1), 2, top_mat=2,
                              top_col=L(220, 210, 190))
                top_face(mb_r, inner_p, H + 0.05, tcol, 0)
            else:
                top_face(mb_r, lot, H + 0.05, tcol, 0)
            self.roof_clutter(mb_r, lot, H, party)
        if cornice:
            for k in range(len(c)):
                a, b = c[k], c[(k + 1) % len(c)]
                m = Point((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
                if ext.distance(m) > 0.15:
                    continue
                d = np.array(b) - np.array(a)
                ln = np.linalg.norm(d)
                if ln < 1:
                    continue
                t = d / ln
                out = np.array([t[1], -t[0]])
                mid = (np.array(a) + np.array(b)) / 2 + out * 0.3
                ang = math.atan2(t[1], t[0])
                mb_r.box((mid[0], mid[1], H - 0.55), (ln + 0.6, 0.6, 0.45),
                         L(236, 228, 210), 2, rot=ang)
                mb_r.box((mid[0], mid[1], 4.15), (ln, 0.5, 0.25), L(220, 210, 190), 2,
                         rot=ang)

    def roof_clutter(self, mb, lot, H, col):
        rng = self.rng
        minx, miny, maxx, maxy = lot.bounds
        for _ in range(rng.randint(1, 3)):
            for _try in range(6):
                p = (rng.uniform(minx, maxx), rng.uniform(miny, maxy))
                if lot.buffer(-2.2).contains(Point(p)):
                    break
            else:
                continue
            r = rng.random()
            if r < 0.4:     # stair housing (casetón)
                mb.box((p[0], p[1], H), (rng.uniform(2.4, 3.6), rng.uniform(2.4, 3.4), 2.6),
                       jitter_col(col, rng, 0.1), 2, rot=rng.uniform(0, 3))
            elif r < 0.65:  # water tank
                mb.cyl((p[0], p[1], H), rng.uniform(0.5, 0.8), 1.3, 8,
                       L(170, 166, 156), 2)
            elif r < 0.85:  # chimney stack
                mb.box((p[0], p[1], H), (0.7, 0.7, 2.0), L(176, 110, 82), 2)
            else:           # laundry line
                ang = rng.uniform(0, math.pi)
                ln = rng.uniform(3, 6)
                dx, dy = math.cos(ang) * ln / 2, math.sin(ang) * ln / 2
                self.roof_life.append(((p[0] - dx, p[1] - dy, H + 1.7),
                                       (p[0] + dx, p[1] + dy, H + 1.7)))
                for s in (-1, 1):
                    mb.beam((p[0] + s * dx, p[1] + s * dy, H),
                            (p[0] + s * dx, p[1] + s * dy, H + 1.8), 0.06,
                            L(120, 110, 100), 2)

    def buildings(self):
        rng = self.rng
        mb_f = MB()
        mb_r = MB()
        self.blocks = [b for b in plan.BLOCKS
                       if b.distance(Point(VIEW_C)) < self.bounds_r]
        lm = Landmarks(self)
        for bi, blk in enumerate(self.blocks):
            F0 = plan.footprint(blk)
            if F0 is None:
                continue
            dist = F0.distance(Point(VIEW_C))
            depth = 13.0 + rng.uniform(-1, 3)
            base_floors = rng.choice([4, 5, 5, 6, 6, 7])
            F, inner, lots = self.make_lots(F0, depth)
            for lot, anchor in lots:
                kind = lm.claim(lot, anchor)
                if kind == "skip":
                    continue
                floors = max(3, base_floors + rng.choice([-1, 0, 0, 1, 1, 2]))
                col = rng.choice(FACADE_COLS)
                if kind == "ateneo":
                    floors = rng.choice([8, 9])
                    col = jitter_col(L(238, 234, 224), rng, 0.03)
                H = 4.6 + floors * 3.3
                roof = "tiles" if rng.random() < 0.18 else "terrace"
                self.lot_building(mb_f, mb_r, lot, F, inner, H, col, rng.random(),
                                  roof=roof)
                if kind == "ateneo":
                    lm.ateneo_corner(lot, anchor, H)
        lm.build_all()
        self.bld = mb_f.build("buildings", [self.M["facade"], self.M["stucco"]], self.coll)
        self.roofs = mb_r.build("roofs", [self.M["terrace"], self.M["rooftile"],
                                          self.M["plain"]], self.coll)

    # -- street furniture ---------------------------------------------------
    def street_trees_and_lamps(self):
        rng = self.rng
        mb = MB()
        # trees along the plaza walks (two rows) and the side streets
        for blk in self.blocks:
            near_plaza = blk.distance(plan.PLAZA) < 1.0
            for off, step, prob in ((1.4, 8.5, 1.0), (5.6, 9.5, 0.9)) if near_plaza else \
                    ((1.2, 15.0, 0.5),):
                ring = blk.buffer(-off, join_style=2)
                if ring.is_empty or ring.geom_type != "Polygon":
                    continue
                ln = ring.exterior
                n = int(ln.length / step)
                for k in range(n):
                    pt = ln.interpolate(k * step + rng.uniform(-0.6, 0.6))
                    if pt.distance(Point(VIEW_C)) > 240:
                        continue
                    on_plaza = pt.distance(plan.PLAZA) < off + 0.5
                    if near_plaza and not on_plaza:
                        continue
                    if rng.random() > prob:
                        continue
                    kind = "palm" if (pt.x > 60 and on_plaza and off < 2 and rng.random() < 0.7) \
                        else "tree"
                    self.trees.append((pt.x, pt.y, 0.16, rng.uniform(0.85, 1.15), kind))
                    if near_plaza and off < 2 and k % 2 == 1:
                        lq = ln.interpolate(k * step + step / 2)
                        lamp(mb, (lq.x, lq.y, 0.16))
        # orange trees and gardens in the block courtyards
        for blk in self.blocks:
            f = plan.footprint(blk)
            if f is None:
                continue
            inner = f.buffer(-16, join_style=2)
            if inner.is_empty or inner.geom_type != "Polygon" or inner.area < 150:
                continue
            minx, miny, maxx, maxy = inner.bounds
            for _ in range(int(inner.area / 90)):
                p = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
                if inner.contains(p) and p.distance(Point(VIEW_C)) < 240:
                    self.trees.append((p.x, p.y, 0.0, rng.uniform(0.6, 0.9), "tree"))
        # kiosks, benches on the plaza sidewalks
        for x, side in ((-110, 1), (-40, -1), (20, 1), (-140, -1)):
            y = (plan.y_n(x) + 3.5) if side > 0 else (plan.y_s(x) - 3.5)
            kiosk(mb, (x, y, 0.16), rng)
        self.furn = mb.build("street_furniture", [self.M["metal"], self.M["paint"],
                                                  self.M["plain"]], self.coll)

    def tram_lines(self):
        """Rails around the island and along both arms of the plaza, with
        overhead wire on poles."""
        mb = MB()
        loop = plan.ISLAND.buffer(8.5, join_style=1).exterior
        self.tram_loop = list(loop.coords)
        west = LineString([(-300, -1.5), (plan.X_MIN - 8.5, -1.5)])
        east = LineString([(plan.X_MAX + 8.5, 1.5), (330, 1.5)])
        rails = [loop, west, east]
        # rails as thin strips: difference of buffers
        for ln in rails:
            for off in (0.72,):
                band = ln.buffer(off + 0.035, join_style=1).difference(
                    ln.buffer(off - 0.035, join_style=1))
                for p in polys_of(band):
                    shp_prism(mb, p, 0.0, 0.025, L(70, 66, 62), 0, top_mat=0,
                              top_col=L(160, 160, 165))
        # overhead line: poles + wire
        wire_z = 6.2
        for ln in rails:
            n = max(2, int(ln.length / 26))
            pts = [ln.interpolate(ln.length * k / n) for k in range(n + (0 if ln.is_ring else 1))]
            for p in pts:
                # pole on the outer side
                if ln.is_ring:
                    q = Point(p)
                    cen = plan.ISLAND.centroid
                    d = np.array([q.x - cen.x, q.y - cen.y])
                    o = plan.ISLAND.exterior.interpolate(plan.ISLAND.exterior.project(q))
                    v = np.array([q.x - o.x, q.y - o.y])
                    v = v / (np.linalg.norm(v) + 1e-9)
                    base = (q.x + v[0] * 4.0, q.y + v[1] * 4.0)
                else:
                    base = (p.x, p.y + 4.0)
                mb.cyl((base[0], base[1], 0), 0.12, wire_z + 0.6, 6, L(60, 66, 62), 0, r2=0.09)
                mb.beam((base[0], base[1], wire_z + 0.3), (p.x, p.y, wire_z + 0.2), 0.05,
                        L(60, 66, 62), 0)
            seq = pts + ([pts[0]] if ln.is_ring else [])
            for a, b in zip(seq[:-1], seq[1:]):
                mb.beam((a.x, a.y, wire_z), (b.x, b.y, wire_z), 0.035, L(50, 50, 50), 0)
        self.rails = mb.build("tram_lines", [self.M["metal"]], self.coll)

    def build_trees(self):
        """Linked duplicates of a few tree variants (they sway per frame)."""
        rng = self.rng
        coll = new_coll("trees")
        variants = {"tree": [], "palm": []}
        for k in range(6):
            me = tree_mesh(rng, "tree", k).mesh("tree_%d" % k)
            me.materials.append(self.M["foliage"])
            me.materials.append(self.M["wood"])
            variants["tree"].append(me)
        for k in range(3):
            me = tree_mesh(rng, "palm", k).mesh("palm_%d" % k)
            me.materials.append(self.M["foliage"])
            me.materials.append(self.M["wood"])
            variants["palm"].append(me)
        self.tree_objs = []
        for i, (x, y, z, s, kind) in enumerate(self.trees):
            me = rng.choice(variants[kind])
            ob = obj_from_mesh("t%d" % i, me, coll, (x, y, z), rng.uniform(0, 6.28), s)
            self.tree_objs.append((ob, rng.uniform(0, 6.28), rng.uniform(0.6, 1.1)))

    def animate(self, t):
        for ob, ph, f in self.tree_objs:
            a = 0.012 * math.sin(t * f * 1.3 + ph) + 0.008 * math.sin(t * 2.9 + ph * 2)
            ob.rotation_euler[0] = a
            ob.rotation_euler[1] = 0.7 * a * math.cos(ph)

    def build(self):
        self.ground()
        self.buildings()
        self.sidewalks()
        self.street_trees_and_lamps()
        self.tram_lines()
        self.build_trees()


# ---------------------------------------------------------------------------
def lamp(mb, p):
    """Cast-iron 'fernandina'-type street lamp."""
    mb.cyl((p[0], p[1], p[2]), 0.22, 0.5, 6, L(40, 44, 42), 0, r2=0.12)
    mb.cyl((p[0], p[1], p[2] + 0.5), 0.07, 3.6, 6, L(40, 44, 42), 0, r2=0.05)
    mb.cyl((p[0], p[1], p[2] + 4.1), 0.18, 0.5, 6, L(236, 228, 196), 1, r2=0.25)
    mb.cone((p[0], p[1], p[2] + 4.6), 0.28, 0.3, 6, L(40, 44, 42), 0)


def kiosk(mb, p, rng):
    col = rng.choice([L(60, 96, 70), L(120, 50, 40), L(70, 70, 96)])
    mb.cyl(p, 1.4, 2.4, 8, col, 1)
    mb.cyl((p[0], p[1], p[2] + 2.4), 1.8, 0.15, 8, L(50, 54, 50), 0)
    mb.cone((p[0], p[1], p[2] + 2.55), 1.8, 0.8, 8, col, 1)


def tree_mesh(rng, kind="tree", variant=0):
    """Round-crowned street tree (plátano / naranjo) or Washingtonia palm.
    Mat slots: 0 foliage, 1 wood (trunk)."""
    mb = MB()
    if kind == "palm":
        h = rng.uniform(10, 14)
        mb.cyl((0, 0, 0), 0.32, h, 7, L(120, 104, 86), 1, r2=0.24)
        for k in range(14):
            a = 2 * math.pi * k / 14 + rng.uniform(-0.1, 0.1)
            droop = rng.uniform(0.2, 0.9)
            ln = rng.uniform(2.6, 3.4)
            pts = []
            for j in range(5):
                t = j / 4
                r = ln * t
                z = h + 0.4 + math.sin(t * 2.2) * 0.9 - droop * t * t * 2.0
                pts.append((r * math.cos(a), r * math.sin(a), z))
            w = 0.55
            for j in range(4):
                p0, p1 = np.array(pts[j]), np.array(pts[j + 1])
                side = np.array([-math.sin(a), math.cos(a), 0]) * w * (1 - j / 5)
                side1 = np.array([-math.sin(a), math.cos(a), 0]) * w * (1 - (j + 1) / 5)
                i = mb.add_verts([p0 - side, p0 + side, p1 + side1, p1 - side1])
                mb.face((i, i + 1, i + 2, i + 3), jitter_col(L(96, 116, 54), rng, 0.15), 0)
        mb.sphere((0, 0, h + 0.2), 0.7, n=6, rings=4, col=L(130, 116, 70), sz=0.8)
        return mb
    h = rng.uniform(3.0, 4.2)
    mb.cyl((0, 0, 0), 0.2, h, 6, L(110, 98, 84), 1, r2=0.14)
    base = jitter_col(L(74, 104, 46) if variant % 2 else L(92, 118, 52), rng, 0.12)
    for k in range(rng.randint(4, 6)):
        c = (rng.uniform(-1.3, 1.3), rng.uniform(-1.3, 1.3), h + rng.uniform(0.4, 2.2))
        mb.sphere(c, rng.uniform(1.4, 2.2), n=9, rings=6, col=base, sz=0.85,
                  noise=0.14, rng=rng)
    return mb


class Landmarks:
    """Claims parcels along the plaza for the civic buildings."""

    def __init__(self, city):
        self.city = city
        self.M = city.M
        self.rng = city.rng
        self.mb = MB()     # mats: 0 facade 1 ashlar 2 plain 3 glass 4 metal 5 rooftile 6 flag
        self.ateneo_done = set()
        self.ayto_lots = []
        self.correos_lots = []

    def claim(self, lot, anchor):
        x, y = anchor
        if plan.AYTO_X[0] <= x <= plan.AYTO_X[1] and abs(y - plan.y_s(x)) < plan.PLAZA_SW + 30 \
                and y < 0:
            self.ayto_lots.append(lot)
            return "skip"
        if plan.CORREOS_X[0] <= x <= plan.CORREOS_X[1] and y > 0 and \
                abs(y - plan.y_n(x)) < plan.PLAZA_SW + 24:
            self.correos_lots.append(lot)
            return "skip"
        if plan.ATENEO_X[0] <= x <= plan.ATENEO_X[1] and y > 0 and \
                abs(y - plan.y_n(x)) < plan.PLAZA_SW + 2:
            return "ateneo"
        return "lot"

    # --- Casas Consistoriales -------------------------------------------------
    def ayuntamiento(self):
        mb = self.mb
        rng = self.rng
        if not self.ayto_lots:
            return
        body = unary_union(self.ayto_lots).buffer(0.01).buffer(-0.01)
        body = largest(body)
        stone = L(224, 200, 160)
        trim = L(242, 232, 210)
        H = 4.6 + 5 * 3.3
        shp_prism(mb, body, 0, H, (*stone, 0.37), 0, top_mat=2, top_col=L(200, 170, 140))
        # frontage geometry
        xa, xb = plan.AYTO_X[0] + 2, plan.AYTO_X[1] - 2
        A = np.array([xa, plan.y_s(xa) - plan.PLAZA_SW])
        B = np.array([xb, plan.y_s(xb) - plan.PLAZA_SW])
        t = (B - A) / np.linalg.norm(B - A)
        out = np.array([t[1] * -1, t[0]])   # towards the plaza (north)
        if out[1] < 0:
            out = -out
        ang = math.atan2(t[1], t[0])
        mid = (A + B) / 2
        # balustraded attic over the cornice
        ln = np.linalg.norm(B - A)
        mb.box((*(mid + out * 0.3), H - 0.6), (ln + 1, 1.2, 0.6), trim, 2, rot=ang)
        mb.box((*(mid + out * 0.05), H), (ln, 0.5, 1.4), trim, 1, rot=ang)
        # rusticated base and giant-order pilasters along the frontage
        mb.box((*(mid + out * 0.25), 0), (ln, 0.5, 4.6), L(214, 190, 150), 1, rot=ang)
        mb.box((*(mid + out * 0.3), 4.5), (ln + 0.4, 0.7, 0.35), trim, 2, rot=ang)
        n_p = int(ln / 7.2)
        for k in range(n_p + 1):
            q = A + (B - A) * k / n_p + out * 0.2
            mb.box((*q, 4.85), (0.9, 0.45, H - 5.6), trim, 1, rot=ang)
        for z in (8.0, 14.6):   # balcony lines
            mb.box((*(mid + out * 0.45), z), (ln - 2, 0.9, 0.22), trim, 2, rot=ang)
        # central projecting body and great tower
        c = mid + out * 2.5
        mb.box((*c, 0), (26, 7, H + 3), (*stone, 0.61), 0, rot=ang)
        mb.box((*(c + out * 3.8), 8.0), (20, 2.4, 0.35), trim, 2, rot=ang)   # balcony
        for k in range(6):   # paired columns of the main balcony bay
            q = c + out * 3.7 + t * (k - 2.5) * 4.0
            mb.cyl((*q, 8.35), 0.35, 9.0, 8, trim, 1)
        # pediment over the central body
        pc = c + out * 3.4
        g0 = pc - t * 12
        g1 = pc + t * 12
        i = mb.add_verts([(*g0, H + 3), (*g1, H + 3), (*pc, H + 8.5)])
        mb.face((i, i + 1, i + 2), trim, 1, [(0, H + 3), (24, H + 3), (12, H + 8.5)])
        i = mb.add_verts([(*(g0 - out * 6.8), H + 3), (*g0, H + 3), (*pc, H + 8.5),
                          (*(pc - out * 6.8), H + 8.5)])
        mb.face((i, i + 1, i + 2, i + 3), L(150, 120, 90), 5)
        i = mb.add_verts([(*g1, H + 3), (*(g1 - out * 6.8), H + 3), (*(pc - out * 6.8), H + 8.5),
                          (*pc, H + 8.5)])
        mb.face((i, i + 1, i + 2, i + 3), L(150, 120, 90), 5)
        tw = c - out * 2
        mb.box((*tw, 0), (14, 11, 42), (*stone, 0.83), 0, rot=ang)
        mb.box((*tw, 41.5), (15, 12, 1.0), trim, 2, rot=ang)
        mb.box((*tw, 42.5), (10, 9, 6.5), (*trim, 0.2), 1, rot=ang)
        for s in (-1, 1):   # clock faces
            p = tw + out * 4.55 * s
            disc = MB()
            disc.cyl((0, 0, 0), 2.0, 0.1, 16, L(246, 240, 226), 2)
            for i, v in enumerate(disc.v):
                disc.v[i] = (v[0], v[2], v[1])
            mb.merge(disc, offset=(p[0], p[1], 45.8), rot=ang)
        mb.cyl((*tw, 49.0), 4.0, 1.0, 12, trim, 2)
        mb.dome((*tw, 50.0), 3.8, n=12, rings=5, col=L(150, 120, 90), mat=5, squash=1.25)
        mb.cyl((*tw, 54.6), 0.9, 2.4, 8, trim, 2, r2=0.7)
        mb.dome((*tw, 57.0), 0.9, n=8, rings=3, col=L(90, 80, 60), mat=4)
        mb.beam((*tw, 57.8), (*tw, 61.0), 0.12, L(40, 40, 40), 4)
        # corner towers with cupolas
        for s, pt in ((-1, A), (1, B)):
            q = pt - t * 3 * s + out * 1.0
            mb.box((*q, 0), (11, 11, 31.5), (*stone, 0.7 + 0.1 * s), 0, rot=ang)
            mb.box((*q, 31), (12, 12, 0.9), trim, 2, rot=ang)
            mb.cyl((*q, 31.9), 3.6, 2.6, 12, trim, 1)
            mb.dome((*q, 34.5), 3.6, n=12, rings=5, col=L(116, 130, 120), mat=5, squash=1.3)
            mb.cyl((*q, 39.0), 0.6, 1.6, 8, trim, 2)
        # flags on the balcony
        for s in (-1, 0, 1):
            p = c + out * 4.6 + t * s * 6
            self.city.flags.append(((p[0], p[1], 8.4), ang, out))
        self.ayto_front = (A, B, out)

    # --- Palacio de Comunicaciones ------------------------------------------
    def correos(self):
        mb = self.mb
        if not self.correos_lots:
            return
        body = largest(unary_union(self.correos_lots).buffer(0.01).buffer(-0.01))
        stone = L(232, 220, 204)
        H = 4.6 + 6 * 3.3
        shp_prism(mb, body, 0, H, (*stone, 0.52), 0, top_mat=2, top_col=L(196, 176, 150))
        xa, xb = plan.CORREOS_X[0] + 2, plan.CORREOS_X[1] - 2
        A = np.array([xa, plan.y_n(xa) + plan.PLAZA_SW])
        B = np.array([xb, plan.y_n(xb) + plan.PLAZA_SW])
        t = (B - A) / np.linalg.norm(B - A)
        out = np.array([t[1], -t[0]])
        ang = math.atan2(t[1], t[0])
        mid = (A + B) / 2 + t * 4
        ln = np.linalg.norm(B - A)
        mb.box((*(A + B) / 2 + out * 0.3, H - 0.7), (ln, 1.0, 0.7), L(244, 238, 226), 2, rot=ang)
        # monumental central body with arch and crowned tower
        c = mid + out * 1.8
        mb.box((*c, 0), (20, 5, H + 6), (*stone, 0.12), 1, rot=ang)
        mb.box((*(c + out * 2.6), 2.0), (9, 0.3, 13), L(40, 46, 54), 3, rot=ang)  # glazed arch
        tw = c - out * 2
        mb.box((*tw, H + 5.5), (11, 9, 9), (*stone, 0.3), 0, rot=ang)
        mb.cyl((*tw, H + 14.5), 3.2, 3.0, 12, L(240, 232, 214), 1)
        # iron crown
        for k in range(12):
            a = 2 * math.pi * k / 12
            p0 = (tw[0] + 3.0 * math.cos(a), tw[1] + 3.0 * math.sin(a), H + 17.5)
            p1 = (tw[0] + 1.0 * math.cos(a), tw[1] + 1.0 * math.sin(a), H + 23.5)
            mb.beam(p0, p1, 0.18, L(70, 84, 80), 4)
        mb.cyl((*tw, H + 23.5), 1.1, 0.5, 10, L(70, 84, 80), 4)
        mb.beam((*tw, H + 24), (*tw, H + 28.5), 0.15, L(60, 70, 66), 4)
        # glass dome over the public hall
        dc = (A + B) / 2 - out * 22
        mb.cyl((*dc, H), 11.5, 1.4, 24, L(214, 200, 180), 2)
        mb.dome((*dc, H + 1.4), 11.0, n=24, rings=7, col=L(120, 150, 150), mat=3, squash=0.62)
        for k in range(12):
            a = 2 * math.pi * k / 12
            pts = []
            for j in range(8):
                ph = (math.pi / 2) * j / 7
                pts.append((dc[0] + 11.1 * math.cos(ph) * math.cos(a),
                            dc[1] + 11.1 * math.cos(ph) * math.sin(a),
                            H + 1.4 + 11.1 * 0.62 * math.sin(ph)))
            for p0, p1 in zip(pts[:-1], pts[1:]):
                mb.beam(p0, p1, 0.22, L(56, 70, 66), 4)
        mb.cyl((*dc, H + 8.2), 1.4, 1.4, 10, L(56, 70, 66), 4)
        self.correos_front = (A, B, out)

    # --- late-1920s corner cupolas ------------------------------------------
    def ateneo_corner(self, lot, anchor, H):
        x = anchor[0]
        for cx in (plan.ATENEO_X[0] + 6, plan.ATENEO_X[1] - 6):
            if cx in self.ateneo_done:
                continue
            if lot.distance(Point(cx, plan.y_n(cx) + plan.PLAZA_SW + 3)) < 0.1:
                self.ateneo_done.add(cx)
                q = lot.centroid
                p = np.array([cx, plan.y_n(cx) + plan.PLAZA_SW + 4.5])
                mb = self.mb
                white = L(244, 240, 230)
                mb.cyl((*p, H), 4.6, 6.0, 16, (*white, 0.71), 0)
                mb.cyl((*p, H + 6.0), 5.0, 0.6, 16, white, 2)
                for k in range(10):   # ring of columns
                    a = 2 * math.pi * k / 10
                    mb.cyl((p[0] + 4.0 * math.cos(a), p[1] + 4.0 * math.sin(a), H + 6.6),
                           0.28, 4.0, 6, white, 1)
                mb.cyl((*p, H + 6.6), 3.2, 4.0, 12, L(60, 60, 64), 2)
                mb.cyl((*p, H + 10.6), 4.7, 0.7, 16, white, 2)
                mb.dome((*p, H + 11.3), 3.9, n=14, rings=5, col=L(236, 232, 222), mat=1,
                        squash=1.2)
                for k in range(8):   # pinnacles
                    a = 2 * math.pi * k / 8
                    mb.cone((p[0] + 4.3 * math.cos(a), p[1] + 4.3 * math.sin(a), H + 11.3),
                            0.35, 2.0, 6, white, 1)
                mb.cyl((*p, H + 15.9), 0.7, 1.8, 8, white, 1)
                mb.cone((*p, H + 17.7), 0.7, 1.6, 8, white, 1)

    def san_agustin_tower(self):
        """Brick bell tower to the south-west, a red accent in the skyline."""
        mb = self.mb
        p = (-70.0, -98.0)
        brick = L(170, 92, 60)
        mb.box((*p, 0), (7.5, 7.5, 34), brick, 2)
        mb.box((*p, 34), (8.4, 8.4, 0.8), L(220, 200, 170), 2)
        mb.cyl((*p, 34.8), 3.6, 6.0, 8, brick, 2, r2=3.4)
        mb.cyl((*p, 40.8), 3.9, 0.6, 8, L(220, 200, 170), 2)
        mb.cone((*p, 41.4), 3.4, 4.5, 8, L(150, 90, 60), 5)
        mb.beam((*p, 45.9), (*p, 48.0), 0.12, L(40, 40, 40), 4)

    def build_all(self):
        self.ayuntamiento()
        self.correos()
        self.san_agustin_tower()
        M = self.M
        self.obj = self.mb.build("landmarks", [M["facade"], M["ashlar"], M["plain"],
                                               M["glass"], M["metal"], M["rooftile"],
                                               M["flag"]], self.city.coll)
