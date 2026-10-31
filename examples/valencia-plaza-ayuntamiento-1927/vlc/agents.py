"""Everything that moves on its own: workers with roles tied to construction
progress, townsfolk on the walks, onlookers at the hoarding, trams, motor
cars, mule carts, pigeons, flags and washing on the roofs.

All trajectories are simulated up-front for every frame (deterministic),
then applied per frame by swapping pose meshes on linked objects."""
import math
import random

import numpy as np
from shapely.geometry import LineString, Point, Polygon

from . import plan, site
from . import figures as F
from .common import MB, L, new_coll, obj_from_mesh, smooth, clamp, ramp


def sample_in(poly, rng, tries=60):
    minx, miny, maxx, maxy = poly.bounds
    for _ in range(tries):
        p = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
        if poly.contains(p):
            return np.array([p.x, p.y])
    c = poly.representative_point()
    return np.array([c.x, c.y])


class PathLoop:
    def __init__(self, pts, closed=True):
        pts = [tuple(p[:2]) for p in pts]
        if closed and pts[0] != pts[-1]:
            pts.append(pts[0])
        self.p = np.array(pts, float)
        seg = np.linalg.norm(np.diff(self.p, axis=0), axis=1)
        self.cum = np.concatenate([[0], np.cumsum(seg)])
        self.length = self.cum[-1]
        self.closed = closed

    def at(self, s):
        s = s % self.length
        i = int(np.searchsorted(self.cum, s, side="right") - 1)
        i = min(max(i, 0), len(self.p) - 2)
        t = (s - self.cum[i]) / max(self.cum[i + 1] - self.cum[i], 1e-9)
        a, b = self.p[i], self.p[i + 1]
        d = b - a
        d = d / (np.linalg.norm(d) + 1e-9)
        return a + (b - a) * t, d


class Agent:
    def __init__(self, ob, n):
        self.ob = ob
        self.pos = np.zeros((n, 3))
        self.head = np.zeros(n)
        self.key = [None] * n
        self.alpha = np.ones(n)


class Crowd:
    def __init__(self, M, rng, times, pfun, fps):
        self.M = M
        self.rng = rng
        self.times = times
        self.n = len(times)
        self.pfun = pfun
        self.fps = fps
        self.coll = new_coll("people")
        self.agents = []
        styles, self.n_workers, lib = F.build_pose_library(rng)
        self.styles = styles
        self.meshes = {}
        for k, mb in lib.items():
            me = mb.mesh("pose_%d_%s_%d" % k)
            me.materials.append(M["people"])
            me.materials.append(M["wood"])
            self.meshes[k] = me
        self.worker_ids = [i for i, s in enumerate(styles) if s.kind == "worker"]
        self.citizen_ids = [i for i, s in enumerate(styles) if s.kind in ("man", "woman", "child")]
        self.foreman_id = [i for i, s in enumerate(styles) if s.kind == "foreman"][0]
        self.dust = []   # (frame, x, y, z, strength)

    def new_agent(self, style):
        any_key = (style, "stand", 0)
        ob = obj_from_mesh("ag", self.meshes[any_key], self.coll)
        a = Agent(ob, self.n)
        a.style = style
        self.agents.append(a)
        return a

    # ------------------------------------------------------------------ util
    def _walk_key(self, style, mode, dist):
        k = int(dist / 1.45 * F.N_WALK) % F.N_WALK
        return (style, mode, k)

    def _work_key(self, style, mode, t, rate=1.3, ph=0.0):
        k = int((t * rate + ph) * F.N_WORK) % F.N_WORK
        return (style, mode, k)

    # ------------------------------------------------------------- wanderer
    def wanderer(self, style, region_fn, p_in, p_out, walk_mode="walk",
                 work_modes=("work",), speed=1.15, work_time=(1.5, 4.5), z_fn=None,
                 face_fn=None, fade=0.35):
        rng = self.rng
        a = self.new_agent(style)
        pos = None
        target = None
        state = "work"
        timer = 0.0
        dist = rng.uniform(0, 5)
        head = rng.uniform(0, 6.28)
        mode = rng.choice(work_modes)
        ph = rng.random()
        t_on = None
        for i, t in enumerate(self.times):
            p = self.pfun(t)
            active = p_in <= p < p_out
            if t_on is None and active:
                t_on = t
            reg = region_fn(p)
            if pos is None:
                pos = sample_in(reg, rng)
            if active:
                if state == "walk":
                    d = target - pos
                    ln = np.linalg.norm(d)
                    step = speed / self.fps
                    if ln < step:
                        pos = target.copy()
                        state = "work"
                        timer = rng.uniform(*work_time)
                        mode = rng.choice(work_modes)
                    else:
                        pos = pos + d / ln * step
                        dist += step
                        head = math.atan2(d[1], d[0]) - math.pi / 2
                else:
                    timer -= 1.0 / self.fps
                    if face_fn is not None:
                        head = face_fn(pos, p)
                    if timer <= 0:
                        target = sample_in(reg, rng)
                        state = "walk"
                if not reg.buffer(0.5).contains(Point(*pos)):
                    pos = sample_in(reg, rng)
            z = z_fn(pos, p) if z_fn else 0.0
            a.pos[i] = (pos[0], pos[1], z)
            a.head[i] = head
            if state == "walk":
                a.key[i] = self._walk_key(style, walk_mode, dist)
            else:
                a.key[i] = self._work_key(style, mode, t, ph=ph)
            fin = clamp((p - p_in) / 0.006) if fade else float(active)
            fout = 1 - clamp((p - p_out) / 0.006) if fade else 1.0
            a.alpha[i] = min(fin, fout) if p >= p_in else 0.0
        return a

    # --------------------------------------------------------- construction
    def workers(self):
        rng = self.rng
        W = self.worker_ids
        isl = plan.ISLAND
        patio = Point(plan.PATIO_C).buffer(plan.PATIO_R + 2.0)
        pit = isl.buffer(-2.5).difference(patio)
        pitz = lambda pos, p: plan.PIT_Z + (0.12 if p > 0.1 else 0.0)
        deckz = lambda pos, p: plan.DECK_Z

        def deck_built(p, depth=None):
            fx = site.deck_front_x(p)
            x1 = plan.X_MAX + 5 if depth is None else fx + depth
            from shapely.geometry import box
            r = isl.buffer(-1.6).difference(patio).intersection(box(fx + 1.5, -80, x1, 80))
            if r.is_empty or r.area < 10:
                r = isl.buffer(-1.6).difference(patio).intersection(box(plan.X_MAX - 25, -80, 99, 80))
            return r if r.geom_type == "Polygon" else max(r.geoms, key=lambda g: g.area)

        # diggers on the spoil heaps
        for k in range(8):
            c = [(-20, 2, 7), (20, -6, 8), (40, 10, 6), (-45, -2, 5)][k % 4]
            reg = Point(c[0], c[1]).buffer(c[2] + 1.5).intersection(pit)
            self.wanderer(rng.choice(W), lambda p, reg=reg: reg, -1, 0.17 + 0.01 * k,
                          work_modes=("dig",), work_time=(3, 7), z_fn=pitz)
        # barrow men and plank carriers in the pit
        for k in range(7):
            self.wanderer(rng.choice(W), lambda p: pit, -1, 0.30 + 0.02 * k,
                          walk_mode="barrow", work_modes=("dig", "work"), speed=1.0,
                          work_time=(1, 3), z_fn=pitz)
        for k in range(6):
            self.wanderer(rng.choice(W), lambda p: pit, 0.10 + 0.01 * k, 0.42,
                          walk_mode="plank", work_modes=("work",), speed=1.0, z_fn=pitz)
        # masons following the rising wall front on both sides
        for k in range(14):
            self.mason(rng.choice(W), side=1 if k % 2 else -1, off=1.0 + (k // 2) * 1.6)
        for k in range(6):
            self.patio_mason(rng.choice(W), k)
        # hod carriers circling the hoarding
        ring = PathLoop(list(isl.buffer(3.3, join_style=1).exterior.coords))
        for k in range(10):
            self.follower(rng.choice(W), ring, rng.uniform(0, ring.length),
                          rng.choice((-1, 1)) * rng.uniform(0.9, 1.2), "hod",
                          p_in=0.08 + 0.01 * k, p_out=0.62 + 0.02 * k, lat=rng.uniform(-0.6, 0.6))
        # winch men at the derricks
        for cr in site.CRANES:
            (bx, by), _ = cr
            for s in (-1, 1):
                a = self.new_agent(rng.choice(W))
                for i, t in enumerate(self.times):
                    p = self.pfun(t)
                    a.pos[i] = (bx + 1.6 + s * 0.9, by - 1.5, plan.PIT_Z + 0.12)
                    a.head[i] = 0.0 if s < 0 else math.pi
                    a.head[i] = math.pi / 2 * s
                    a.key[i] = self._work_key(a.style, "work", t, rate=0.9, ph=0.5 * (s > 0))
                    a.alpha[i] = ramp(p, 0.14, 0.16) * (1 - ramp(p, 0.62, 0.64))
        # deck crew close to the setting front
        for k in range(12):
            self.wanderer(rng.choice(W), lambda p: deck_built(p, 14), 0.44, 0.70 + 0.004 * k,
                          work_modes=("work", "dig"), work_time=(2, 5), z_fn=deckz)
        # finishing crew and gardeners on the whole deck
        for k in range(12):
            self.wanderer(rng.choice(W), lambda p: deck_built(p), 0.60 + 0.005 * k,
                          0.88 + 0.004 * k, walk_mode=rng.choice(("walk", "hod", "barrow")),
                          work_modes=("work", "dig"), work_time=(2, 5), z_fn=deckz)
        for k, (c, aa, bb) in enumerate(plan.parterres()):
            reg = Point(c).buffer(aa + 1.4).difference(Point(c).buffer(0.3)).intersection(
                plan.ISLAND.buffer(-1.5))
            self.wanderer(rng.choice(W), lambda p, reg=reg: reg, 0.72 + 0.005 * k, 0.89,
                          work_modes=("dig",), work_time=(3, 6), z_fn=deckz)
        # yard hands sorting brick, ashlar and timber
        yard = site.COMPOUND.buffer(-1.0)
        for k in range(4):
            self.wanderer(rng.choice(W), lambda p: yard, -1, 0.86 + 0.01 * k, walk_mode="hod",
                          work_modes=("dig", "work"), work_time=(2, 5), speed=1.0)
        # foremen
        self.wanderer(self.foreman_id, lambda p: yard, -1, 0.93, work_modes=("plan", "point"),
                      work_time=(4, 8), speed=0.9, fade=0.3)
        self.wanderer(self.foreman_id, lambda p: deck_built(p), 0.5, 0.90,
                      work_modes=("point", "plan"), work_time=(4, 8), speed=0.9, z_fn=deckz)

    def mason(self, style, side, off):
        a = self.new_agent(style)
        H = 0.6 - plan.PIT_Z
        for i, t in enumerate(self.times):
            p = self.pfun(t)
            fx = site.wall_front_x(p) + off + 0.6 * math.sin(t * 0.4 + off)
            b = site.boundary_at(fx, side)
            b2 = site.boundary_at(fx + 0.5, side)
            d = b2 - b
            inw = np.array([-d[1], d[0]])
            inw = inw / (np.linalg.norm(inw) + 1e-9)
            if not plan.ISLAND.contains(Point(*(b + inw))):
                inw = -inw
            # wall height already laid at this spot
            prog = clamp((p - (0.10 + 0.28 * (plan.X_MAX - fx) / site.LEN)) / 0.075)
            top = plan.PIT_Z + H * prog
            z = plan.PIT_Z + 0.12
            for lv in (plan.PIT_Z + 1.9 * (k + 1) for k in range(3)):
                if lv + 1.0 < top:
                    z = lv + 0.1
            pos = b + inw * (0.95 + 0.3 * (int(off) % 2))
            a.pos[i] = (pos[0], pos[1], z)
            a.head[i] = math.atan2(-inw[1], -inw[0]) - math.pi / 2
            a.key[i] = self._work_key(style, "work", t, rate=1.1, ph=off)
            a.alpha[i] = ramp(p, 0.11, 0.125) * (1 - ramp(p, 0.42, 0.44))
            if i % 6 == 0 and a.alpha[i] > 0.5:
                self.dust.append((i, pos[0], pos[1], top, 0.5))

    def patio_mason(self, style, k):
        a = self.new_agent(style)
        c, r = plan.PATIO_C, plan.PATIO_R
        for i, t in enumerate(self.times):
            p = self.pfun(t)
            ang = 2 * math.pi * (k / 6 + (p - 0.2) * 0.8)
            prog = clamp((p - 0.20) / 0.18)
            z = plan.PIT_Z + 0.12
            if prog > 0.55:
                z = plan.PIT_Z + 1.9 + 0.1
            pos = (c[0] + (r - 1.4) * math.cos(ang), c[1] + (r - 1.4) * math.sin(ang))
            a.pos[i] = (*pos, z)
            a.head[i] = ang - math.pi / 2
            a.key[i] = self._work_key(style, "work", t, ph=k * 0.3)
            a.alpha[i] = ramp(p, 0.20, 0.21) * (1 - ramp(p, 0.39, 0.40))

    def follower(self, style, path, s0, speed, mode, p_in=-1, p_out=2, lat=0.0, z=0.0,
                 stop_prob=0.0):
        rng = self.rng
        a = self.new_agent(style)
        s = s0
        dist = rng.uniform(0, 3)
        pause = 0.0
        for i, t in enumerate(self.times):
            p = self.pfun(t)
            if pause > 0:
                pause -= 1 / self.fps
                key = (style, "stand", 0)
            else:
                s += speed / self.fps
                dist += abs(speed) / self.fps
                key = self._walk_key(style, mode, dist)
                if stop_prob and rng.random() < stop_prob / self.fps:
                    pause = rng.uniform(2, 6)
            pos, d = path.at(s)
            if speed < 0:
                d = -d
            nrm = np.array([-d[1], d[0]])
            pos = pos + nrm * lat
            a.pos[i] = (pos[0], pos[1], z)
            a.head[i] = math.atan2(d[1], d[0]) - math.pi / 2
            a.key[i] = key
            a.alpha[i] = ramp(p, p_in, p_in + 0.006) * (1 - ramp(p, p_out, p_out + 0.006)) \
                if p_in >= 0 else (1 - ramp(p, p_out, p_out + 0.006))
        return a

    # ------------------------------------------------------------- townsfolk
    def townsfolk(self, walk_loops):
        rng = self.rng
        C = self.citizen_ids
        view = Point(10, 0)
        loops = [PathLoop(l) for l in walk_loops
                 if Polygon(l).distance(view) < 140 and len(l) > 3]
        for lp in loops:
            n = max(2, int(lp.length / 26))
            for k in range(n):
                st = rng.choice(C)
                mode = "basket" if (st, "basket", 0) in self.meshes and rng.random() < 0.5 else "walk"
                self.follower(st, lp, rng.uniform(0, lp.length),
                              rng.choice((-1, 1)) * rng.uniform(1.0, 1.45), mode,
                              lat=rng.uniform(-0.9, 0.9), z=0.16, stop_prob=0.04)
        # onlookers watching the works through the hoarding gaps
        ring = plan.ISLAND.buffer(2.7, join_style=1).exterior
        cen = plan.ISLAND.centroid
        for k in range(18):
            q = ring.interpolate(rng.uniform(0, ring.length))
            o = plan.ISLAND.exterior.interpolate(plan.ISLAND.exterior.project(q))
            st = rng.choice(C)
            a = self.new_agent(st)
            ph = rng.uniform(0, 6)
            p_out = rng.uniform(0.86, 0.9)
            for i, t in enumerate(self.times):
                p = self.pfun(t)
                a.pos[i] = (q.x, q.y, 0.0)
                a.head[i] = math.atan2(o.y - q.y, o.x - q.x) - math.pi / 2 + 0.3 * math.sin(t * 0.3 + ph)
                a.key[i] = (st, "stand", 0)
                a.alpha[i] = 1 - ramp(p, p_out, p_out + 0.01)
        # citizens strolling on the finished platform
        deck = plan.ISLAND.buffer(-1.8).difference(Point(plan.PATIO_C).buffer(plan.PATIO_R + 1.2))
        for c, aa, bb in plan.parterres():
            deck = deck.difference(Polygon(plan.ellipse(c, aa + 0.4, bb + 0.4)))
        deck = deck.difference(Point(plan.C2[0] + 1.5, plan.C2[1]).buffer(3.8))
        for k in range(46):
            p_in = 0.905 + 0.07 * rng.random()
            self.wanderer(rng.choice(C), lambda p: deck, p_in, 9, walk_mode="walk",
                          work_modes=("stand",), work_time=(1, 6), speed=rng.uniform(0.9, 1.3),
                          z_fn=lambda pos, p: plan.DECK_Z)
        patio = Point(plan.PATIO_C).buffer(plan.PATIO_R - 1.2).difference(
            Point(plan.PATIO_C).buffer(2.4))
        for k in range(8):
            self.wanderer(rng.choice(C), lambda p: patio, 0.80 + 0.01 * k, 9,
                          work_modes=("stand",), work_time=(2, 6), speed=0.8,
                          z_fn=lambda pos, p: plan.PIT_Z + 0.14)

    # -------------------------------------------------------------- vehicles
    def vehicles(self, tram_loop):
        rng = self.rng
        self.vehicles_ = []
        coll = self.coll
        ring = PathLoop(tram_loop)
        # tram meshes
        tm = F.tram(rng).mesh("tram")
        tm.materials.append(self.M["people"])
        tm.materials.append(self.M["paint"])
        for k in range(2):
            ob = obj_from_mesh("tram", tm, coll)
            self.vehicles_.append((ob, ring, ring.length * k / 2 + 30, 5.2, 0.0, None))
        # motor cars on the inner lane around the works and along the bands
        inner = PathLoop(list(plan.ISLAND.buffer(5.0, join_style=1).exterior.coords))
        xs = np.linspace(-330, 100, 60)
        band_s = PathLoop([(x, plan.y_s(x) + 5.0) for x in xs] + [(103.0, -60.0), (103.0, -330.0)],
                          closed=False)
        band_n = PathLoop([(111.4, 330.0), (111.4, 62.0)] +
                          [(x, plan.y_n(x) - 5.0) for x in xs[::-1]], closed=False)
        sotelo = PathLoop([(330.0, -4.0), (110.0, -4.0), (110.0, 4.0), (330.0, 4.0)], closed=False)
        cols = [L(24, 24, 26), L(30, 46, 38), L(70, 30, 30), L(40, 40, 60), L(200, 190, 160),
                L(24, 24, 26), L(110, 96, 70)]
        car_meshes = []
        for k, c in enumerate(cols):
            me = F.car(c, rng).mesh("car%d" % k)
            me.materials.append(self.M["people"])
            me.materials.append(self.M["paint"])
            car_meshes.append(me)
        for path, n, sp in ((inner, 3, 6.0), (band_s, 5, 7.5), (band_n, 5, 7.5), (sotelo, 4, 8.0)):
            for k in range(n):
                ob = obj_from_mesh("car", rng.choice(car_meshes), coll)
                self.vehicles_.append((ob, path, path.length * (k + rng.uniform(0, 0.5)) / n,
                                       sp * rng.uniform(0.85, 1.1), 0.0, None))
        # mule carts: earth away early, bricks and ashlar later
        cart_meshes = {}
        for load in ("earth", "bricks", "barrels"):
            for ph in range(6):
                me = F.horse(L(110, 84, 60) if ph % 2 else L(110, 84, 60), 2 * math.pi * ph / 6,
                             cart=load, rng=rng).mesh("cart_%s_%d" % (load, ph))
                me.materials.append(self.M["people"])
                me.materials.append(self.M["wood"])
                cart_meshes[(load, ph)] = me
        self.cart_meshes = cart_meshes
        for path, n in ((band_s, 3), (band_n, 3)):
            for k in range(n):
                ob = obj_from_mesh("cart", cart_meshes[("earth", 0)], coll)
                self.vehicles_.append((ob, path, path.length * (k + 0.3) / n,
                                       1.7 * rng.uniform(0.9, 1.1), 0.0, "cart"))

    def vehicle_frame(self, fi, t, p):
        for ob, path, s0, sp, _, kind in self.vehicles_:
            s = s0 + sp * t
            if not path.closed:
                s = s % path.length
            pos, d = path.at(s)
            ob.location = (pos[0], pos[1], 0.0)
            ob.rotation_euler = (0, 0, math.atan2(d[1], d[0]) - math.pi / 2)
            if kind == "cart":
                load = "earth" if p < 0.3 else ("bricks" if p < 0.7 else "barrels")
                ph = int(s / 0.9) % 6
                ob.data = self.cart_meshes[(load, ph)]

    # ------------------------------------------------------------- pigeons
    def pigeons(self):
        rng = self.rng
        up = F.pigeon(True).mesh("pig_up")
        dn = F.pigeon(False).mesh("pig_dn")
        for me in (up, dn):
            me.materials.append(self.M["people"])
        self.pig_meshes = (up, dn)
        self.pigs = []
        for k in range(28):
            ob = obj_from_mesh("pigeon", up, self.coll)
            ob.scale = (1.4, 1.4, 1.4)
            self.pigs.append((ob, rng.uniform(-6, 6), rng.uniform(-6, 6), rng.uniform(-2, 2),
                              rng.uniform(0, 6.28), rng.uniform(7, 10)))

    def pigeon_frame(self, t):
        T = 20.0
        w = 2 * math.pi / T
        cx = 15 + 55 * math.sin(w * t)
        cy = 10 * math.sin(2 * w * t)
        cz = 18 + 4 * math.sin(w * t + 1)
        dx = 55 * w * math.cos(w * t)
        dy = 20 * w * math.cos(2 * w * t)
        hd = math.atan2(dy, dx) - math.pi / 2
        for ob, ox, oy, oz, ph, fr in self.pigs:
            ob.location = (cx + ox + math.sin(t * 0.7 + ph), cy + oy + math.cos(t * 0.6 + ph),
                           cz + oz + 0.4 * math.sin(t * 2 + ph))
            ob.rotation_euler = (0, 0, hd)
            ob.data = self.pig_meshes[int((t * fr + ph) * 2) % 2]

    # ------------------------------------------------------------- per frame
    def apply(self, fi):
        for a in self.agents:
            al = a.alpha[fi]
            ob = a.ob
            if al < 0.01 or a.key[fi] is None:
                ob.hide_render = True
                continue
            ob.hide_render = False
            k = a.key[fi]
            me = self.meshes.get(k) or self.meshes.get((k[0], k[1], 0)) or \
                self.meshes[(k[0], "stand", 0)]
            ob.data = me
            ob.location = tuple(a.pos[fi])
            ob.rotation_euler = (0, 0, a.head[fi])
            ob.color[3] = float(al)

    def build_all(self, city):
        self.workers()
        self.townsfolk(city.paths_walk)
        self.vehicles(city.tram_loop)
        self.pigeons()


class Ambient:
    """Flags on the Casas Consistoriales and washing on the roof terraces."""

    def __init__(self, M, city, rng):
        self.coll = new_coll("ambient")
        self.flag_meshes = []
        for k in range(8):
            mb = MB()
            nu, nv = 8, 3
            ph = 2 * math.pi * k / 8
            verts = []
            for i in range(nu + 1):
                for j in range(nv + 1):
                    u = i / nu
                    x = u * 2.6
                    y = 0.35 * u * math.sin(ph - u * 4.5) + 0.12 * u * math.sin(2 * ph - u * 9)
                    z = -1.7 * j / nv - 0.25 * u * u
                    verts.append((x, y, z))
            b = mb.add_verts(verts)
            for i in range(nu):
                for j in range(nv):
                    a0 = b + i * (nv + 1) + j
                    mb.face((a0, a0 + nv + 1, a0 + nv + 2, a0 + 1), (1, 1, 1), 0,
                            [(i / nu, 1 - j / nv), ((i + 1) / nu, 1 - j / nv),
                             ((i + 1) / nu, 1 - (j + 1) / nv), (i / nu, 1 - (j + 1) / nv)])
            me = mb.mesh("flag%d" % k)
            me.materials.append(M["flag"])
            self.flag_meshes.append(me)
        self.flags = []
        pole = MB()
        for (x, y, z), ang, out in city.flags:
            pole.beam((x, y, z - 0.3), (x + out[0] * 3.2, y + out[1] * 3.2, z + 2.6), 0.08,
                      L(60, 60, 60), 0)
            ob = obj_from_mesh("flag", self.flag_meshes[0], self.coll,
                               (x + out[0] * 3.2, y + out[1] * 3.2, z + 2.6))
            self.flags.append((ob, math.atan2(out[1], out[0]), rng.uniform(0, 6)))
        if pole.v:
            pole.build("flagpoles", [M["metal"]], self.coll)
        # washing lines
        self.wash = []
        cl = [L(240, 238, 230), L(230, 226, 210), L(190, 70, 70), L(90, 120, 170), L(220, 200, 120)]
        for a, b in city.roof_life:
            mb = MB()
            mb.beam(a, b, 0.03, L(200, 200, 200), 0)
            n = rng.randint(3, 6)
            for k in range(n):
                t = (k + 0.5) / n
                p = np.array(a) + (np.array(b) - np.array(a)) * t
                d = np.array(b) - np.array(a)
                d = d / np.linalg.norm(d) * 0.35
                i = mb.add_verts([(p[0] - d[0], p[1] - d[1], p[2]), (p[0] + d[0], p[1] + d[1], p[2]),
                                  (p[0] + d[0], p[1] + d[1], p[2] - 0.8), (p[0] - d[0], p[1] - d[1], p[2] - 0.8)])
                mb.face((i, i + 1, i + 2, i + 3), rng.choice(cl), 0)
            mid = (np.array(a) + np.array(b)) / 2
            d = np.array(b) - np.array(a)
            ang = math.atan2(d[1], d[0])
            ca, sa = math.cos(-ang), math.sin(-ang)
            mb.v = [((x - mid[0]) * ca - (y - mid[1]) * sa, (x - mid[0]) * sa + (y - mid[1]) * ca,
                     z - mid[2]) for x, y, z in mb.v]
            ob = mb.build("wash", [M["people"]], self.coll, location=tuple(mid))
            self.wash.append((ob, ang, rng.uniform(0, 6)))

    def frame(self, t):
        for ob, ang, ph in self.flags:
            ob.data = self.flag_meshes[int(t * 9 + ph * 3) % 8]
            ob.rotation_euler = (0, 0, ang + 0.25 * math.sin(t * 0.5 + ph))
        for ob, ang, ph in self.wash:
            # sway about the line axis
            s = 0.25 * math.sin(t * 1.7 + ph) + 0.1 * math.sin(t * 3.1 + ph)
            ob.rotation_euler = (s, 0, ang)
