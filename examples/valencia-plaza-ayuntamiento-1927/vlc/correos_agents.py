"""Crews for the Palacio de Comunicaciones works, plus life on the plaza."""
import math

import numpy as np
from shapely.geometry import Point, Polygon

from . import plan
from .agents import PathLoop, sample_in
from .common import clamp, ramp
from .correos import NB, PIT_Z, band_z, t_band


def build_crews(crowd, S):
    rng = crowd.rng
    W = crowd.worker_ids
    F = S.F
    ext = F.exterior
    s0 = ext.project(Point(*S.A))

    def ext_point(s):
        q = ext.interpolate((s0 + s * ext.length) % ext.length)
        q2 = ext.interpolate((s0 + s * ext.length + 0.5) % ext.length)
        d = np.array([q2.x - q.x, q2.y - q.y])
        d /= np.linalg.norm(d) + 1e-9
        o = np.array([d[1], -d[0]])
        if F.contains(Point(q.x + o[0], q.y + o[1])):
            o = -o
        return np.array([q.x, q.y]), o

    # --- excavation: diggers and barrow men in the pit
    pit = F.buffer(-2.0)
    pz = lambda pos, p: PIT_Z
    for k in range(9):
        crowd.wanderer(rng.choice(W), lambda p: pit, 0.035 + 0.003 * k, 0.17 + 0.004 * k,
                       work_modes=("dig",), work_time=(2.5, 6), z_fn=pz)
    for k in range(6):
        crowd.wanderer(rng.choice(W), lambda p: pit, 0.04, 0.16 + 0.005 * k, walk_mode="barrow",
                       work_modes=("dig", "work"), speed=1.0, work_time=(1, 3), z_fn=pz)
    # --- masons on the scaffold following the course wave, storey by storey
    for k in range(NB):
        z0, z1 = band_z(k)
        tk = t_band(k)
        for j in range(8):
            off = 0.02 + 0.05 * j
            a = crowd.new_agent(rng.choice(W))
            for i, t in enumerate(crowd.times):
                p = crowd.pfun(t)
                s = clamp((p - tk) / 0.045) - off * 0.3 + 0.12 * j
                pos, o = ext_point(s % 1.0)
                pos = pos + o * (1.45 + 0.25 * (j % 2))
                zz = z0 + (0.05 if k > 0 else 0.0)
                a.pos[i] = (pos[0], pos[1], zz)
                a.head[i] = math.atan2(-o[1], -o[0]) - math.pi / 2
                a.key[i] = crowd._work_key(a.style, "work", t, rate=1.1, ph=j * 0.37)
                a.alpha[i] = ramp(p, tk - 0.004, tk) * (1 - ramp(p, tk + 0.085, tk + 0.09))
                if i % 6 == 0 and a.alpha[i] > 0.5:
                    crowd.dust.append((i, pos[0], pos[1], zz + 1.0, 0.35))
        # floor crew on the slab just set
        ring = S.ring.buffer(-1.5)
        for j in range(5):
            crowd.wanderer(rng.choice(W), lambda p, ring=ring: ring, tk + 0.06, tk + 0.135,
                           walk_mode=rng.choice(("walk", "hod", "barrow")), work_modes=("work", "dig"),
                           work_time=(1.5, 4), z_fn=lambda pos, p, z1=z1: z1)
    # --- hall crew, roof crew, dome crew
    hall = S.hall.buffer(-1.5)
    for k in range(6):
        crowd.wanderer(rng.choice(W), lambda p: hall, 0.38, 0.56, work_modes=("work",), z_fn=lambda pos, p: 0.0)
    zt = band_z(NB - 1)[1]
    roof = S.ring.buffer(-2.0)
    for k in range(8):
        crowd.wanderer(rng.choice(W), lambda p: roof, 0.60 + 0.004 * k, 0.80, walk_mode="hod",
                       work_modes=("work", "dig"), work_time=(2, 5), z_fn=lambda pos, p: zt)
    hz = band_z(2)[1] + 0.6
    drum = Point(*S.dome_c).buffer(13.5).difference(Point(*S.dome_c).buffer(10.6)).intersection(S.hall)
    for k in range(6):
        crowd.wanderer(rng.choice(W), lambda p: drum, 0.58, 0.78, work_modes=("work",), z_fn=lambda pos, p: hz)
    # --- hod carriers between the yard and the hoist, inside the hoarding
    loop = PathLoop(list(F.buffer(3.8, join_style=2).exterior.coords))
    for k in range(12):
        crowd.follower(rng.choice(W), loop, rng.uniform(0, loop.length),
                       rng.choice((-1, 1)) * rng.uniform(0.9, 1.2), rng.choice(("hod", "plank", "walk")),
                       p_in=0.08 + 0.01 * k, p_out=0.80 + 0.005 * k, lat=rng.uniform(-0.5, 0.5), z=0.1)
    # --- stone cutters dressing ashlar in the yard
    for k, (bx, by) in enumerate(S.bankers):
        a = crowd.new_agent(rng.choice(W))
        for i, t in enumerate(crowd.times):
            p = crowd.pfun(t)
            a.pos[i] = (bx - 0.95, by, 0.0)
            a.head[i] = -math.pi / 2
            a.key[i] = crowd._work_key(a.style, "work", t, rate=1.6, ph=k * 0.29)
            a.alpha[i] = ramp(p, 0.12, 0.13) * (1 - ramp(p, 0.79, 0.80))
            if i % 4 == 0 and a.alpha[i] > 0.5:
                crowd.dust.append((i, bx, by, 1.0, 0.5))
    yard = S.yard.buffer(-0.8)
    for k in range(4):
        crowd.wanderer(rng.choice(W), lambda p: yard, 0.06, 0.86 + 0.005 * k, walk_mode="hod",
                       work_modes=("dig", "work"), work_time=(2, 5), speed=1.0)
    # --- winch men at the derricks and the hoist
    winches = [(cr["base"][0] + 1.8, cr["base"][1] - 0.9, 0.58 if cr["k"] == 0 else 0.80)
               for cr in S.cranes]
    if S.hoist_p is not None:
        winches.append((S.hoist_p[0] + 2.0, S.hoist_p[1] - 1.5, 0.80))
    for wx, wy, tout in winches:
        for s_ in (-1, 1):
            a = crowd.new_agent(rng.choice(W))
            for i, t in enumerate(crowd.times):
                p = crowd.pfun(t)
                a.pos[i] = (wx + s_ * 1.0, wy, 0.0)
                a.head[i] = math.pi / 2 * s_
                a.key[i] = crowd._work_key(a.style, "work", t, rate=0.9, ph=0.5 * (s_ > 0))
                a.alpha[i] = ramp(p, 0.20, 0.21) * (1 - ramp(p, tout - 0.01, tout))
    # --- foremen with drawings at the front
    front = S.work_zone.difference(F.buffer(3.0)).intersection(
        Point(*S.front_mid).buffer(20))
    front = front if front.geom_type == "Polygon" else max(front.geoms, key=lambda g: g.area)
    for k in range(2):
        crowd.wanderer(crowd.foreman_id, lambda p: front, -1, 0.9, work_modes=("plan", "point"),
                       work_time=(4, 8), speed=0.9)


# eye-level camera stations: nobody stands on the lens
KEEP_OUT = [(106.8, 64.5), (106.0, 64.0), (93.5, 56.6), (78.0, 54.0), (61.9, 51.3), (64.8, 52.4)]


def build_townsfolk(crowd, S, city):
    rng = crowd.rng
    C = crowd.citizen_ids
    view = Point(*S.front_mid)
    loops = []
    for l in city.paths_walk:
        poly = Polygon(l)
        if poly.distance(view) < 120 and len(l) > 3 and not poly.intersects(S.work_zone):
            loops.append(PathLoop(l))
    for lp in loops:
        n = max(2, int(lp.length / 22))
        for k in range(n):
            st = rng.choice(C)
            mode = "basket" if (st, "basket", 0) in crowd.meshes and rng.random() < 0.5 else "walk"
            crowd.follower(st, lp, rng.uniform(0, lp.length), rng.choice((-1, 1)) * rng.uniform(1.0, 1.45),
                           mode, lat=rng.uniform(-0.9, 0.9), z=0.16, stop_prob=0.04)
    # onlookers at the hoarding on the plaza side
    ring = S.work_zone.buffer(0.7, join_style=2).exterior
    for k in range(16):
        q = ring.interpolate(rng.uniform(0, ring.length))
        if q.y > S.front_mid[1] + 8:
            continue
        if any(math.hypot(q.x - cx, q.y - cy) < 6.0 for cx, cy in KEEP_OUT):
            continue
        o = S.F.exterior.interpolate(S.F.exterior.project(q))
        st = rng.choice(C)
        a = crowd.new_agent(st)
        ph = rng.uniform(0, 6)
        p_out = rng.uniform(0.84, 0.88)
        for i, t in enumerate(crowd.times):
            p = crowd.pfun(t)
            a.pos[i] = (q.x, q.y, 0.16)
            a.head[i] = math.atan2(o.y - q.y, o.x - q.x) - math.pi / 2 + 0.3 * math.sin(t * 0.3 + ph)
            a.key[i] = (st, "stand", 0)
            a.alpha[i] = 1 - ramp(p, p_out, p_out + 0.01)
    # strollers in the island gardens
    isl = plan.ISLAND.buffer(-1.6)
    for c, aa, bb in plan.parterres():
        isl = isl.difference(Polygon(plan.ellipse(c, aa + 0.4, bb + 0.4)))
    isl = isl.difference(Point(*plan.PATIO_C).buffer(5.6))
    for k in range(40):
        crowd.wanderer(rng.choice(C), lambda p: isl, -1, 9, work_modes=("stand",), work_time=(1, 6),
                       speed=rng.uniform(0.9, 1.3), z_fn=lambda pos, p: 0.18)
    # the public arriving for the opening
    zone = S.work_zone.difference(S.F.buffer(0.5)).intersection(Point(*S.front_mid).buffer(30))
    zone = zone if zone.geom_type == "Polygon" else max(zone.geoms, key=lambda g: g.area)
    for k in range(30):
        crowd.wanderer(rng.choice(C), lambda p: zone, 0.94 + 0.04 * rng.random(), 9, work_modes=("stand",),
                       work_time=(1, 5), speed=rng.uniform(0.9, 1.3), z_fn=lambda pos, p: 0.16)
