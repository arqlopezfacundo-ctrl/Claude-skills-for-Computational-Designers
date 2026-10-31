"""40-second single-shot film of the Palacio de Comunicaciones rising.

One continuous perspective camera move (no cuts): overhead plan view taken
over from the unfolding map -> bird's-eye orbit -> descent to eye level in
the stonecutters' yard -> tilt up the scaffolded corner -> lateral dolly
along the front past the derrick winch -> low-angle look up at the tower
and dome -> rise back to the plan view, where the map folds again.
Focus racks between near and far subjects throughout.

    python film.py --frames 60:110 --res 1080x1200 --samples 5 --out film/exr
"""
import argparse
import json
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bpy  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Vector  # noqa: E402

from vlc import agents, city, correos, correos_agents, materials, plan, stage  # noqa: E402
from vlc import figures_hd, hero  # noqa: E402
from vlc.common import smooth, clamp, MB  # noqa: E402

FPS = 24
DURATION = 40.0
N_FRAMES = int(FPS * DURATION)
T_BUILD0, T_BUILD1 = 4.0, 34.5
RENDER_WINDOW = (2.4, 38.2)      # outside: the map film covers the frame

# (t, camera position, look-at, lens mm, focus distance or None, f-stop)
KEYS = [
    (2.4, (71.0, 72.0, 235.0), (71.0, 92.0, 0.0), 32, None, 16),
    (3.0, (71.0, 72.0, 230.0), (71.0, 92.0, 0.0), 32, None, 16),
    (7.5, (12.0, 2.0, 150.0), (72.0, 92.0, 0.0), 35, None, 16),
    (12.0, (132.0, 14.0, 92.0), (76.0, 90.0, 6.0), 35, None, 11),
    (16.0, (108.0, 46.0, 24.0), (95.0, 80.0, 4.0), 30, None, 8),
    (19.5, (106.8, 64.5, 1.75), (104.0, 78.0, 1.0), 28, 9.0, 2.8),
    (23.0, (106.0, 64.0, 1.7), (93.0, 73.0, 19.0), 24, 20.0, 4.0),
    (26.0, (93.5, 56.6, 1.75), (86.0, 70.0, 4.0), 28, 9.0, 2.8),
    (29.5, (61.9, 51.3, 1.75), (60.0, 68.0, 5.0), 28, 9.0, 2.8),
    (32.0, (64.8, 52.4, 1.6), (71.0, 65.0, 31.0), 22, 26.0, 4.0),
    (35.0, (28.0, 14.0, 72.0), (72.0, 88.0, 14.0), 32, None, 11),
    (37.6, (71.0, 72.0, 230.0), (71.0, 92.0, 0.0), 32, None, 16),
    (38.2, (71.0, 72.0, 235.0), (71.0, 92.0, 0.0), 32, None, 16),
]


def p_of_t(t):
    u = clamp((t - T_BUILD0) / (T_BUILD1 - T_BUILD0))
    return 0.12 * smooth(u) + 0.88 * u


def _cr(p0, p1, p2, p3, t0, t1, t2, t3, t):
    """Non-uniform Catmull-Rom (Barry-Goldman) evaluated at time t in [t1, t2]."""
    def lerp(a, b, ta, tb):
        if tb - ta < 1e-9:
            return a
        return a + (b - a) * ((t - ta) / (tb - ta))
    a1 = lerp(p0, p1, t0, t1)
    a2 = lerp(p1, p2, t1, t2)
    a3 = lerp(p2, p3, t2, t3)
    b1 = lerp(a1, a2, t0, t2)
    b2 = lerp(a2, a3, t1, t3)
    return lerp(b1, b2, t1, t2)


def cam_at(t):
    ts = [k[0] for k in KEYS]
    t = min(max(t, ts[0]), ts[-1])
    i = max(0, min(len(KEYS) - 2, int(np.searchsorted(ts, t, side="right")) - 1))
    i0, i1, i2, i3 = max(i - 1, 0), i, i + 1, min(i + 2, len(KEYS) - 1)
    T = [KEYS[j][0] for j in (i0, i1, i2, i3)]
    if T[0] == T[1]:
        T[0] -= 1
    if T[3] == T[2]:
        T[3] += 1
    # ease within each segment so the camera never jerks
    u = (t - T[1]) / max(T[2] - T[1], 1e-9)
    te = T[1] + (T[2] - T[1]) * (0.5 - 0.5 * math.cos(math.pi * u))
    te = 0.35 * t + 0.65 * te
    pos = _cr(*[np.array(KEYS[j][1], float) for j in (i0, i1, i2, i3)], *T, te)
    tgt = _cr(*[np.array(KEYS[j][2], float) for j in (i0, i1, i2, i3)], *T, te)
    k = (te - T[1]) / max(T[2] - T[1], 1e-9)
    k = smooth(k)
    lens = KEYS[i1][3] + (KEYS[i2][3] - KEYS[i1][3]) * k
    f1 = KEYS[i1][4] if KEYS[i1][4] is not None else float(np.linalg.norm(np.array(KEYS[i1][2]) - np.array(KEYS[i1][1])))
    f2 = KEYS[i2][4] if KEYS[i2][4] is not None else float(np.linalg.norm(np.array(KEYS[i2][2]) - np.array(KEYS[i2][1])))
    focus = math.exp(math.log(f1) + (math.log(f2) - math.log(f1)) * k)
    fstop = math.exp(math.log(KEYS[i1][5]) + (math.log(KEYS[i2][5]) - math.log(KEYS[i1][5])) * k)
    # floor the eye a little above the walk and add a breathing hand-held drift
    if pos[2] < 4.0:
        pos = pos + np.array([0.04 * math.sin(t * 0.9), 0.03 * math.sin(t * 1.3 + 1), 0.025 * math.sin(t * 1.7 + 2)])
    return pos, tgt, lens, focus, fstop


def setup_camera():
    cd = bpy.data.cameras.new("film")
    cd.type = "PERSP"
    cd.sensor_fit = "HORIZONTAL"
    cd.sensor_width = 36.0
    cd.clip_start = 0.1
    cd.clip_end = 3000
    cd.dof.use_dof = True
    ob = bpy.data.objects.new("film", cd)
    bpy.context.scene.collection.objects.link(ob)
    bpy.context.scene.camera = ob
    return ob


def place(cam, t):
    pos, tgt, lens, focus, fstop = cam_at(t)
    cam.location = Vector(pos)
    d = Vector(tgt) - Vector(pos)
    q = d.to_track_quat("-Z", "Y")
    cam.rotation_mode = "XYZ"
    cam.rotation_euler = q.to_euler()
    cam.data.lens = lens
    cam.data.dof.focus_distance = focus
    cam.data.dof.aperture_fstop = fstop
    return pos, tgt, lens, focus, fstop


def sky_world():
    """Camera sees a clean gradient sky (replaced by the painted sky in post);
    lighting uses a warm-horizon / blue-zenith dome."""
    w = bpy.context.scene.world
    nt = w.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    nt.links.new(tc.outputs["Generated"], sep.inputs[0])
    nt.links.new(sep.outputs[2], ramp.inputs[0])
    cr = ramp.color_ramp
    cr.elements[0].position = 0.48
    cr.elements[0].color = (0.72, 0.62, 0.50, 1)
    cr.elements[1].position = 0.80
    cr.elements[1].color = (0.30, 0.46, 0.80, 1)
    nt.links.new(ramp.outputs[0], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 1.05
    nt.links.new(bg.outputs[0], out.inputs["Surface"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="0:%d" % N_FRAMES)
    ap.add_argument("--res", default="1080x1200")
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(HERE, "film", "exr"))
    ap.add_argument("--png", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args()
    if ":" in a.frames:
        parts = [int(x) for x in a.frames.split(":")]
        frames = list(range(parts[0], parts[1], parts[2] if len(parts) > 2 else 1))
    else:
        frames = [int(x) for x in a.frames.split(",")]
    os.makedirs(a.out, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    rng = random.Random(1915)
    res = tuple(int(v) for v in a.res.split("x"))
    stage.SUN_AZ = 62.0
    stage.SUN_ELEV = 24.0
    stage.setup(res, a.samples)
    stage.world()
    sky_world()
    stage.sun()
    cam = setup_camera()
    M = materials.build_all(plan.X_MIN, plan.X_MAX)
    hero.extra_materials(M)
    t0 = time.time()
    times = np.arange(N_FRAMES) / FPS
    S = correos.Correos(M, rng)
    C = city.City(M, rng)
    C.bounds_r = 230.0
    C.ground_hole = S.F
    C.skip_pt = (70.0, 90.0)
    C.no_correos = True
    C.exclude_zone = S.work_zone.buffer(1.5)
    C.build()
    S.build()
    H = hero.Hero(M, rng, S)
    H.build()
    S.plan_cranes(times, p_of_t, FPS)
    crowd = agents.Crowd(M, rng, times, p_of_t, FPS)
    # swap the crowd's pose library for the articulated close-up figures
    styles, nW, lib = figures_hd.build_hd_library(M)
    crowd.styles = styles
    crowd.meshes = lib
    crowd.worker_ids = [i for i, s_ in enumerate(styles) if s_.kind == "worker"]
    crowd.citizen_ids = [i for i, s_ in enumerate(styles) if s_.kind in ("man", "woman", "child")]
    crowd.foreman_id = [i for i, s_ in enumerate(styles) if s_.kind == "foreman"][0]
    correos_agents.build_crews(crowd, S)
    correos_agents.build_townsfolk(crowd, S, C)
    crowd.vehicles(C.tram_loop)
    crowd.pigeons()
    H.animals(times, FPS)
    carter = crowd.new_agent(crowd.worker_ids[1])
    for i, tt in enumerate(times):
        cp, y = H.cart_pos(tt)
        carter.pos[i] = (cp[0] - 1.05, cp[1] - 1.2, 0.0)
        carter.head[i] = math.pi
        carter.key[i] = crowd._walk_key(carter.style, "walk", 92.0 - y)
    amb = agents.Ambient(M, C, rng)
    print("scene built in %.1fs, %d objects" % (time.time() - t0, len(bpy.data.objects)), flush=True)

    sc = bpy.context.scene
    if a.png:
        sc.render.image_settings.media_type = "IMAGE"
        sc.render.image_settings.file_format = "PNG"
        sc.view_settings.view_transform = "AgX"
    dust_by_frame = {}
    for (fi, x, y, z, s_) in crowd.dust:
        dust_by_frame.setdefault(fi, []).append((x, y, z, s_))

    for fi in frames:
        t = fi / FPS
        p = p_of_t(t)
        ext = ".png" if a.png else ".exr"
        path = os.path.join(a.out, "f%04d%s" % (fi, ext))
        pos, tgt, lens, focus, fstop = place(cam, t)
        bpy.context.view_layer.update()
        meta = dict(frame=fi, t=t, p=p, res=res, subject="correos", persp=True, lens=lens,
                    cam_world=[list(r) for r in cam.matrix_world], focus=focus,
                    cam=[list(r) for r in cam.matrix_world.inverted()], sun=list(stage.sun_dir()))
        dust = list(dust_by_frame.get(fi, []))
        for cr in S.cranes:
            h = cr["hook"][fi]
            if cr["att"][fi] and abs(h[2] - S.work_level(p) - 1.2) < 1.5:
                dust.append((*h, 0.9))
        meta["dust"] = dust
        render_it = RENDER_WINDOW[0] <= t <= RENDER_WINDOW[1]
        meta["rendered"] = render_it
        with open(os.path.join(a.out, "f%04d.json" % fi), "w") as fh:
            json.dump(meta, fh)
        if not render_it or (a.skip_existing and os.path.exists(path)):
            continue
        S.update(p, fi, materials.LIVE)
        H.update(p, fi, t)
        crowd.apply(fi)
        crowd.vehicle_frame(fi, t, p)
        crowd.pigeon_frame(t)
        amb.frame(t)
        C.animate(t)
        sc.render.filepath = path
        ts = time.time()
        bpy.ops.render.render(write_still=True)
        print("frame %d  t=%.2f p=%.3f  %.1fs" % (fi, t, p, time.time() - ts), flush=True)


if __name__ == "__main__":
    main()
