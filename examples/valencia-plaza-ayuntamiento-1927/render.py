"""Build the scene and render the loop to multilayer EXR (+ per-frame JSON).

    python render.py --frames 0:480 --res 1920x1080 --samples 40 --out render/exr

The construction progress p(t) is a time-lapse; the first and last seconds of
the loop are drawn by post.py as ink-on-paper, so the loop closes seamlessly.
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

from vlc import agents, city, materials, plan, site, stage  # noqa: E402
from vlc.common import smooth, clamp, new_coll, MB  # noqa: E402

FPS = 24
DURATION = 20.0
N_FRAMES = int(FPS * DURATION)
T_BUILD0, T_BUILD1 = 1.0, 15.4
PLAN_ONLY = (0.30, 19.40)    # outside this window post.py shows only the drawing


def p_of_t(t):
    u = clamp((t - T_BUILD0) / (T_BUILD1 - T_BUILD0))
    return 0.15 * smooth(u) + 0.85 * u


def cam_state(t):
    w = 2 * math.pi * t / DURATION
    tx = stage.CAM_TARGET[0] + 5.0 * math.sin(w)
    ty = stage.CAM_TARGET[1] + 2.5 * (1 - math.cos(w))
    ortho = stage.ORTHO * (1 - 0.045 * (1 - math.cos(w)) / 2)
    az = stage.CAM_AZ + 1.6 * math.sin(w)
    return (tx, ty, 0.0), ortho, az


def cloud_shadows():
    mb = MB()
    mb.box((0, 0, 0), (900, 900, 0.1), (1, 1, 1), 0)
    ob = mb.build("cloud_shadows", [materials.M["cloudshadow"]])
    ob.location = (0, 0, 140)
    ob.visible_camera = False
    ob.visible_diffuse = False
    ob.visible_glossy = False
    ob.visible_transmission = False
    ob.visible_volume_scatter = False
    return ob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="0:%d" % N_FRAMES)
    ap.add_argument("--res", default="1920x1080")
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--out", default=os.path.join(HERE, "render", "exr"))
    ap.add_argument("--png", action="store_true", help="quick tone-mapped PNG previews")
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:])
    if ":" in a.frames:
        f0, f1 = (int(x) for x in a.frames.split(":"))
        step = 1
        if a.frames.count(":") == 2:
            f0, f1, step = (int(x) for x in a.frames.split(":"))
        frames = list(range(f0, f1, step))
    else:
        frames = [int(x) for x in a.frames.split(",")]
    os.makedirs(a.out, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    rng = random.Random(1927)
    res = tuple(int(v) for v in a.res.split("x"))
    stage.setup(res, a.samples)
    stage.world()
    stage.sun()
    cam = stage.camera()
    M = materials.build_all(plan.X_MIN, plan.X_MAX)
    t0 = time.time()
    C = city.City(M, rng)
    C.build()
    S = site.Site(M, rng)
    S.build()
    times = np.arange(N_FRAMES) / FPS
    S.plan_cranes(times, p_of_t, FPS)
    crowd = agents.Crowd(M, rng, times, p_of_t, FPS)
    crowd.build_all(C)
    amb = agents.Ambient(M, C, rng)
    clouds = cloud_shadows()
    print("scene built in %.1fs, %d objects" % (time.time() - t0, len(bpy.data.objects)),
          flush=True)

    sc = bpy.context.scene
    if a.png:
        sc.render.image_settings.media_type = "IMAGE"
        sc.render.image_settings.file_format = "PNG"
        sc.view_settings.view_transform = "AgX"

    dust_by_frame = {}
    for (fi, x, y, z, s) in crowd.dust:
        dust_by_frame.setdefault(fi, []).append((x, y, z, s))

    for fi in frames:
        t = fi / FPS
        p = p_of_t(t)
        ext = ".png" if a.png else ".exr"
        path = os.path.join(a.out, "f%04d%s" % (fi, ext))
        tgt, ortho, az = cam_state(t)
        stage.place_camera(cam, tgt, ortho, az=az)
        bpy.context.view_layer.update()
        meta = dict(frame=fi, t=t, p=p, res=res, ortho=ortho,
                    cam=[list(r) for r in cam.matrix_world.inverted()],
                    sun=list(stage.sun_dir()))
        dust = list(dust_by_frame.get(fi, []))
        for cr in S.cranes:
            if not cr["load"].hide_render and cr["hook"][fi][2] < plan.DECK_Z + 2.0:
                dust.append((*cr["hook"][fi], 1.0))
        if 0.42 < p < 0.68:
            fx = site.deck_front_x(p)
            for k in range(4):
                y = plan.axis_y(fx) + (k - 1.5) / 1.5 * plan.half_width(fx) * 0.8
                dust.append((fx + 1, y, plan.DECK_Z, 0.7))
        if p < 0.2:
            for x, y, r in ((-20, 2, 7), (20, -6, 8), (40, 10, 6), (-45, -2, 5)):
                dust.append((x, y, plan.PIT_Z + 1.5, 0.8 * (1 - p / 0.2)))
        meta["dust"] = dust
        render_it = PLAN_ONLY[0] - 0.05 <= t <= PLAN_ONLY[1] + 0.05
        meta["rendered"] = render_it
        with open(os.path.join(a.out, "f%04d.json" % fi), "w") as fh:
            json.dump(meta, fh)
        if not render_it or (a.skip_existing and os.path.exists(path)):
            continue
        S.update(p, fi, materials.LIVE)
        crowd.apply(fi)
        crowd.vehicle_frame(fi, t, p)
        crowd.pigeon_frame(t)
        amb.frame(t)
        C.animate(t)
        clouds.location = (-120 + 7.0 * t, -60 + 2.5 * t, 140)
        sc.render.filepath = path
        ts = time.time()
        bpy.ops.render.render(write_still=True)
        print("frame %d  p=%.3f  %.1fs" % (fi, p, time.time() - ts), flush=True)


if __name__ == "__main__":
    main()
