"""Close-up set dressing for the eye-level passages of the film: the
stonecutters' yard, the front of the works with mortar troughs, buckets,
ladders and rope, the geared winches of the derricks (their drums turn with
the hoisting), a site board, and draught mules with carts and carters."""
import math
import random

import bpy
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Point

from . import props as P
from . import figures_hd as H
from .common import MB, L, jitter_col, new_coll, obj_from_mesh, ramp, smooth, new_mat, G
from .correos import NB, band_z


def extra_materials(M):
    """Painted site board with lettering (image texture)."""
    w, h = 1200, 340
    im = Image.new("RGB", (w, h), (226, 214, 186))
    d = ImageDraw.Draw(im)
    f1 = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf", 74)
    f2 = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf", 44)
    d.rectangle([12, 12, w - 12, h - 12], outline=(120, 40, 30), width=8)
    for txt, f, y, col in (("CORREOS Y TELÉGRAFOS", f1, 40, (130, 36, 28)),
                           ("OBRAS DEL PALACIO DE COMUNICACIONES", f2, 150, (50, 40, 32)),
                           ("VALENCIA", f2, 230, (50, 40, 32))):
        tw = d.textbbox((0, 0), txt, font=f)[2]
        d.text(((w - tw) / 2, y), txt, font=f, fill=col)
    path = "/tmp/vlc_signboard.png"
    im.save(path)
    img = bpy.data.images.load(path)
    m = new_mat("signboard")
    g = G(m)
    tex = g.node("ShaderNodeTexImage")
    tex.image = img
    g.link(g.uv(), tex.inputs["Vector"])
    pos = g.geo().outputs["Position"]
    n = g.noise(pos, scale=6, detail=4).outputs["Fac"]
    c = g.mix(0.25, tex.outputs["Color"], g.comb(n, n, n), "MULTIPLY")
    g.finish(g.bsdf(c, rough=0.85), alpha_fade=True)
    M["signboard"] = m


class Hero:
    def __init__(self, M, rng, S):
        self.M = M
        self.rng = rng
        self.S = S
        self.coll = new_coll("hero")
        self.items = []
        self.winches = []
        self.carts = []

    def mats(self):
        M = self.M
        return [M["wood"], M["metal"], M["ashlar"], M["people"], M["earth"], M["signboard"]]

    def static(self, mb, name, t_out=(0.86, 0.89)):
        ob = mb.build(name, self.mats(), self.coll)
        self.items.append((ob, 0.06, t_out))
        return ob

    def build(self):
        rng = self.rng
        S = self.S
        x0, y0, x1, y1 = S.yard.bounds
        # --- stonecutters' yard
        mb = MB()
        for k, (bx, by) in enumerate(S.bankers):
            P.banker(mb, (bx, by, 0.0), rng.uniform(-0.15, 0.15), rng)
            P.bucket(mb, (bx + 0.9, by + 0.5, 0.0), full=False)
        for j in range(9):
            P.ashlar_block(mb, (x1 - 1.4 + rng.uniform(-0.3, 0.3), y0 + 4 + j * 1.25, 0.0 if j % 3 else 0.6),
                           (1.2, 0.8, 0.6), rng.uniform(-0.08, 0.08), rng, rough=j % 2 == 0)
        P.tool_rack(mb, (x0 + 1.2, y0 + 2.6, 0), 0.0)
        P.sieve(mb, (x1 - 2.0, y0 + 16.5, 0), 0.4)
        mb.cone((x1 - 2.6, y0 + 18.0, 0), 1.6, 1.0, 16, L(214, 190, 140), 4)
        P.barrel(mb, (x0 + 1.0, y0 + 13.5, 0))
        P.barrel(mb, (x0 + 1.7, y0 + 14.1, 0))
        P.trestles_with_saw(mb, (x0 + 4.2, y0 + 22.0, 0), 0.1)
        P.basket(mb, (x0 + 3.2, y0 + 3.2, 0), fill=L(214, 190, 140))
        P.basket(mb, (x0 + 3.7, y0 + 3.0, 0))
        P.sign_board(mb, (x0 + 4.0, y0 - 1.6, 0), 0.0)
        self.static(mb, "yard_props", t_out=(0.90, 0.93))
        # --- front of the works: troughs, buckets, ladders, ropes, stone, brick
        fm, t, out = S.front_mid, S.t, S.out
        mb = MB()
        for u in (-17.0, -6.0, 4.0, 15.0):
            q = fm + t * u + out * 2.9
            P.mortar_trough(mb, (q[0], q[1], 0.0), S.ang + rng.uniform(-0.2, 0.2))
            for j in range(3):
                qq = q + t * (1.1 + 0.4 * j) + out * rng.uniform(-0.3, 0.4)
                P.bucket(mb, (qq[0], qq[1], 0.0), full=j != 1)
        for u in (-21.0, -12.5, -2.5, 8.5, 19.0):
            a = fm + t * u + out * 2.3
            b = fm + t * (u + 0.3) + out * 2.15
            P.ladder(mb, (a[0], a[1], 0.0), (b[0], b[1], 4.7))
        for u in (-9.0, 12.0):
            q = fm + t * u + out * 6.0
            P.rope_coil(mb, (q[0], q[1], 0.0))
            P.pulley_block(mb, (q[0] + 0.7, q[1] + 0.3, 0.42), 0.3)
        for j, u in enumerate((-14.0, -13.0, 0.5, 1.6, 6.5)):
            q = fm + t * u + out * 7.2
            P.ashlar_block(mb, (q[0], q[1], 0.0), (1.3, 0.85, 0.6), S.ang + rng.uniform(-0.1, 0.1), rng,
                           rough=j % 2 == 1)
        for u in (-4.0, 18.0):     # brick pallets
            q = fm + t * u + out * 6.4
            for j in range(4):
                mb.box((q[0] + (j % 2) * 1.1, q[1], (j // 2) * 0.5), (1.0, 0.8, 0.48), L(176, 88, 58), 3)
        P.trestles_with_saw(mb, tuple(fm + t * -19.0 + out * 6.8) + (0.0,), S.ang)
        self.static(mb, "front_props")
        # --- winches at the derricks, with rotating drums and gears
        for cr in S.cranes:
            b = cr["base"]
            p = (b[0] + 2.2, b[1] - 1.0, 0.0)
            mb = MB()
            P.winch_frame(mb, p, 0.0)
            self.static(mb, "winch_frame", t_out=(0.58 if cr["k"] == 0 else 0.80, 0.62 if cr["k"] == 0 else 0.83))
            drums = []
            for k in range(8):
                me = P.winch_drum(2 * math.pi * k / 8 / 30 * 30 / 8 * 1.0 + k * 0.1).mesh("drum%d_%d" % (cr["k"], k))
                for m in self.mats():
                    me.materials.append(m)
                drums.append(me)
            ob = obj_from_mesh("drum", drums[0], self.coll, (p[0], p[1], p[2] + 1.0))
            self.winches.append((ob, drums, cr))

    def animals(self, times, fps):
        """Hero mule cart and carter on the cross street, a tethered mule."""
        rng = random.Random(7)
        self.times = times
        self.cart_meshes = [H.mule_mesh("hmule%d" % k, L(104, 80, 60), 2 * math.pi * k / 8, self.M, cart="stone",
                                        rng=rng) for k in range(8)]
        self.cart = obj_from_mesh("hero_cart", self.cart_meshes[0], self.coll)
        self.idle = [H.mule_mesh("imule%d" % k, L(84, 78, 74), 0.0 + 0.15 * math.sin(k), self.M, rng=rng)
                     for k in range(4)]
        x0, y0, x1, y1 = self.S.yard.bounds
        self.tether = obj_from_mesh("tethered", self.idle[0], self.coll, (x0 + 1.4, y0 + 19.0, 0.0), 0.3)

    def cart_pos(self, t):
        # drives south down the cross street towards the eye-level camera
        y = 92.0 - 1.45 * (t - 14.0)
        return np.array([109.6, y]), y

    def update(self, p, fi, t):
        for ob, t_in, t_out in self.items:
            a = ramp(p, t_in, t_in + 0.02) * (1 - ramp(p, t_out[0], t_out[1]))
            ob.color[3] = float(a)
            ob.hide_render = bool(a < 0.01)
        for ob, drums, cr in self.winches:
            vis = not cr["mast"].ob.hide_render
            ob.hide_render = not vis
            ln = cr["tip"][fi][2] - cr["hook"][fi][2]
            ob.data = drums[int(ln / 0.22 / (2 * math.pi) * 8 * 3) % 8]
        if hasattr(self, "cart"):
            pos, y = self.cart_pos(t)
            self.cart.location = (pos[0], pos[1], 0.0)
            self.cart.rotation_euler = (0, 0, math.pi)
            self.cart.data = self.cart_meshes[int((92.0 - y) / 0.35) % 8]
            self.cart.hide_render = bool(not (-60 < y < 140))
            self.tether.data = self.idle[int(t * 0.8) % 4]
