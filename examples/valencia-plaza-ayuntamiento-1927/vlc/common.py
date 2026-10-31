"""Shared helpers: deterministic RNG, a vertex-coloured mesh builder and the
procedural (aged, imperfect) material library used across the scene."""
import math
import random

import bpy
import numpy as np


# ---------------------------------------------------------------------------
# Small math helpers
# ---------------------------------------------------------------------------
def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def smooth(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def smoother(x):
    x = clamp(x)
    return x * x * x * (x * (x * 6 - 15) + 10)


def ramp(t, a, b):
    """0 before a, 1 after b, smooth in between."""
    if b <= a:
        return 1.0 if t >= a else 0.0
    return smooth((t - a) / (b - a))


def settle(x):
    """Ease-out with a small damped overshoot: organic 'placing' motion."""
    x = clamp(x)
    return 1 - math.exp(-6 * x) * math.cos(7 * x) * (1 - x)


def hsv(h, s, v):
    import colorsys
    return colorsys.hsv_to_rgb(h % 1.0, clamp(s), clamp(v))


def srgb2lin(c):
    c = np.asarray(c, dtype=float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def L(*rgb):
    """sRGB 0-255 or 0-1 triplet -> linear tuple."""
    v = np.array(rgb, dtype=float)
    if v.max() > 1.0:
        v = v / 255.0
    return tuple(float(x) for x in srgb2lin(v))


def jitter_col(c, rng, amt=0.06):
    return tuple(clamp(x * (1 + rng.uniform(-amt, amt))) for x in c)


# ---------------------------------------------------------------------------
# Mesh builder (per-face colour + per-face material + UV per loop)
# ---------------------------------------------------------------------------
class MB:
    def __init__(self):
        self.v = []
        self.f = []
        self.fc = []      # (r,g,b,a) per face
        self.fm = []      # material slot per face
        self.uv = []      # list of per-loop uv tuples per face

    # -- primitives ---------------------------------------------------------
    def add_verts(self, pts):
        b = len(self.v)
        self.v.extend([tuple(map(float, p)) for p in pts])
        return b

    def face(self, idx, col=(1, 1, 1, 1), mat=0, uv=None):
        if len(col) == 3:
            col = (*col, 1.0)
        self.f.append(tuple(idx))
        self.fc.append(col)
        self.fm.append(mat)
        if uv is None:
            uv = [(self.v[i][0], self.v[i][1]) for i in idx]
        self.uv.append(uv)

    def quad_wall(self, a, b, z0, z1, col, mat=0, u0=0.0):
        """Vertical quad from 2D a->b (outward normal to the right of a->b
        when the polygon is CCW).  UV = (distance along wall, absolute z)."""
        i = self.add_verts([(a[0], a[1], z0), (b[0], b[1], z0),
                            (b[0], b[1], z1), (a[0], a[1], z1)])
        d = math.dist(a, b)
        self.face((i, i + 1, i + 2, i + 3), col, mat,
                  [(u0, z0), (u0 + d, z0), (u0 + d, z1), (u0, z1)])
        return d

    def prism(self, poly, z0, z1, col, mat=0, top=True, bottom=False,
              wall_mat=None, top_mat=None, top_col=None, u0=0.0):
        """Extrude a CCW 2D polygon between z0 and z1."""
        poly = [tuple(p) for p in poly]
        if signed_area(poly) < 0:
            poly = poly[::-1]
        n = len(poly)
        u = u0
        wm = mat if wall_mat is None else wall_mat
        for k in range(n):
            u += self.quad_wall(poly[k], poly[(k + 1) % n], z0, z1, col, wm, u)
        if top:
            i = self.add_verts([(p[0], p[1], z1) for p in poly])
            self.face(range(i, i + n), top_col or col,
                      mat if top_mat is None else top_mat)
        if bottom:
            i = self.add_verts([(p[0], p[1], z0) for p in poly[::-1]])
            self.face(range(i, i + n), col, mat)

    def box(self, c, s, col, mat=0, rot=0.0, top=True, bottom=False):
        """Axis box; c = (x, y, zbottom), s = (sx, sy, sz), rotated about z."""
        hx, hy = s[0] / 2, s[1] / 2
        ca, sa = math.cos(rot), math.sin(rot)
        pts = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
        poly = [(c[0] + x * ca - y * sa, c[1] + x * sa + y * ca) for x, y in pts]
        self.prism(poly, c[2], c[2] + s[2], col, mat, top=top, bottom=bottom)

    def cyl(self, c, r, h, n, col, mat=0, r2=None, top=True, z_top_col=None):
        r2 = r if r2 is None else r2
        b = self.add_verts([(c[0] + r * math.cos(2 * math.pi * k / n),
                             c[1] + r * math.sin(2 * math.pi * k / n), c[2])
                            for k in range(n)])
        t = self.add_verts([(c[0] + r2 * math.cos(2 * math.pi * k / n),
                             c[1] + r2 * math.sin(2 * math.pi * k / n), c[2] + h)
                            for k in range(n)])
        per = 2 * math.pi * max(r, 1e-3)
        for k in range(n):
            k2 = (k + 1) % n
            u0, u1 = per * k / n, per * (k + 1) / n
            self.face((b + k, b + k2, t + k2, t + k), col, mat,
                      [(u0, c[2]), (u1, c[2]), (u1, c[2] + h), (u0, c[2] + h)])
        if top and r2 > 1e-4:
            self.face(range(t, t + n), z_top_col or col, mat)

    def cone(self, c, r, h, n, col, mat=0):
        b = self.add_verts([(c[0] + r * math.cos(2 * math.pi * k / n),
                             c[1] + r * math.sin(2 * math.pi * k / n), c[2])
                            for k in range(n)])
        t = self.add_verts([(c[0], c[1], c[2] + h)])
        for k in range(n):
            self.face((b + k, b + (k + 1) % n, t), col, mat)

    def dome(self, c, r, n=12, rings=5, col=(1, 1, 1), mat=0, squash=1.0,
             open_bottom=True, rgt=None):
        """Hemisphere (or squashed) on z = c[2]."""
        rows = []
        for j in range(rings + 1):
            phi = (math.pi / 2) * j / rings
            rr = r * math.cos(phi)
            z = c[2] + r * squash * math.sin(phi)
            if j == rings:
                rows.append([self.add_verts([(c[0], c[1], z)])])
                continue
            i = self.add_verts([(c[0] + rr * math.cos(2 * math.pi * k / n),
                                 c[1] + rr * math.sin(2 * math.pi * k / n), z)
                                for k in range(n)])
            rows.append(list(range(i, i + n)))
        for j in range(rings):
            a, b = rows[j], rows[j + 1]
            for k in range(n):
                k2 = (k + 1) % n
                if len(b) == 1:
                    self.face((a[k], a[k2], b[0]), col, mat)
                else:
                    self.face((a[k], a[k2], b[k2], b[k]), col, mat)

    def sphere(self, c, r, n=10, rings=6, col=(1, 1, 1), mat=0, sz=1.0,
               noise=0.0, rng=None):
        rows = []
        for j in range(rings + 1):
            th = math.pi * j / rings
            if j in (0, rings):
                z = c[2] - r * sz * math.cos(th)
                rows.append([self.add_verts([(c[0], c[1], z)])])
                continue
            pts = []
            for k in range(n):
                ph = 2 * math.pi * k / n
                rr = r * (1 + (rng.uniform(-noise, noise) if rng else 0))
                pts.append((c[0] + rr * math.sin(th) * math.cos(ph),
                            c[1] + rr * math.sin(th) * math.sin(ph),
                            c[2] - rr * sz * math.cos(th)))
            i = self.add_verts(pts)
            rows.append(list(range(i, i + n)))
        for j in range(rings):
            a, b = rows[j], rows[j + 1]
            for k in range(n):
                k2 = (k + 1) % n
                if len(a) == 1:
                    self.face((a[0], b[k2], b[k]), col, mat)
                elif len(b) == 1:
                    self.face((a[k], a[k2], b[0]), col, mat)
                else:
                    self.face((a[k], a[k2], b[k2], b[k]), col, mat)

    def beam(self, p0, p1, w, col, mat=0, h=None):
        """Square-section member between two 3D points (poles, ropes, struts)."""
        h = w if h is None else h
        p0 = np.array(p0, float)
        p1 = np.array(p1, float)
        d = p1 - p0
        ln = np.linalg.norm(d)
        if ln < 1e-6:
            return
        d /= ln
        up = np.array([0, 0, 1.0]) if abs(d[2]) < 0.95 else np.array([1.0, 0, 0])
        a = np.cross(d, up)
        a /= np.linalg.norm(a)
        b = np.cross(a, d)
        a *= w / 2
        b *= h / 2
        corners = [a + b, -a + b, -a - b, a - b]
        i = self.add_verts([p0 + c for c in corners] + [p1 + c for c in corners])
        for k in range(4):
            k2 = (k + 1) % 4
            self.face((i + k, i + k2, i + 4 + k2, i + 4 + k), col, mat,
                      [(0, 0), (w, 0), (w, ln), (0, ln)])
        self.face((i + 4, i + 5, i + 6, i + 7), col, mat)

    def merge(self, other, offset=(0, 0, 0), rot=0.0, scale=1.0):
        ca, sa = math.cos(rot), math.sin(rot)
        pts = []
        for x, y, z in other.v:
            x, y, z = x * scale, y * scale, z * scale
            pts.append((offset[0] + x * ca - y * sa,
                        offset[1] + x * sa + y * ca, offset[2] + z))
        b = self.add_verts(pts)
        for f, c, m, uv in zip(other.f, other.fc, other.fm, other.uv):
            self.f.append(tuple(i + b for i in f))
            self.fc.append(c)
            self.fm.append(m)
            self.uv.append(uv)

    # -- output -------------------------------------------------------------
    def mesh(self, name):
        me = bpy.data.meshes.new(name)
        me.from_pydata(self.v, [], self.f)
        nl = len(me.loops)
        col = np.zeros((nl, 4), np.float32)
        uvs = np.zeros((nl, 2), np.float32)
        li = 0
        for f, c, uv in zip(self.f, self.fc, self.uv):
            for k in range(len(f)):
                col[li] = c
                uvs[li] = uv[k]
                li += 1
        ca = me.color_attributes.new("Col", "FLOAT_COLOR", "CORNER")
        ca.data.foreach_set("color", col.ravel())
        uvl = me.uv_layers.new(name="UVMap")
        uvl.data.foreach_set("uv", uvs.ravel())
        me.polygons.foreach_set("material_index", np.array(self.fm, np.int32))
        me.update()
        return me

    def build(self, name, mats, coll=None, location=(0, 0, 0)):
        me = self.mesh(name)
        for m in mats:
            me.materials.append(m)
        ob = bpy.data.objects.new(name, me)
        ob.location = location
        (coll or bpy.context.scene.collection).objects.link(ob)
        return ob


def signed_area(poly):
    a = 0.0
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i][:2]
        x1, y1 = poly[(i + 1) % n][:2]
        a += x0 * y1 - x1 * y0
    return a / 2


def obj_from_mesh(name, me, coll=None, loc=(0, 0, 0), rot=0.0, scale=1.0):
    ob = bpy.data.objects.new(name, me)
    ob.location = loc
    ob.rotation_euler = (0, 0, rot)
    ob.scale = (scale, scale, scale)
    (coll or bpy.context.scene.collection).objects.link(ob)
    return ob


def new_coll(name):
    c = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(c)
    return c


# ---------------------------------------------------------------------------
# Shader-graph helper
# ---------------------------------------------------------------------------
class G:
    """Tiny DSL around a material node tree."""

    def __init__(self, mat):
        self.nt = mat.node_tree
        self.N = self.nt.nodes
        self.Lk = self.nt.links
        for n in list(self.N):
            self.N.remove(n)
        self.out = self.N.new("ShaderNodeOutputMaterial")

    def node(self, t, **props):
        n = self.N.new(t)
        for k, v in props.items():
            setattr(n, k, v)
        return n

    def link(self, a, b):
        self.Lk.new(a, b)

    def val(self, x):
        n = self.node("ShaderNodeValue")
        n.outputs[0].default_value = x
        return n.outputs[0]

    def _in(self, sock, v):
        if v is None:
            return
        if isinstance(v, bpy.types.NodeSocket):
            self.link(v, sock)
        else:
            try:
                sock.default_value = v
            except (ValueError, TypeError):
                sock.default_value = (*v, 1.0)

    def math(self, op, a, b=None, c=None, clamp_=False):
        n = self.node("ShaderNodeMath", operation=op, use_clamp=clamp_)
        self._in(n.inputs[0], a)
        if b is not None:
            self._in(n.inputs[1], b)
        if c is not None:
            self._in(n.inputs[2], c)
        return n.outputs[0]

    def vmath(self, op, a, b=None, scale=None):
        n = self.node("ShaderNodeVectorMath", operation=op)
        self._in(n.inputs[0], a)
        if b is not None:
            self._in(n.inputs[1], b)
        if scale is not None:
            self._in(n.inputs[3], scale)
        out = n.outputs[1] if op in ("DOT_PRODUCT", "LENGTH", "DISTANCE") else n.outputs[0]
        return out

    def mix(self, fac, a, b, blend="MIX"):
        n = self.node("ShaderNodeMix", data_type="RGBA", blend_type=blend)
        self._in(n.inputs[0], fac)
        self._in(n.inputs[6], a if not isinstance(a, tuple) or len(a) == 4 else (*a, 1))
        self._in(n.inputs[7], b if not isinstance(b, tuple) or len(b) == 4 else (*b, 1))
        return n.outputs[2]

    def mixf(self, fac, a, b):
        n = self.node("ShaderNodeMix", data_type="FLOAT")
        self._in(n.inputs[0], fac)
        self._in(n.inputs[2], a)
        self._in(n.inputs[3], b)
        return n.outputs[0]

    def noise(self, vec=None, scale=5.0, detail=4.0, rough=0.55, dist=0.0, dim="3D"):
        n = self.node("ShaderNodeTexNoise", noise_dimensions=dim)
        self._in(n.inputs["Vector"], vec)
        n.inputs["Scale"].default_value = scale
        n.inputs["Detail"].default_value = detail
        n.inputs["Roughness"].default_value = rough
        n.inputs["Distortion"].default_value = dist
        return n

    def white(self, vec, dim="3D"):
        n = self.node("ShaderNodeTexWhiteNoise", noise_dimensions=dim)
        self._in(n.inputs["Vector"], vec)
        return n

    def ramp(self, fac, stops):
        n = self.node("ShaderNodeValToRGB")
        self._in(n.inputs[0], fac)
        cr = n.color_ramp
        while len(cr.elements) < len(stops):
            cr.elements.new(0.5)
        for e, (pos, col) in zip(cr.elements, stops):
            e.position = pos
            e.color = col if len(col) == 4 else (*col, 1)
        return n.outputs[0]

    def sep(self, v):
        n = self.node("ShaderNodeSeparateXYZ")
        self._in(n.inputs[0], v)
        return n.outputs

    def comb(self, x=0.0, y=0.0, z=0.0):
        n = self.node("ShaderNodeCombineXYZ")
        self._in(n.inputs[0], x)
        self._in(n.inputs[1], y)
        self._in(n.inputs[2], z)
        return n.outputs[0]

    def attr(self, name="Col"):
        n = self.node("ShaderNodeAttribute", attribute_name=name)
        return n

    def geo(self):
        return self.node("ShaderNodeNewGeometry")

    def texco(self):
        return self.node("ShaderNodeTexCoord")

    def uv(self):
        return self.node("ShaderNodeUVMap", uv_map="UVMap").outputs[0]

    def bump(self, height, strength=0.3, distance=0.05, normal=None):
        n = self.node("ShaderNodeBump")
        n.inputs["Strength"].default_value = strength
        n.inputs["Distance"].default_value = distance
        self._in(n.inputs["Height"], height)
        if normal is not None:
            self._in(n.inputs["Normal"], normal)
        return n.outputs[0]

    def bsdf(self, color, rough=0.8, normal=None, spec=0.3, metallic=0.0,
             sss=None, sheen=0.0, emission=None, estr=0.0, transmission=0.0,
             ior=1.45):
        b = self.node("ShaderNodeBsdfPrincipled")
        self._in(b.inputs["Base Color"], color)
        self._in(b.inputs["Roughness"], rough)
        b.inputs["Metallic"].default_value = metallic
        b.inputs["Specular IOR Level"].default_value = spec
        b.inputs["IOR"].default_value = ior
        if transmission:
            b.inputs["Transmission Weight"].default_value = transmission
        if sheen:
            b.inputs["Sheen Weight"].default_value = sheen
        if sss is not None:
            b.inputs["Subsurface Weight"].default_value = sss
            b.inputs["Subsurface Radius"].default_value = (0.3, 0.25, 0.1)
        if normal is not None:
            self._in(b.inputs["Normal"], normal)
        if emission is not None:
            self._in(b.inputs["Emission Color"], emission)
            self._in(b.inputs["Emission Strength"], estr)
        return b.outputs[0]

    def finish(self, shader, alpha_fade=False):
        """Connect to output.  alpha_fade lets objects dissolve through their
        object colour alpha (used for scaffolds, cranes, workers...)."""
        if alpha_fade:
            oi = self.node("ShaderNodeObjectInfo")
            tr = self.node("ShaderNodeBsdfTransparent")
            mx = self.node("ShaderNodeMixShader")
            self.link(oi.outputs["Alpha"], mx.inputs[0])
            self.link(tr.outputs[0], mx.inputs[1])
            self.link(shader, mx.inputs[2])
            shader = mx.outputs[0]
        self.link(shader, self.out.inputs["Surface"])


def new_mat(name):
    m = bpy.data.materials.new(name)
    try:
        m.use_nodes = True
    except Exception:
        pass
    return m
