"""Urban plan of the Plaza de Emilio Castelar (today Plaza del Ayuntamiento)
as fixed by Javier Goerlich's 1927 'Plano de proyecto de ensanche y
rectificación de la Bajada de San Francisco y plaza de Emilio Castelar'.

Local frame (metres): +x runs along the plaza from the narrow Bajada de San
Francisco end (-x) to the wide end facing Marqués de Sotelo (+x).  The
Casas Consistoriales front the south (-y) side, the Palacio de
Comunicaciones (Correos) the north-east corner.

The central island follows the 1927 sheet: an elongated platform formed by
two circular ends of unequal radius joined by tangents, raised above the
street, with oval parterres along its length and a circular patio lighting
the semi-basement flower market under its wide end."""
import math

import numpy as np
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

# Plaza frontages (building lines): piecewise, the triangle of the plaza
# opening from the Bajada de San Francisco towards Marqués de Sotelo.
S_PTS = [(-340.0, -19.0), (-75.0, -36.0), (100.0, -60.0)]
N_PTS = [(-340.0, 17.0), (-75.0, 33.0), (100.0, 62.0)]
S_A, S_B = S_PTS[0], S_PTS[-1]
N_A, N_B = N_PTS[0], N_PTS[-1]


def _pw(pts, x):
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        if x <= x1 or (x1, y1) == pts[-1]:
            return y0 + (x - x0) * (y1 - y0) / (x1 - x0)
    return pts[-1][1]


def y_s(x):
    return _pw(S_PTS, x)


def y_n(x):
    return _pw(N_PTS, x)


def _dir(pts, x):
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        if x <= x1 or (x1, y1) == pts[-1]:
            d = np.array([x1 - x0, y1 - y0])
            return d / np.linalg.norm(d)


# --- Goerlich's central island --------------------------------------------
C1, R1 = (52.0, 0.6), 26.0     # wide end (semicircle) with the market patio
C2, R2 = (-52.0, -0.4), 11.5   # narrow end
ISLAND = Point(C1).buffer(R1, 48).union(Point(C2).buffer(R2, 24)).convex_hull
ISLAND = ISLAND.simplify(0.05)
X_MIN, X_MAX = C2[0] - R2, C1[0] + R1

PATIO_C, PATIO_R = C1, 9.5
DECK_Z = 1.1          # platform top above street
SLAB_T = 0.5
PIT_Z = -4.6          # semi-basement floor


def half_width(x):
    t = (x - C2[0]) / (C1[0] - C2[0])
    return R2 + (R1 - R2) * min(max(t, 0), 1)


def axis_y(x):
    t = (x - C2[0]) / (C1[0] - C2[0])
    return C2[1] + (C1[1] - C2[1]) * t


def parterres():
    """Oval flower beds paired either side of the axis (1927 sheet)."""
    out = []
    for x in (-34.0, -14.0, 6.0, 25.0):
        w = half_width(x)
        off = w * 0.5
        a, b = 7.5 if x < 20 else 6.0, max(2.6, w * 0.24)
        for s in (-1, 1):
            out.append(((x, axis_y(x) + s * off), a, b))
    return out


def ellipse(c, a, b, n=28, rot=0.0):
    pts = []
    for k in range(n):
        t = 2 * math.pi * k / n
        x, y = a * math.cos(t), b * math.sin(t)
        pts.append((c[0] + x * math.cos(rot) - y * math.sin(rot),
                    c[1] + x * math.sin(rot) + y * math.cos(rot)))
    return pts


# Stairs from the street onto the platform: (boundary point, outward dir)
def _stair(px, side):
    """Point on the island boundary at abscissa px on side (+1 N / -1 S)."""
    ln = LineString([(px, 0), (px, 90 * side)])
    q = ISLAND.exterior.intersection(ln)
    if q.geom_type != "Point":
        q = list(q.geoms)[0]
    return (q.x, q.y), (0.0, float(side))


STAIRS = [((X_MIN, C2[1]), (-1.0, 0.0)),
          _stair(-6.0, 1), _stair(-6.0, -1),
          ((X_MAX, C1[1]), (1.0, 0.0))]


# --- City fabric -------------------------------------------------------------
EXT = 330.0

PLAZA = Polygon([(S_A[0], S_A[1]), S_PTS[1], S_B, (106.0, S_B[1]), (106.0, N_B[1]), N_B,
                 N_PTS[1], (N_A[0], N_A[1])])
PLAZA_SW = 7.5        # broad tree-lined walks along the plaza frontages


def _street(p0, p1, w):
    return LineString([p0, p1]).buffer(w / 2, cap_style=2)


def _perp_s(x, w, length=400):
    """Street leaving the south frontage, perpendicular to it."""
    d = _dir(S_PTS, x)
    n = np.array([d[1], -d[0]])     # pointing away from plaza (south)
    p = np.array([x, y_s(x)])
    return _street(tuple(p - n * 5), tuple(p + n * length), w)


def _perp_n(x, w, length=400):
    d = _dir(N_PTS, x)
    n = np.array([-d[1], d[0]])
    p = np.array([x, y_n(x)])
    return _street(tuple(p - n * 5), tuple(p + n * length), w)


def _par(pts, off, w):
    ext = list(pts[:-1]) + [(135.0, _pw(pts, 135.0))]
    ln = LineString(ext).offset_curve(off, join_style=2)
    return ln.buffer(w / 2, cap_style=2, join_style=2).intersection(box(-400, -400, 108, 400))


def _par_s(off, w):
    return _par(S_PTS, -off, w)


def _par_n(off, w):
    return _par(N_PTS, off, w)


STREETS = [
    _street((106, -EXT), (106, EXT), 13),        # cross street at the wide end
    _street((100, 0), (EXT, 0), 22),             # Marqués de Sotelo
    _street((100, 82), (EXT, 82), 11),
    _street((100, -84), (EXT, -84), 11),
    _street((180, -EXT), (180, EXT), 11),
    _perp_s(-150, 10), _perp_s(-86, 9), _perp_s(-12, 11),
    _perp_n(-160, 10), _perp_n(-98, 9), _perp_n(-34, 10), _perp_n(38, 11),
    _par_s(78, 12), _par_n(74, 12),
    _par_s(160, 11), _par_n(150, 11),
]

FABRIC = box(-EXT, -EXT, EXT, EXT).difference(unary_union([PLAZA] + STREETS))
BLOCKS = [g for g in getattr(FABRIC, "geoms", [FABRIC]) if g.area > 400]

SIDEWALK_W = 3.6


def footprint(blk):
    """Buildable area of a block: 3.6 m pavements, 7.5 m walks on the plaza."""
    f = blk.buffer(-SIDEWALK_W, join_style=2).difference(PLAZA.buffer(PLAZA_SW, join_style=2))
    if f.is_empty:
        return None
    if f.geom_type != "Polygon":
        f = max(f.geoms, key=lambda g: g.area)
    return f

# Landmark anchors
AYTO_X = (-6.5, 100.0)       # Casas Consistoriales frontage range (south)
CORREOS_X = (43.5, 100.0)    # Palacio de Comunicaciones (north, wide end)
ATENEO_X = (-29.0, 33.0)     # tall 1920s blocks with corner cupolas (north)


def block_at(pt):
    p = Point(pt)
    best = min(BLOCKS, key=lambda b: b.distance(p))
    return best
