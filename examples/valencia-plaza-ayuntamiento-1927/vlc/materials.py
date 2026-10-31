"""Procedural material library.  Every surface carries some imperfection:
rain streaks, ground dirt, stone/brick colour scatter, sun-faded shutters,
weathered timber.  Base colours come from the per-face 'Col' attribute so a
single material serves hundreds of differently tinted elements."""
from .common import G, new_mat, L

M = {}            # name -> material
LIVE = {}         # name -> value socket animated per frame


# -- graph sugar -------------------------------------------------------------
def lt(g, a, b):
    return g.math("LESS_THAN", a, b)


def gt(g, a, b):
    return g.math("GREATER_THAN", a, b)


def mul(g, *xs):
    o = xs[0]
    for x in xs[1:]:
        o = g.math("MULTIPLY", o, x)
    return o


def add(g, a, b):
    return g.math("ADD", a, b)


def sub(g, a, b):
    return g.math("SUBTRACT", a, b)


def band(g, x, lo, hi):
    return mul(g, gt(g, x, lo), lt(g, x, hi))


def sstep(g, lo, hi, x):
    n = g.node("ShaderNodeMapRange", interpolation_type="SMOOTHSTEP")
    g._in(n.inputs["Value"], x)
    n.inputs["From Min"].default_value = lo
    n.inputs["From Max"].default_value = hi
    return n.outputs[0]


def patina(g, base, pos, streak=0.28, dirt=0.25, var=0.18, scale=0.6):
    """Colour scatter + vertical rain streaks + ground dirt."""
    sx = g.sep(pos)
    nv = g.noise(pos, scale=scale, detail=6, rough=0.6).outputs["Fac"]
    fine = g.noise(pos, scale=9.0, detail=3, rough=0.7).outputs["Fac"]
    # vertical streaks: squash noise in z
    sv = g.comb(mul(g, sx[0], 2.2), mul(g, sx[1], 2.2), mul(g, sx[2], 0.12))
    st = g.noise(sv, scale=1.0, detail=4, rough=0.6).outputs["Fac"]
    st = sstep(g, 0.45, 0.85, st)
    k = add(g, 1.0 - var * 0.5, mul(g, nv, var))
    k = add(g, k, mul(g, sub(g, fine, 0.5), 0.12))
    k = mul(g, k, sub(g, 1.0, mul(g, st, streak)))
    gd = sstep(g, 0.0, 2.2, sx[2])
    k = mul(g, k, add(g, 1.0 - dirt, mul(g, gd, dirt)))
    return g.mix(1.0, base, g.comb(k, k, k), "MULTIPLY"), nv


# ---------------------------------------------------------------------------
def m_facade():
    m = new_mat("facade")
    g = G(m)
    uv = g.sep(g.uv())
    u, z = uv[0], uv[1]
    ca = g.attr("Col")
    base, seed = ca.outputs["Color"], ca.outputs["Alpha"]
    pos = g.geo().outputs["Position"]
    Wb, Hf, G0 = 3.6, 3.3, 4.6
    uu = g.math("DIVIDE", u, Wb)
    zz = g.math("DIVIDE", sub(g, z, G0), Hf)
    fu = g.math("FRACT", uu)
    fz = g.math("FRACT", zz)
    iu = g.math("FLOOR", uu)
    iz = g.math("FLOOR", zz)
    upper = gt(g, z, G0)
    wr = g.white(g.comb(iu, iz, mul(g, seed, 173.0)))
    r = wr.outputs["Value"]
    r2 = g.white(g.comb(iz, iu, mul(g, seed, 91.0))).outputs["Value"]
    dx = g.math("ABSOLUTE", sub(g, fu, 0.5))
    win = mul(g, lt(g, dx, 0.18), band(g, fz, 0.16, 0.80), upper)
    # roll-down 'persianas': each window lowered to a random level
    shut = mul(g, win, gt(g, fz, sub(g, 0.80, mul(g, r, 0.66))))
    balc = mul(g, lt(g, dx, 0.27), band(g, fz, 0.105, 0.16), upper)
    balc_sh = mul(g, lt(g, dx, 0.25), band(g, fz, 0.045, 0.105), upper)
    rail = mul(g, win, lt(g, fz, 0.36),
               lt(g, g.math("FRACT", mul(g, u, 9.0)), 0.35))
    cornice = mul(g, band(g, fz, 0.955, 1.0), upper)
    # ground floor shops and striped awnings
    gf = lt(g, z, G0)
    shop = mul(g, gf, lt(g, dx, 0.36), band(g, z, 0.12, 3.35))
    awn_on = lt(g, g.white(g.comb(iu, 7.0, mul(g, seed, 31.0))).outputs["Value"], 0.45)
    awn = mul(g, gf, lt(g, dx, 0.42), band(g, z, 3.35, 3.95), awn_on)
    stripes = lt(g, g.math("FRACT", mul(g, u, 2.2)), 0.5)
    awn_col = g.ramp(g.white(g.comb(iu, 3.0, seed)).outputs["Value"],
                     [(0.0, L(150, 40, 35)), (0.35, L(40, 80, 60)),
                      (0.7, L(170, 120, 50)), (1.0, L(60, 70, 110))])
    awn_c = g.mix(stripes, awn_col, L(225, 215, 190))

    wall, nv = patina(g, base, pos)
    wall = g.mix(mul(g, cornice, 0.5), wall, (1.0, 0.97, 0.9, 1.0), "SCREEN")
    wall = g.mix(mul(g, balc_sh, 0.65), wall, (0.05, 0.04, 0.035, 1), "MULTIPLY")
    glass = g.mix(r2, L(16, 18, 22), L(48, 40, 32))
    shut_c = g.ramp(r2, [(0.0, L(62, 92, 64)), (0.3, L(95, 112, 70)),
                         (0.55, L(120, 88, 52)), (0.8, L(150, 140, 110)),
                         (1.0, L(70, 100, 98))])
    shut_c = g.mix(mul(g, g.noise(pos, scale=4, detail=2).outputs["Fac"], 0.5),
                   shut_c, L(170, 165, 140))
    c = g.mix(win, wall, glass)
    c = g.mix(shut, c, shut_c)
    c = g.mix(mul(g, rail, 0.8), c, L(20, 20, 18))
    c = g.mix(balc, c, g.mix(0.4, base, (0.9, 0.88, 0.82, 1)))
    c = g.mix(shop, c, L(26, 22, 18))
    c = g.mix(awn, c, awn_c)
    rough = g.mixf(win, 0.85, 0.25)
    h = sub(g, 1.0, add(g, mul(g, win, 0.6), mul(g, shop, 0.5)))
    h = add(g, h, add(g, mul(g, balc, 0.35), mul(g, nv, 0.08)))
    nrm = g.bump(h, strength=0.55, distance=0.12)
    g.finish(g.bsdf(c, rough=rough, normal=nrm, spec=0.35))
    M["facade"] = m


def m_plain(name="plain", fade=False, streak=0.22, dirt=0.2, rough=0.85,
            var=0.18, bump=0.15):
    m = new_mat(name)
    g = G(m)
    base = g.attr("Col").outputs["Color"]
    pos = g.geo().outputs["Position"]
    c, nv = patina(g, base, pos, streak=streak, dirt=dirt, var=var)
    n = g.bump(g.noise(pos, scale=7, detail=4).outputs["Fac"], strength=bump,
               distance=0.05)
    g.finish(g.bsdf(c, rough=rough, normal=n, spec=0.3), alpha_fade=fade)
    M[name] = m


def m_ashlar():
    """Dressed stone in courses (sillería), UV = (along wall, z)."""
    m = new_mat("ashlar")
    g = G(m)
    base = g.attr("Col").outputs["Color"]
    pos = g.geo().outputs["Position"]
    b = g.node("ShaderNodeTexBrick", offset=0.5, squash=1.0)
    g.link(g.uv(), b.inputs["Vector"])
    b.inputs["Scale"].default_value = 1.0
    b.inputs["Brick Width"].default_value = 1.1
    b.inputs["Row Height"].default_value = 0.5
    b.inputs["Mortar Size"].default_value = 0.012
    b.inputs["Color1"].default_value = (0.92, 0.9, 0.86, 1)
    b.inputs["Color2"].default_value = (1.0, 0.97, 0.9, 1)
    b.inputs["Mortar"].default_value = (0.55, 0.5, 0.45, 1)
    c, nv = patina(g, base, pos, streak=0.3, dirt=0.3, var=0.22)
    c = g.mix(1.0, c, b.outputs["Color"], "MULTIPLY")
    n = g.bump(add(g, mul(g, b.outputs["Fac"], -0.6),
                   mul(g, g.noise(pos, scale=11).outputs["Fac"], 0.4)),
               strength=0.35, distance=0.04)
    g.finish(g.bsdf(c, rough=0.9, normal=n), alpha_fade=True)
    M["ashlar"] = m


def m_brick():
    """Load-bearing brick; courses keyed to world z so a wall that rises out
    of the ground reads as being laid course by course."""
    m = new_mat("brick")
    g = G(m)
    pos = g.geo().outputs["Position"]
    uvs = g.sep(g.uv())
    zs = g.sep(pos)[2]
    vec = g.comb(uvs[0], zs, 0)
    b = g.node("ShaderNodeTexBrick", offset=0.5)
    g.link(vec, b.inputs["Vector"])
    b.inputs["Scale"].default_value = 1.0
    b.inputs["Brick Width"].default_value = 0.29
    b.inputs["Row Height"].default_value = 0.085
    b.inputs["Mortar Size"].default_value = 0.012
    b.inputs["Bias"].default_value = 0.2
    b.inputs["Color1"].default_value = (*L(168, 82, 52), 1)
    b.inputs["Color2"].default_value = (*L(196, 112, 70), 1)
    b.inputs["Mortar"].default_value = (*L(200, 188, 165), 1)
    c, nv = patina(g, b.outputs["Color"], pos, streak=0.15, dirt=0.3, var=0.35,
                   scale=1.2)
    n = g.bump(mul(g, b.outputs["Fac"], -1.0), strength=0.4, distance=0.02)
    g.finish(g.bsdf(c, rough=0.92, normal=n), alpha_fade=True)
    M["brick"] = m


def m_wood():
    m = new_mat("wood")
    g = G(m)
    base = g.attr("Col").outputs["Color"]
    uv = g.sep(g.uv())
    vec = g.comb(mul(g, uv[0], 40.0), mul(g, uv[1], 1.4), g.attr("Col").outputs["Alpha"])
    grain = g.noise(vec, scale=1.0, detail=6, rough=0.7, dist=0.6).outputs["Fac"]
    pos = g.geo().outputs["Position"]
    weather = g.noise(pos, scale=0.8, detail=3).outputs["Fac"]
    c = g.mix(mul(g, grain, 0.6), base, (0.55, 0.5, 0.45, 1), "MULTIPLY")
    c = g.mix(mul(g, sstep(g, 0.4, 0.8, weather), 0.55), c, L(150, 145, 135))
    n = g.bump(grain, strength=0.25, distance=0.02)
    g.finish(g.bsdf(c, rough=0.9, normal=n), alpha_fade=True)
    M["wood"] = m


def m_earth():
    m = new_mat("earth")
    g = G(m)
    pos = g.geo().outputs["Position"]
    n1 = g.noise(pos, scale=0.35, detail=6, rough=0.6).outputs["Fac"]
    n2 = g.noise(pos, scale=6, detail=4, rough=0.7).outputs["Fac"]
    vor = g.node("ShaderNodeTexVoronoi")
    g.link(pos, vor.inputs["Vector"])
    vor.inputs["Scale"].default_value = 14
    pebble = lt(g, vor.outputs["Distance"], 0.12)
    c = g.ramp(n1, [(0.25, L(96, 70, 46)), (0.5, L(138, 104, 70)),
                    (0.75, L(170, 140, 100))])
    c = g.mix(mul(g, sub(g, n2, 0.5), 0.6), c, (0.4, 0.33, 0.25, 1), "OVERLAY")
    c = g.mix(mul(g, pebble, 0.5), c, L(190, 180, 160))
    n = g.bump(add(g, n2, mul(g, pebble, 0.3)), strength=0.7, distance=0.08)
    g.finish(g.bsdf(c, rough=0.98, normal=n), alpha_fade=True)
    M["earth"] = m


def m_glass():
    m = new_mat("glass")
    g = G(m)
    base = g.attr("Col").outputs["Color"]
    g.finish(g.bsdf(base, rough=0.12, spec=0.9, metallic=0.2))
    M["glass"] = m


def m_people():
    m = new_mat("people")
    g = G(m)
    base = g.attr("Col").outputs["Color"]
    g.finish(g.bsdf(base, rough=0.75, sheen=0.4), alpha_fade=True)
    M["people"] = m


def m_foliage():
    m = new_mat("foliage")
    g = G(m)
    base = g.attr("Col").outputs["Color"]
    pos = g.geo().outputs["Position"]
    n1 = g.noise(pos, scale=1.6, detail=5, rough=0.65).outputs["Fac"]
    n2 = g.noise(pos, scale=7.0, detail=3, rough=0.7).outputs["Fac"]
    k = add(g, 0.55, mul(g, n1, 0.9))
    c = g.mix(1.0, base, g.comb(k, k, mul(g, k, 0.9)), "MULTIPLY")
    c = g.mix(mul(g, sstep(g, 0.62, 0.8, n2), 0.5), c, L(170, 160, 70))
    n = g.bump(add(g, n1, n2), strength=0.8, distance=0.25)
    g.finish(g.bsdf(c, rough=0.7, normal=n, spec=0.25, sss=0.08), alpha_fade=True)
    M["foliage"] = m


def m_cobble():
    m = new_mat("cobble")
    g = G(m)
    pos = g.geo().outputs["Position"]
    b = g.node("ShaderNodeTexBrick", offset=0.5)
    g.link(pos, b.inputs["Vector"])
    b.inputs["Brick Width"].default_value = 0.22
    b.inputs["Row Height"].default_value = 0.13
    b.inputs["Mortar Size"].default_value = 0.018
    b.inputs["Color1"].default_value = (*L(118, 108, 96), 1)
    b.inputs["Color2"].default_value = (*L(142, 132, 116), 1)
    b.inputs["Mortar"].default_value = (*L(80, 72, 62), 1)
    n1 = g.noise(pos, scale=0.08, detail=5).outputs["Fac"]
    n2 = g.noise(pos, scale=0.9, detail=4).outputs["Fac"]
    k = add(g, 0.7, mul(g, n1, 0.55))
    c = g.mix(1.0, b.outputs["Color"], g.comb(k, k, k), "MULTIPLY")
    c = g.mix(mul(g, sstep(g, 0.55, 0.8, n2), 0.35), c, L(70, 62, 52))
    n = g.bump(mul(g, b.outputs["Fac"], -1), strength=0.3, distance=0.02)
    g.finish(g.bsdf(c, rough=0.8, normal=n, spec=0.35))
    M["cobble"] = m


def m_sidewalk():
    m = new_mat("sidewalk")
    g = G(m)
    pos = g.geo().outputs["Position"]
    base = g.attr("Col").outputs["Color"]
    b = g.node("ShaderNodeTexBrick", offset=0.0)
    g.link(pos, b.inputs["Vector"])
    b.inputs["Brick Width"].default_value = 0.4
    b.inputs["Row Height"].default_value = 0.4
    b.inputs["Mortar Size"].default_value = 0.008
    b.inputs["Color1"].default_value = (0.95, 0.93, 0.9, 1)
    b.inputs["Color2"].default_value = (1, 0.98, 0.95, 1)
    b.inputs["Mortar"].default_value = (0.7, 0.66, 0.6, 1)
    c, nv = patina(g, base, pos, streak=0, dirt=0, var=0.3, scale=0.3)
    c = g.mix(1.0, c, b.outputs["Color"], "MULTIPLY")
    g.finish(g.bsdf(c, rough=0.85))
    M["sidewalk"] = m


def m_deck(x0, x1):
    """Raw concrete deck that is progressively paved (animated mask)."""
    m = new_mat("deck")
    g = G(m)
    pos = g.geo().outputs["Position"]
    sx = g.sep(pos)
    rv = g.val(-0.3)
    LIVE["deck_reveal"] = rv
    fx = g.math("DIVIDE", sub(g, sx[0], x0), x1 - x0)
    n = g.noise(pos, scale=0.15, detail=4).outputs["Fac"]
    front = add(g, sub(g, rv, fx), mul(g, sub(g, n, 0.5), 0.35))
    mask = sstep(g, 0.0, 0.06, front)
    conc = g.ramp(g.noise(pos, scale=0.5, detail=5).outputs["Fac"],
                  [(0.3, L(140, 136, 126)), (0.7, L(176, 170, 156))])
    # paving: diagonal checker of cream limestone and fired-clay tiles
    rot = g.comb(add(g, sx[0], sx[1]), sub(g, sx[0], sx[1]), 0)
    ck = g.node("ShaderNodeTexChecker")
    g.link(rot, ck.inputs["Vector"])
    ck.inputs["Scale"].default_value = 0.9
    ck.inputs["Color1"].default_value = (*L(214, 196, 162), 1)
    ck.inputs["Color2"].default_value = (*L(190, 120, 84), 1)
    pv, nv = patina(g, ck.outputs["Color"], pos, streak=0, dirt=0, var=0.3,
                    scale=0.4)
    c = g.mix(mask, conc, pv)
    rim = mul(g, band(g, front, -0.01, 0.025), 0.5)  # wet mortar line at the front
    c = g.mix(rim, c, L(90, 84, 74))
    g.finish(g.bsdf(c, rough=0.8))
    M["deck"] = m


def m_metal():
    m = new_mat("metal")
    g = G(m)
    base = g.attr("Col").outputs["Color"]
    pos = g.geo().outputs["Position"]
    c, nv = patina(g, base, pos, streak=0.3, dirt=0.0, var=0.3)
    g.finish(g.bsdf(c, rough=0.45, metallic=0.6, spec=0.5), alpha_fade=True)
    M["metal"] = m


def m_water():
    m = new_mat("water")
    g = G(m)
    pos = g.geo().outputs["Position"]
    rip = g.noise(pos, scale=3.0, detail=3).outputs["Fac"]
    n = g.bump(rip, strength=0.3, distance=0.05)
    g.finish(g.bsdf(L(70, 110, 120), rough=0.05, spec=1.0, normal=n))
    M["water"] = m


def m_flowers():
    m = new_mat("flowers")
    g = G(m)
    pos = g.geo().outputs["Position"]
    bloom = g.val(0.0)
    LIVE["bloom"] = bloom
    vor = g.node("ShaderNodeTexVoronoi")
    g.link(pos, vor.inputs["Vector"])
    vor.inputs["Scale"].default_value = 3.2
    fc = g.ramp(vor.outputs["Color"],
                [(0.0, L(196, 40, 52)), (0.25, L(236, 160, 40)),
                 (0.45, L(232, 120, 150)), (0.65, L(245, 236, 220)),
                 (0.85, L(140, 70, 150)), (1.0, L(220, 70, 40))])
    n = g.noise(pos, scale=2.5, detail=4).outputs["Fac"]
    leaf = g.mix(n, L(46, 82, 34), L(80, 116, 50))
    k = mul(g, bloom, sstep(g, 0.38, 0.6, n))
    c = g.mix(k, leaf, fc)
    nb = g.bump(n, strength=0.8, distance=0.1)
    g.finish(g.bsdf(c, rough=0.7, normal=nb))
    M["flowers"] = m


def m_cloudshadow():
    m = new_mat("cloudshadow")
    g = G(m)
    tc = g.texco().outputs["Object"]
    n = g.noise(tc, scale=0.012, detail=5, rough=0.55, dist=0.4).outputs["Fac"]
    a = sstep(g, 0.5, 0.66, n)
    a = mul(g, a, 0.62)
    tr = g.node("ShaderNodeBsdfTransparent")
    df = g.node("ShaderNodeBsdfDiffuse")
    mx = g.node("ShaderNodeMixShader")
    g.link(a, mx.inputs[0])
    g.link(tr.outputs[0], mx.inputs[1])
    g.link(df.outputs[0], mx.inputs[2])
    g.finish(mx.outputs[0])
    M["cloudshadow"] = m


def m_rooftile():
    m = new_mat("rooftile")
    g = G(m)
    base = g.attr("Col").outputs["Color"]
    pos = g.geo().outputs["Position"]
    uv = g.sep(g.uv())
    rows = g.math("FRACT", mul(g, uv[0], 4.0))
    ridge = g.math("SINE", mul(g, rows, 3.14159))
    c, nv = patina(g, base, pos, streak=0.0, dirt=0.0, var=0.4, scale=0.9)
    lich = sstep(g, 0.6, 0.75, g.noise(pos, scale=1.5, detail=5).outputs["Fac"])
    c = g.mix(mul(g, lich, 0.5), c, L(90, 88, 70))
    c = g.mix(mul(g, sub(g, 1.0, ridge), 0.4), c, (0.3, 0.2, 0.15, 1), "MULTIPLY")
    n = g.bump(ridge, strength=0.5, distance=0.05)
    g.finish(g.bsdf(c, rough=0.75, normal=n))
    M["rooftile"] = m


def m_terrace():
    m = new_mat("terrace")
    g = G(m)
    base = g.attr("Col").outputs["Color"]
    pos = g.geo().outputs["Position"]
    b = g.node("ShaderNodeTexBrick", offset=0.0)
    g.link(pos, b.inputs["Vector"])
    b.inputs["Brick Width"].default_value = 0.28
    b.inputs["Row Height"].default_value = 0.28
    b.inputs["Mortar Size"].default_value = 0.01
    b.inputs["Color1"].default_value = (0.95, 0.93, 0.9, 1)
    b.inputs["Color2"].default_value = (1, 1, 1, 1)
    b.inputs["Mortar"].default_value = (0.75, 0.75, 0.75, 1)
    c, nv = patina(g, base, pos, streak=0, dirt=0, var=0.45, scale=0.25)
    c = g.mix(1.0, c, b.outputs["Color"], "MULTIPLY")
    stain = sstep(g, 0.58, 0.8, g.noise(pos, scale=0.3, detail=6).outputs["Fac"])
    c = g.mix(mul(g, stain, 0.45), c, L(90, 80, 70))
    g.finish(g.bsdf(c, rough=0.9))
    M["terrace"] = m


def m_flag():
    m = new_mat("flag")
    g = G(m)
    uv = g.sep(g.uv())
    # Senyera: four red bars on gold, blue crowned band at the hoist
    bars = lt(g, g.math("FRACT", mul(g, uv[1], 4.5)), 0.5)
    c = g.mix(bars, L(236, 186, 40), L(196, 30, 36))
    c = g.mix(lt(g, uv[0], 0.22), c, L(30, 60, 140))
    g.finish(g.bsdf(c, rough=0.7, sheen=0.5))
    M["flag"] = m


def m_paint():
    """Vehicles / painted joinery: Col with light dust and a bit of gloss."""
    m = new_mat("paint")
    g = G(m)
    base = g.attr("Col").outputs["Color"]
    pos = g.geo().outputs["Position"]
    c, nv = patina(g, base, pos, streak=0.1, dirt=0.35, var=0.1)
    g.finish(g.bsdf(c, rough=0.45, spec=0.5), alpha_fade=True)
    M["paint"] = m


def build_all(deck_x0, deck_x1):
    m_facade()
    m_plain()
    m_plain("plain_fade", fade=True)
    m_plain("stucco", streak=0.35, dirt=0.3, var=0.25)
    m_ashlar()
    m_brick()
    m_wood()
    m_earth()
    m_glass()
    m_people()
    m_foliage()
    m_cobble()
    m_sidewalk()
    m_deck(deck_x0, deck_x1)
    m_metal()
    m_water()
    m_flowers()
    m_cloudshadow()
    m_rooftile()
    m_terrace()
    m_flag()
    m_paint()
    return M
