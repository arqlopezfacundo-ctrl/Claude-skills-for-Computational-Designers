"""Render settings, golden-hour sun, sky, orthographic axonometric camera."""
import math

import bpy
from mathutils import Vector

from . import plan

CAM_TARGET = (14.0, -2.0, 0.0)
CAM_ELEV = 42.0          # degrees above horizon
CAM_AZ = -28.0           # degrees: view direction rotated from +y
ORTHO = 182.0
SUN_ELEV = 27.0
SUN_AZ = 128.0           # light coming from the left-front of the view


def setup(res=(1920, 1080), samples=48):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = samples
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.adaptive_threshold = 0.03
    sc.cycles.use_denoising = True
    sc.cycles.denoiser = "OPENIMAGEDENOISE"
    sc.cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    sc.cycles.max_bounces = 4
    sc.cycles.diffuse_bounces = 2
    sc.cycles.glossy_bounces = 1
    sc.cycles.transparent_max_bounces = 6
    sc.cycles.transmission_bounces = 1
    sc.cycles.caustics_reflective = False
    sc.cycles.caustics_refractive = False
    sc.cycles.blur_glossy = 1.0
    sc.cycles.sample_clamp_indirect = 4.0
    sc.render.use_persistent_data = True
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = False
    sc.view_settings.view_transform = "Standard"
    sc.render.image_settings.media_type = "MULTI_LAYER_IMAGE"
    sc.render.image_settings.file_format = "OPEN_EXR_MULTILAYER"
    sc.render.image_settings.color_depth = "16"
    sc.render.image_settings.exr_codec = "ZIP"
    vl = sc.view_layers[0]
    vl.use_pass_z = True
    vl.use_pass_normal = True
    vl.use_pass_mist = True
    vl.use_pass_diffuse_color = True
    vl.use_pass_shadow = False
    try:
        sc.cycles.use_light_tree = True
    except Exception:
        pass
    sc.render.threads_mode = "AUTO"


def world():
    sc = bpy.context.scene
    w = bpy.data.worlds.new("sky")
    sc.world = w
    try:
        w.use_nodes = True
    except Exception:
        pass
    nt = w.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    lp = nt.nodes.new("ShaderNodeLightPath")
    mix = nt.nodes.new("ShaderNodeMixShader")
    bg2 = nt.nodes.new("ShaderNodeBackground")
    # sky dome: cool blue fill for shadows
    grad = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    nt.links.new(grad.outputs["Generated"], sep.inputs[0])
    cr = ramp.color_ramp
    cr.elements[0].position = 0.45
    cr.elements[0].color = (0.62, 0.55, 0.48, 1)
    cr.elements[1].position = 0.75
    cr.elements[1].color = (0.30, 0.45, 0.78, 1)
    nt.links.new(sep.outputs[2], ramp.inputs[0])
    nt.links.new(ramp.outputs[0], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 1.05
    bg2.inputs["Color"].default_value = (0.82, 0.74, 0.62, 1)   # warm paper haze for camera
    bg2.inputs["Strength"].default_value = 1.0
    nt.links.new(lp.outputs["Is Camera Ray"], mix.inputs[0])
    nt.links.new(bg.outputs[0], mix.inputs[1])
    nt.links.new(bg2.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    w.mist_settings.start = 0.0
    w.mist_settings.depth = 400.0
    w.mist_settings.falloff = "LINEAR"


def sun():
    d = bpy.data.lights.new("sun", "SUN")
    d.energy = 4.6
    d.color = (1.0, 0.80, 0.58)
    d.angle = math.radians(1.6)
    ob = bpy.data.objects.new("sun", d)
    bpy.context.scene.collection.objects.link(ob)
    ob.rotation_euler = (math.radians(90 - SUN_ELEV), 0, math.radians(SUN_AZ))
    return ob


def sun_dir():
    """Unit vector pointing *towards* the sun."""
    e, a = math.radians(SUN_ELEV), math.radians(SUN_AZ)
    # light rotated about X by (90-elev) then Z by az; it shines along -Z local
    v = Vector((0, 0, 1))
    from mathutils import Euler
    v.rotate(Euler((math.radians(90 - SUN_ELEV), 0, a)))
    return v


def camera():
    cd = bpy.data.cameras.new("cam")
    cd.type = "ORTHO"
    cd.ortho_scale = ORTHO
    cd.clip_start = 1
    cd.clip_end = 2000
    ob = bpy.data.objects.new("cam", cd)
    bpy.context.scene.collection.objects.link(ob)
    bpy.context.scene.camera = ob
    place_camera(ob, CAM_TARGET, ORTHO)
    return ob


def place_camera(ob, target, ortho, az=CAM_AZ, elev=CAM_ELEV):
    from mathutils import Euler
    rx = math.radians(90 - elev)
    rz = math.radians(az)
    ob.rotation_euler = Euler((rx, 0, rz))
    fwd = Vector((0, 0, -1))
    fwd.rotate(ob.rotation_euler)
    ob.location = Vector(target) - fwd * 600
    ob.data.ortho_scale = ortho
