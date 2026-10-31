import sys, time, random, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
from vlc import common, materials, stage, city, plan

bpy.ops.wm.read_factory_settings(use_empty=True)
rng = random.Random(7)
res = tuple(int(v) for v in sys.argv[1].split("x")) if len(sys.argv) > 1 else (960, 540)
samples = int(sys.argv[2]) if len(sys.argv) > 2 else 16
az = float(sys.argv[3]) if len(sys.argv) > 3 else stage.CAM_AZ
stage.setup(res, samples)
stage.world()
stage.sun()
cam = stage.camera()
ortho = float(sys.argv[4]) if len(sys.argv) > 4 else stage.ORTHO
stage.place_camera(cam, stage.CAM_TARGET, ortho, az=az)
M = materials.build_all(plan.X_MIN, plan.X_MAX)
t = time.time()
C = city.City(M, rng)
C.build()
print("build", time.time() - t)
sc = bpy.context.scene
sc.render.image_settings.media_type = "IMAGE"
sc.render.image_settings.file_format = "PNG"
sc.view_settings.view_transform = "AgX"
sc.render.filepath = "/tmp/city_test.png"
t = time.time()
bpy.ops.render.render(write_still=True)
print("render", time.time() - t)
