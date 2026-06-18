"""Introspect the COLMAP reconstruction objects for this pycolmap version."""
import numpy as np
import pycolmap

rec = pycolmap.Reconstruction("data/table_sharp/sparse/0")
img = next(iter(rec.images.values()))
print("IMAGE attrs:", [a for a in dir(img) if not a.startswith("_")])
cfw = img.cam_from_world
cfw = cfw() if callable(cfw) else cfw
print("cam_from_world type:", type(cfw).__name__)
print("cfw attrs:", [a for a in dir(cfw) if not a.startswith("_")])
try:
    print("cfw.matrix() shape:", np.asarray(cfw.matrix()).shape)
except Exception as e:
    print("matrix err:", e)
if hasattr(cfw, "rotation"):
    r = cfw.rotation
    print("rotation attrs:", [a for a in dir(r) if not a.startswith("_")])

cam = next(iter(rec.cameras.values()))
print("CAMERA attrs:", [a for a in dir(cam) if not a.startswith("_")])
print("cam.model:", cam.model, "params:", list(cam.params))

pt = next(iter(rec.points3D.values()))
print("POINT3D attrs:", [a for a in dir(pt) if not a.startswith("_")])
print("xyz:", np.asarray(pt.xyz), "color:", np.asarray(pt.color))
