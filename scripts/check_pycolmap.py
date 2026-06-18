"""Introspect the installed pycolmap API (sanity check before SfM)."""
import inspect

import pycolmap as p

print("version:", p.__version__)
for fn in ["extract_features", "match_exhaustive", "match_sequential", "incremental_mapping"]:
    print(f"  {fn}: {hasattr(p, fn)}")
print("CameraMode.SINGLE:", hasattr(p.CameraMode, "SINGLE"))
print("Device:", [d for d in dir(p.Device) if not d.startswith("_")])
print("extract_features sig:", str(inspect.signature(p.extract_features))[:240])
