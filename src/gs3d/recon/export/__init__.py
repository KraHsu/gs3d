"""Export a segmented 3DGS scene into per-object simulation assets.

Pipeline (segmented reference-3DGS checkpoint -> physics-ready scene):

    clusters.py   split by `_cluster_indices` -> per-object point subsets + manifest
    meshing.py    per-object point subset    -> visual + collision mesh (trimesh/CoACD)
    urdf.py       per-object mesh + physics   -> URDF (+ a scene) for Genesis/PyBullet

Each stage is a thin, inspectable step: the cluster split is simulator-agnostic
and uses only numpy/plyfile, so you can eyeball the per-object PLYs before paying
for meshing. See ``docs/3dgs-to-physics-sim-survey.md`` for the rationale.
"""
