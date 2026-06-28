"""Gravity / table-plane alignment for the captured scene.

COLMAP/3DGS world axes are arbitrary; a physics sim needs z-up gravity and a
support surface at a known height. We fit the dominant plane (the table) by
RANSAC, then build a rigid transform that rotates its normal to +z and drops the
plane to z=0. Applied to every Gaussian and object pose, this puts the scene in a
metric, z-up frame where objects rest on the z=0 plane — exactly what Genesis
(and a robot) expect. This is the GS<->sim alignment the survey flags as a
recurring footgun.
"""

from __future__ import annotations

import numpy as np


def fit_ground_plane(
    points: np.ndarray,
    *,
    thresh: float,
    n_iter: int = 2000,
    sample: int = 20000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RANSAC-fit the dominant plane. Returns (normal unit, point_on_plane, inlier_mask).

    ``thresh`` is the inlier distance (same units as ``points``). The plane with
    the most inliers wins — in a tabletop capture that is the table (or floor).
    """
    pts = points.astype(np.float64)
    if pts.shape[0] > sample:
        sel = np.linspace(0, pts.shape[0] - 1, sample).astype(int)
        fit_pts = pts[sel]
    else:
        fit_pts = pts

    best_n, best_d, best_inl = None, None, -1
    # Deterministic triplets: stride through the array so we don't need RNG.
    n = fit_pts.shape[0]
    step = max(1, n // 3 // max(1, n_iter))
    for k in range(n_iter):
        i = (k * 3 * step) % n
        j = (i + step) % n
        l = (i + 2 * step) % n
        p0, p1, p2 = fit_pts[i], fit_pts[j], fit_pts[l]
        nrm = np.cross(p1 - p0, p2 - p0)
        ln = np.linalg.norm(nrm)
        if ln < 1e-9:
            continue
        nrm = nrm / ln
        d = -nrm @ p0
        dist = np.abs(fit_pts @ nrm + d)
        inl = int((dist < thresh).sum())
        if inl > best_inl:
            best_inl, best_n, best_d = inl, nrm, d

    # Refine with a least-squares fit over the inliers (PCA normal).
    dist = np.abs(fit_pts @ best_n + best_d)
    inliers = fit_pts[dist < thresh]
    c = inliers.mean(axis=0)
    _, _, vh = np.linalg.svd(inliers - c)
    normal = vh[-1]
    normal = normal / np.linalg.norm(normal)
    full_dist = np.abs((pts - c) @ normal)
    return normal.astype(np.float32), c.astype(np.float32), (full_dist < thresh)


def alignment_transform(
    normal: np.ndarray, point_on_plane: np.ndarray, up_reference: np.ndarray
) -> np.ndarray:
    """4x4 transform mapping the plane to z=0 with normal -> +z.

    ``up_reference`` (e.g. centroid of the objects, which sit *above* the table)
    disambiguates the normal sign so +z points up out of the table.
    """
    n = normal / np.linalg.norm(normal)
    if n @ (up_reference - point_on_plane) < 0:
        n = -n  # make the normal point toward the objects (up)

    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(n, z)
    s = np.linalg.norm(v)
    if s < 1e-8:
        R = np.eye(3) if n @ z > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        c = float(n @ z)
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))  # Rodrigues

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = -(R @ point_on_plane)  # plane point -> origin (so plane is z=0)
    return T
