"""A metric, gravity-aligned Genesis scene with photorealistic gsplat observations.

This is the actual *training-environment* deliverable (vs the throwaway viewers):
a robot-agnostic scene you drop your own robot into, step with Genesis physics,
and observe through a photorealistic camera. The appearance is the real 3DGS
capture (family C / SplatSim pattern) — physics runs on cheap convex proxies,
rendering uses the real Gaussians, so visual observations carry the captured look
for sim2real.

Pipeline (all baked at construction):
  1. metric scale from D435i depth        (scale.estimate_metric_scale)
  2. gravity/table alignment to z-up, table at z=0   (align.*)
  3. split clusters -> small "objects" (dynamic rigid bodies, convex collision +
     mass) vs everything else "background" (render-only context + a support plane)
  4. all Gaussians transformed into the aligned metric frame

`GS3DSimScene` then exposes a Genesis `gs.Scene` (add your robot before `build()`),
`step()`, and `render(c2w, K, W, H) -> RGB` (background Gaussians fixed + each
object's Gaussians at its live physics pose, rasterised with gsplat).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from .align import alignment_transform, fit_ground_plane
from .sim_render import _quat_mul, _quat_to_rotmat


def _quat_from_matrix(R: np.ndarray) -> np.ndarray:
    """3x3 rotation -> wxyz quaternion."""
    w = math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    w = max(w, 1e-8)
    x = (R[2, 1] - R[1, 2]) / (4 * w)
    y = (R[0, 2] - R[2, 0]) / (4 * w)
    z = (R[1, 0] - R[0, 1]) / (4 * w)
    return np.array([w, x, y, z], dtype=np.float32)


def _filter_object(local, scales, opac, *, opacity_min, radius_q, scale_q, aspect_max):
    """Boolean keep-mask dropping floaters/needles that streak when rotated."""
    dist = local.norm(dim=-1)
    sz = torch.exp(scales)
    biggest = sz.max(dim=-1).values
    aspect = biggest / sz.min(dim=-1).values.clamp_min(1e-8)
    keep = torch.sigmoid(opac) >= opacity_min
    keep &= dist <= torch.quantile(dist, radius_q)
    keep &= biggest <= torch.quantile(biggest, scale_q)
    keep &= aspect <= aspect_max
    return keep


class GS3DSimScene:
    """Metric, z-up Genesis scene + photoreal gsplat observations (robot-agnostic).

    Typical use::

        env = GS3DSimScene(checkpoint, data_dir, scale=None)   # bakes scale+align
        env.genesis_scene.add_entity(gs.morphs.URDF(file="my_robot.urdf"))  # your robot
        env.build()
        for _ in range(n):
            env.step()
            rgb = env.render(c2w, K, W, H)     # photorealistic observation
    """

    def __init__(
        self,
        checkpoint: str | Path,
        data_dir: str | Path,
        *,
        scale: float | None = None,
        object_ids: list[int] | None = None,
        max_object_size: float = 0.45,
        min_points: int = 200,
        density: float = 300.0,
        backend: str = "cpu",  # physics on CPU; gsplat render stays on GPU (no VRAM contention)
        device: str = "cuda",
        opacity_min: float = 0.1,
        radius_q: float = 0.98,
        scale_q: float = 0.99,
        aspect_max: float = 30.0,
        work_dir: str | Path | None = None,
    ):
        import trimesh

        from .._cuda import ensure_cuda_toolkit
        from ..inria import load_inria_checkpoint
        from .scale import estimate_metric_scale

        ensure_cuda_toolkit()
        self.device = device
        self.backend = backend
        data_dir = Path(data_dir)
        cameras_json = data_dir / "output" / "cameras.json"
        depth_dir = data_dir / "table_new" / "depth"
        self.work_dir = Path(work_dir) if work_dir else (data_dir / "_env")
        (self.work_dir / "urdf").mkdir(parents=True, exist_ok=True)

        # 1. metric scale -----------------------------------------------------
        if scale is None:
            scale = estimate_metric_scale(checkpoint, cameras_json, depth_dir, device=device)
        self.scale = float(scale)

        model = load_inria_checkpoint(checkpoint, device=device)
        sp, self.sh_degree = model.splats, model.sh_degree
        ids = model.cluster_indices.cpu().numpy()
        means_m = (sp["means"] / self.scale)          # metric, COLMAP orientation
        mn = means_m.detach().cpu().numpy()

        # cluster bookkeeping
        import collections
        cnt = collections.Counter(ids.tolist())
        kept = [c for c, n in cnt.items() if n >= min_points]
        extents = {c: float(np.linalg.norm(mn[ids == c].max(0) - mn[ids == c].min(0))) for c in kept}
        bg_id = max(extents, key=lambda c: extents[c])  # biggest = room/background
        if object_ids is not None:
            obj_ids = [int(c) for c in object_ids if c in cnt]
        else:
            obj_ids = [c for c in kept if c != bg_id and extents[c] <= max_object_size]
        obj_centroid = np.median(mn[np.isin(ids, obj_ids)], axis=0) if obj_ids else mn.mean(0)

        # 2. gravity / table alignment ---------------------------------------
        n0, _, _ = fit_ground_plane(mn, thresh=0.01, n_iter=1500)
        if n0[1] < 0:
            n0 = -n0
        h = mn @ n0
        h0 = float(np.median(h[(h > np.percentile(h, 40)) & (h < np.percentile(h, 70))]))
        band = mn[np.abs(h - h0) < 0.05]
        nt, pt, _ = fit_ground_plane(band, thresh=0.01, n_iter=800)
        T = alignment_transform(nt, pt, obj_centroid)
        self.T = T
        R = torch.tensor(T[:3, :3], dtype=torch.float32, device=device)
        t = torch.tensor(T[:3, 3], dtype=torch.float32, device=device)
        qR = torch.tensor(_quat_from_matrix(T[:3, :3]), device=device)

        # 3. refine objects: keep only compact clusters actually resting ON the
        # table (aligned base z ~ 0). Drops shelf/wall items and floating shards
        # that would otherwise fall through or drop from mid-air.
        mn_aligned = (T[:3, :3] @ mn.T).T + T[:3, 3]
        if object_ids is None:  # auto: keep compact clusters resting on the table
            on_table = []
            for c in obj_ids:
                base_z = float(np.percentile(mn_aligned[ids == c][:, 2], 2))
                if -0.25 <= base_z <= 0.25:  # near the table (allow segmentation bleed)
                    on_table.append(c)
            obj_ids = on_table

        # Pin z=0 to the surface the selected objects actually rest on (their base
        # height), not just the RANSAC plane — which may latch onto the floor and
        # leave objects floating/sunk. This makes the Genesis ground plane coincide
        # with the real table under the objects.
        if obj_ids:
            obj_base = float(np.percentile(mn_aligned[np.isin(ids, obj_ids)][:, 2], 2))
            T[2, 3] -= obj_base
            self.T = T
            t = torch.tensor(T[:3, 3], dtype=torch.float32, device=device)
            mn_aligned[:, 2] -= obj_base

        # 4. split + transform Gaussians into aligned metric frame ------------
        log_inv_s = math.log(1.0 / self.scale)
        obj_mask = np.isin(ids, obj_ids)

        # background: render-only, baked in the aligned frame
        bgm = torch.tensor(~obj_mask, device=device)
        bg_means = means_m[bgm] @ R.T + t
        bg_quats = _quat_mul(qR, sp["quats"][bgm] / sp["quats"][bgm].norm(dim=-1, keepdim=True))
        self.bg = {
            "means": bg_means, "quats": bg_quats,
            "scales": sp["scales"][bgm] + log_inv_s,
            "opacities": sp["opacities"][bgm],
            "sh0": sp["sh0"][bgm], "shN": sp["shN"][bgm],
        }

        # objects: local metric Gaussians + collision URDF + rest pose
        self.objects = []
        for c in obj_ids:
            m = torch.tensor(ids == c, device=device)
            com_m = means_m[m].median(dim=0).values            # metric COM
            local = means_m[m] - com_m
            scales = sp["scales"][m] + log_inv_s
            opac = sp["opacities"][m]
            keep = _filter_object(local, scales, opac, opacity_min=opacity_min,
                                  radius_q=radius_q, scale_q=scale_q, aspect_max=aspect_max)
            obj = {
                "id": int(c),
                "local": local[keep], "quats": sp["quats"][m][keep],
                "scales": scales[keep], "opac": opac[keep],
                "sh0": sp["sh0"][m][keep], "shN": sp["shN"][m][keep],
            }
            # convex-hull collision in object-local metric frame -> URDF
            lp = local[keep].detach().cpu().numpy()
            hull = trimesh.points.PointCloud(lp).convex_hull
            hull.density = density
            urdf = self._write_urdf(int(c), hull)
            # rest pose: COM in aligned xy, dropped so the *hull's* lowest point
            # (after the alignment rotation it spawns with) sits exactly on z=0 ->
            # the object starts at rest on the table, consistent with its collider.
            com_np = com_m.detach().cpu().numpy()
            com_aligned = (T[:3, :3] @ com_np + T[:3, 3]).astype(np.float32)
            hull_z = (T[:3, :3] @ hull.vertices.T).T[:, 2]
            obj["rest_pos"] = np.array(
                [com_aligned[0], com_aligned[1], -float(hull_z.min())], dtype=np.float32
            )
            obj["rest_quat"] = _quat_from_matrix(T[:3, :3])
            obj["urdf"] = urdf
            obj["mass"] = float(hull.mass)
            self.objects.append(obj)

        self.table_z = 0.0
        self._built = False
        self._ents = []
        print(f"[env] scale={self.scale:.3f} u/m | {len(self.objects)} dynamic objects, "
              f"{int(bgm.sum())} background gaussians | table at z=0")

        # create the Genesis scene now so the user can add a robot before build()
        import genesis as gs
        try:
            gs.init(backend=getattr(gs, backend))
        except Exception as e:
            if "already" not in str(e).lower():  # tolerate re-init in same process
                raise
        self.gs = gs
        self.genesis_scene = gs.Scene(show_viewer=False)
        self.genesis_scene.add_entity(gs.morphs.Plane())  # table support at z=0
        for obj in self.objects:
            ent = self.genesis_scene.add_entity(
                gs.morphs.URDF(file=str(obj["urdf"]),
                               pos=tuple(float(x) for x in obj["rest_pos"]),
                               quat=tuple(float(x) for x in obj["rest_quat"]),
                               fixed=False)
            )
            self._ents.append(ent)

    def _write_urdf(self, cid: int, hull) -> Path:
        mdir = self.work_dir / "urdf" / "meshes"
        mdir.mkdir(parents=True, exist_ok=True)
        obj_path = mdir / f"{cid:04d}.obj"
        hull.export(obj_path)
        it = hull.moment_inertia
        urdf = self.work_dir / "urdf" / f"{cid:04d}.urdf"
        rel = f"meshes/{obj_path.name}"
        urdf.write_text(f"""<?xml version="1.0"?>
<robot name="obj_{cid:04d}">
  <link name="base_link">
    <inertial><origin xyz="0 0 0"/><mass value="{max(hull.mass,1e-4):.6g}"/>
      <inertia ixx="{it[0,0]:.6g}" ixy="{it[0,1]:.6g}" ixz="{it[0,2]:.6g}" iyy="{it[1,1]:.6g}" iyz="{it[1,2]:.6g}" izz="{it[2,2]:.6g}"/></inertial>
    <visual><geometry><mesh filename="{rel}"/></geometry></visual>
    <collision><geometry><mesh filename="{rel}"/></geometry></collision>
  </link>
</robot>
""")
        return urdf

    def build(self, settle: int = 30) -> None:
        """Finalise the Genesis scene (call after adding your robot) and let the
        objects settle into contact so the initial state is at rest."""
        self.genesis_scene.build()
        self._built = True
        for _ in range(settle):
            self.genesis_scene.step()

    def step(self, n: int = 1) -> None:
        if not self._built:
            self.build()
        for _ in range(n):
            self.genesis_scene.step()

    def _object_poses(self):
        """Current (pos, quat wxyz) of each dynamic object from Genesis."""
        pos = [np.asarray(e.get_pos().cpu(), dtype=np.float32) for e in self._ents]
        quat = [np.asarray(e.get_quat().cpu(), dtype=np.float32) for e in self._ents]
        return pos, quat

    @torch.no_grad()
    def _scene_splats(self):
        """Full splats dict for the current physics state (background + posed objects)."""
        means = [self.bg["means"]]; quats = [self.bg["quats"]]
        scales = [self.bg["scales"]]; opac = [self.bg["opacities"]]
        sh0 = [self.bg["sh0"]]; shN = [self.bg["shN"]]
        if self._built:
            pos, quat = self._object_poses()
        else:
            pos = [o["rest_pos"] for o in self.objects]
            quat = [o["rest_quat"] for o in self.objects]
        for o, p, q in zip(self.objects, pos, quat):
            p = torch.as_tensor(p, dtype=torch.float32, device=self.device)
            q = torch.as_tensor(q, dtype=torch.float32, device=self.device)
            R = _quat_to_rotmat(q)
            means.append(o["local"] @ R.T + p)
            quats.append(_quat_mul(q, o["quats"]))
            scales.append(o["scales"]); opac.append(o["opac"])
            sh0.append(o["sh0"]); shN.append(o["shN"])
        return {
            "means": torch.cat(means), "quats": torch.cat(quats),
            "scales": torch.cat(scales), "opacities": torch.cat(opac),
            "sh0": torch.cat(sh0), "shN": torch.cat(shN),
        }

    def _demo_camera(self, width, height, fov_deg=50.0):
        """A fixed 3/4 view framing the dynamic objects (aligned z-up world)."""
        from .sim_render import _look_at

        if self.objects:
            center = np.mean([o["rest_pos"] for o in self.objects], axis=0).astype(np.float32)
        else:
            center = self.bg["means"].mean(0).cpu().numpy()
        center[2] = 0.02
        eye = center + np.array([0.28, 0.28, 0.62], dtype=np.float32)  # mostly overhead
        c2w = _look_at(eye, center, np.array([0.0, 0.0, 1.0]))
        f = 0.5 * width / math.tan(0.5 * math.radians(fov_deg))
        K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], dtype=np.float32)
        return c2w, K

    def perturb(self, *, lift: float = 0.12, speed: float = 0.4) -> None:
        """Shove the objects (lift + horizontal velocity) so they fall, collide and
        tumble — a robot-free way to show physical interaction with the real objects."""
        if not self._built:
            self.build()
        n = max(1, len(self._ents))
        for i, ent in enumerate(self._ents):
            ang = 2 * math.pi * i / n
            try:
                p = np.asarray(ent.get_pos().cpu(), dtype=np.float32)
                ent.set_pos(np.array([p[0], p[1], p[2] + lift], dtype=np.float32),
                            relative=False, zero_velocity=True)
                ent.set_dofs_velocity(
                    np.array([speed * math.cos(ang), speed * math.sin(ang), 0, 0, 0, 0],
                             dtype=np.float32)
                )
            except Exception as e:
                print(f"[demo] perturb fallback (lift only) for object {i}: {e}")

    def demo(
        self,
        out_mp4: str | Path,
        *,
        steps: int = 300,
        fps: int = 60,
        width: int = 960,
        height: int = 720,
        settle_frames: int = 20,
        lift: float = 0.12,
        speed: float = 0.4,
    ) -> None:
        """Record a photoreal object-interaction clip: settle -> shove -> tumble."""
        import imageio.v2 as imageio

        if not self._built:
            self.build()
        c2w, K = self._demo_camera(width, height)
        writer = imageio.get_writer(str(out_mp4), fps=fps, macro_block_size=1)
        for _ in range(settle_frames):       # a beat at rest before the shove
            writer.append_data(self.render(c2w, K, width, height))
        self.perturb(lift=lift, speed=speed)
        for _ in range(steps):
            self.step()
            writer.append_data(self.render(c2w, K, width, height))
        writer.close()
        print(f"[demo] wrote {out_mp4} ({settle_frames + steps} frames @ {fps} fps)")

    @torch.no_grad()
    def render(self, c2w, K, width: int, height: int) -> np.ndarray:
        """Photorealistic RGB observation from a camera (c2w 4x4, K 3x3), uint8 HxWx3."""
        from ..model import rasterize_splats

        c2w = torch.as_tensor(c2w, dtype=torch.float32, device=self.device)
        K = torch.as_tensor(K, dtype=torch.float32, device=self.device)
        viewmat = torch.linalg.inv(c2w)[None]
        splats = self._scene_splats()
        colors, _, _ = rasterize_splats(splats, viewmat, K[None], width, height, self.sh_degree)
        return (colors[0, ..., :3].clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
