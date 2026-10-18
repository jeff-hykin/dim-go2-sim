# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""InteriorGS homes (SpatialVerse, CC-BY-NC-4.0), downloaded on first use and cooked into a
scene: Gaussian splats for the camera, the collision meshes for the lidar, and boxes for MuJoCo.

The splats and meshes come from SAGE-3D's mirrors: the splats as 3DGRUT's USDZ export, the
meshes as USD. Both are in the same frame: metres, Z up, the floor at about z = 0.

MuJoCo collides only with convex shapes and the meshes are open surfaces, so the surfaces are
voxelised and the voxels merged into boxes; the floor is a plane. The lidar casts against the
meshes themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import urllib.request
import zipfile

import numpy as np
from numpy.typing import NDArray

PREFIX = "interiorgs/"
SPLATS_URL = (
    "https://huggingface.co/datasets/spatialverse/SAGE-3D_InteriorGS_usdz/resolve/main/"
    "InteriorGS_usdz/{id}.usdz"
)
COLLISION_URL = (
    "https://huggingface.co/datasets/spatialverse/SAGE-3D_Collision_Mesh/resolve/main/"
    "Collision_Mesh/{id}/{id}_collision.usd"
)
# Bump when cooking changes, so cooked scenes are rebuilt.
COOK_VERSION = 1
VOXEL = 0.05  # metres; the boxes MuJoCo collides with are made of these
# Geometry this far above the floor is the floor's (rugs, thresholds); higher is an obstacle.
FLOOR_CLEARANCE = 0.01
# The robot never reaches higher than this above the floor, metres, so MuJoCo needs no boxes there.
COLLISION_HEIGHT = 2.0
# A spawn needs this much free floor around it, metres, and nothing below ROBOT_HEIGHT over it.
SPAWN_CLEARANCE = 0.5
ROBOT_HEIGHT = 0.6
SPAWN_SPACING = 1.5  # metres between spawns
MAX_SPAWNS = 8


@dataclass(frozen=True)
class InteriorScene:
    mjcf: Path
    splats: Path  # an NVIDIA NuRec container
    collision: Path  # .npz of the meshes' world-frame "vertices" and "faces", for the lidar
    floor_z: float
    spawns: list[tuple[float, float, float]]  # x, y, yaw; the most open first


def scene_id(package: str) -> str | None:
    """The InteriorGS id in a package name like "interiorgs/839920", else None."""
    return package[len(PREFIX) :] if package.startswith(PREFIX) else None


def load_scene(scene: str, cache_dir: Path | None = None) -> InteriorScene:
    """The cooked scene, downloading (~100-350 MB) and cooking it the first time."""
    cache = cache_dir or Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    folder = cache / "dim_go2_sim" / "interiorgs" / scene
    info_path = folder / "scene.json"
    info = json.loads(info_path.read_text()) if info_path.is_file() else {}
    if info.get("cook_version") != COOK_VERSION:
        info = _cook(scene, folder)
        info_path.write_text(json.dumps(info))
    return InteriorScene(
        mjcf=folder / f"{scene}.xml",
        splats=folder / f"{scene}.nurec",
        collision=folder / "collision.npz",
        floor_z=info["floor_z"],
        spawns=[tuple(s) for s in info["spawns"]],
    )


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(f".{os.getpid()}.part")
    with urllib.request.urlopen(url) as response, partial.open("wb") as file:
        shutil.copyfileobj(response, file, 1 << 20)
    partial.rename(path)


def _cook(scene: str, folder: Path) -> dict[str, object]:
    usdz, usd = folder / f"{scene}.usdz", folder / f"{scene}_collision.usd"
    if not (folder / f"{scene}.nurec").is_file():
        _download(SPLATS_URL.format(id=scene), usdz)
        with zipfile.ZipFile(usdz) as archive:
            archive.extract(f"{scene}.nurec", folder)
        usdz.unlink()
    if not (folder / "collision.npz").is_file():
        _download(COLLISION_URL.format(id=scene), usd)
        vertices, faces = usd_triangles(usd)
        np.savez(folder / "collision.npz", vertices=vertices, faces=faces)
        usd.unlink()
    mesh = np.load(folder / "collision.npz")
    vertices, faces = mesh["vertices"], mesh["faces"]
    floor_z = floor_height(vertices, faces)
    occupied, origin = voxelize(vertices, faces, floor_z)
    boxes = merged_boxes(occupied[:, :, : math.ceil(COLLISION_HEIGHT / VOXEL)])
    (folder / f"{scene}.xml").write_text(mjcf(boxes, origin, floor_z, scene))
    spawns = spawn_poses(vertices, faces, floor_z, occupied, origin)
    return {"cook_version": COOK_VERSION, "floor_z": floor_z, "spawns": spawns}


def usd_triangles(path: Path) -> tuple[NDArray[np.float32], NDArray[np.uint32]]:
    """Every mesh in a USD stage, triangulated, in the stage's world frame."""
    from pxr import Usd, UsdGeom  # type: ignore[import-untyped]

    stage = Usd.Stage.Open(str(path))
    transforms = UsdGeom.XformCache()
    scale = UsdGeom.GetStageMetersPerUnit(stage)
    vertices, faces, count = [], [], 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), np.float64)
        sizes = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
        if not len(points) or not len(sizes):
            continue
        matrix = np.asarray(transforms.GetLocalToWorldTransform(prim))  # row vectors
        vertices.append((points @ matrix[:3, :3] + matrix[3, :3]) * scale)
        starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        # A polygon of n corners is a fan of n - 2 triangles from its first corner.
        fan = [
            (s, s + k, s + k + 1)
            for s, n in zip(starts, sizes, strict=True)
            for k in range(1, n - 1)
        ]
        faces.append(indices[np.array(fan).reshape(-1, 3)] + count)
        count += len(points)
    return np.concatenate(vertices).astype(np.float32), np.concatenate(faces).astype(np.uint32)


def _area_normals(vertices: NDArray[np.float32], faces: NDArray[np.uint32]) -> NDArray[np.float64]:
    """Each triangle's normal, twice its area long."""
    corners = vertices[faces].astype(np.float64)
    return np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])


def floor_height(vertices: NDArray[np.float32], faces: NDArray[np.uint32]) -> float:
    """The height with the most level surface area: the floor, in a single-storey home."""
    cross = _area_normals(vertices, faces)
    area = np.linalg.norm(cross, axis=1) / 2
    level = np.abs(cross[:, 2]) > 0.99 * 2 * area
    heights = vertices[faces[level]][:, :, 2].mean(axis=1)
    bins = np.round(heights / 0.01).astype(np.int64)
    values, inverse = np.unique(bins, return_inverse=True)
    return float(values[np.argmax(np.bincount(inverse, weights=area[level]))] * 0.01)


def surface_points(
    vertices: NDArray[np.float32], faces: NDArray[np.uint32], spacing: float
) -> list[NDArray[np.float32]]:
    """Points no further than *spacing* apart covering every triangle, in batches."""
    corners = vertices[faces].astype(np.float32)
    edges = np.linalg.norm(corners - np.roll(corners, 1, axis=1), axis=2).max(axis=1)
    steps = np.maximum(1, np.ceil(edges / spacing)).astype(np.int64)
    batches = []
    for n in np.unique(steps):
        i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
        keep = i + j <= n
        u, v = (i[keep] / n).astype(np.float32), (j[keep] / n).astype(np.float32)
        group = corners[steps == n]
        per_batch = max(1, 4_000_000 // len(u))  # a few million points at a time
        for start in range(0, len(group), per_batch):
            a, b, c = (group[start : start + per_batch, k, None] for k in range(3))
            batches.append((a + u[:, None] * (b - a) + v[:, None] * (c - a)).reshape(-1, 3))
    return batches


def voxelize(
    vertices: NDArray[np.float32], faces: NDArray[np.uint32], floor_z: float
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """Voxels any surface passes through, from just above the floor up; and the grid's corner."""
    low = vertices.min(axis=0).astype(np.float64)
    origin = np.array([low[0], low[1], floor_z + FLOOR_CLEARANCE])
    shape = np.ceil((vertices.max(axis=0) - origin) / VOXEL).astype(int) + 1
    occupied = np.zeros(np.maximum(shape, 1), bool)
    for points in surface_points(vertices, faces, VOXEL / 2):
        cell = np.floor((points - origin) / VOXEL).astype(np.int64)
        cell = cell[(cell[:, 2] >= 0) & (cell < shape).all(axis=1) & (cell >= 0).all(axis=1)]
        occupied[cell[:, 0], cell[:, 1], cell[:, 2]] = True
    return occupied, origin


def merged_boxes(occupied: NDArray[np.bool_]) -> NDArray[np.int64]:
    """Occupied voxels merged greedily into boxes, each (low corner, high corner) exclusive."""
    free = occupied.copy()
    boxes = []
    for x, y, z in np.argwhere(occupied):
        if not free[x, y, z]:
            continue
        x1 = x + 1
        while x1 < free.shape[0] and free[x1, y, z]:
            x1 += 1
        y1 = y + 1
        while y1 < free.shape[1] and free[x:x1, y1, z].all():
            y1 += 1
        z1 = z + 1
        while z1 < free.shape[2] and free[x:x1, y:y1, z1].all():
            z1 += 1
        free[x:x1, y:y1, z:z1] = False
        boxes.append((x, y, z, x1, y1, z1))
    return np.array(boxes, np.int64).reshape(-1, 6)


def mjcf(boxes: NDArray[np.int64], origin: NDArray[np.float64], floor_z: float, name: str) -> str:
    low = origin + boxes[:, :3] * VOXEL
    high = origin + boxes[:, 3:] * VOXEL
    center, half = (low + high) / 2, (high - low) / 2
    geoms = "\n".join(
        f'    <geom type="box" pos="{c[0]:.4f} {c[1]:.4f} {c[2]:.4f}" '
        f'size="{h[0]:.4f} {h[1]:.4f} {h[2]:.4f}"/>'
        for c, h in zip(center, half, strict=True)
    )
    return (
        f'<mujoco model="interiorgs-{name}">\n  <worldbody>\n'
        f'    <geom name="floor" type="plane" size="0 0 0.05" pos="0 0 {floor_z:.4f}"/>\n'
        f"{geoms}\n  </worldbody>\n</mujoco>\n"
    )


def spawn_poses(
    vertices: NDArray[np.float32],
    faces: NDArray[np.uint32],
    floor_z: float,
    occupied: NDArray[np.bool_],
    origin: NDArray[np.float64],
) -> list[tuple[float, float, float]]:
    """Open floor, as far from obstacles as possible and SPAWN_SPACING apart, each facing the
    longest clear view; the most open first."""
    from scipy import ndimage

    cross = _area_normals(vertices, faces)
    area2 = np.linalg.norm(cross, axis=1)
    level = cross[:, 2] > 0.99 * area2
    floor = faces[level][np.abs(vertices[faces[level]][:, :, 2].mean(axis=1) - floor_z) < 0.03]
    has_floor = np.zeros(occupied.shape[:2], bool)
    for points in surface_points(vertices, floor, VOXEL / 2):
        cell = np.floor((points[:, :2] - origin[:2]) / VOXEL).astype(np.int64)
        cell = cell[(cell >= 0).all(axis=1) & (cell < has_floor.shape).all(axis=1)]
        has_floor[cell[:, 0], cell[:, 1]] = True
    return open_floor_spawns(ndimage.binary_closing(has_floor, iterations=2), occupied, origin)


def open_floor_spawns(
    has_floor: NDArray[np.bool_], occupied: NDArray[np.bool_], origin: NDArray[np.float64]
) -> list[tuple[float, float, float]]:
    """Spawns on the VOXEL grid: cells with floor (has_floor, x by y) and no occupied voxel
    below ROBOT_HEIGHT, as far from the rest as possible, SPAWN_SPACING apart, each facing the
    longest clear view; the most open first."""
    from scipy import ndimage

    blocked = occupied[:, :, : math.ceil(ROBOT_HEIGHT / VOXEL)].any(axis=2)
    free = has_floor & ~blocked
    clearance = ndimage.distance_transform_edt(free) * VOXEL
    spawns: list[tuple[float, float, float]] = []
    while len(spawns) < MAX_SPAWNS:
        x, y = np.unravel_index(np.argmax(clearance), clearance.shape)
        if clearance[x, y] < SPAWN_CLEARANCE:
            break
        center = origin[:2] + (np.array([x, y]) + 0.5) * VOXEL
        spawns.append((float(center[0]), float(center[1]), _open_heading(free, x, y)))
        xs, ys = np.ogrid[: clearance.shape[0], : clearance.shape[1]]
        clearance[(xs - x) ** 2 + (ys - y) ** 2 <= (SPAWN_SPACING / VOXEL) ** 2] = 0
    return spawns


def _open_heading(free: NDArray[np.bool_], x: int, y: int, headings: int = 32) -> float:
    """The heading from cell (x, y) whose straight line stays on free floor longest."""
    best, best_length = 0.0, -1
    for yaw in np.linspace(-math.pi, math.pi, headings, endpoint=False):
        step = np.array([math.cos(yaw), math.sin(yaw)])
        length = 0
        while True:
            cx, cy = np.floor(np.array([x, y]) + 0.5 + step * (length + 1)).astype(int)
            if not (0 <= cx < free.shape[0] and 0 <= cy < free.shape[1] and free[cx, cy]):
                break
            length += 1
        if length > best_length:
            best, best_length = float(yaw), length
    return best
