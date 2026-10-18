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

"""A real-world Gaussian-splat scan as a scene: a SuperSplat scene by id (downloaded on first use)
or a local 3DGS PLY / SOG. A scan has no collision mesh and an arbitrary frame and scale, so
cooking it:

- levels it: the floor is the densest layer at the bottom of the scan; a plane is fitted to it by
  RANSAC and turned to z = 0, and the walls are turned square to x and y;
- scales it to metres by a factor measured for the scene (SCALES), or given;
- builds what it collides with from the splats themselves: confident (opaque, small) splats are
  voxelised at VOXEL, specks of a few voxels dropped as floaters, walls filled down to the floor
  (their feet are often thin in a scan), and the voxels from just above
  the floor to COLLISION_HEIGHT merged into boxes for MuJoCo, over a floor plane; the lidar casts
  against the voxels' outer faces, at every height, and the floor under the scan.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import urllib.request

import numpy as np
from numpy.typing import NDArray

from dim_go2_sim.interiorgs import (
    COLLISION_HEIGHT,
    VOXEL,
    _download,
    merged_boxes,
    mjcf,
    open_floor_spawns,
)
from dim_go2_sim.splat import Splats, load_splats, save_ply

PREFIX = "supersplat/"
VIEWER_URL = "https://superspl.at/s?id={id}"
# Bump when cooking changes, so cooked scenes are rebuilt.
COOK_VERSION = 1
# A streamed SuperSplat scene keeps levels of detail, each half the last; this one (from the
# finest, 0) is drawn and collided with. The ranch home's is 2.7M splats, 31 MB.
LOD_LEVEL = 3
# Metres per scan unit, measured for each scene (see the README): a scan's scale is arbitrary.
SCALES = {
    # NJ ranch home, by its US-standard fittings, levelled: its kitchen counters (the largest
    # level surface above the seats) are 0.985 units high and stand 36 in (0.914 m): 0.928; its
    # interior door openings are 2.31 units high and an 80 in door's is ~2.05 m: 0.89. The
    # walls, cut off where the ceiling was, reach 2.5 units, under an 8 ft (2.44 m) ceiling at
    # 0.92 as they should be.
    "3f89bbd3": 0.92,
}
# Splats this opaque, and no bigger than this (metres, 1 sigma), are surfaces; the rest
# (haze, floaters, background) are drawn but not collided with.
MIN_OPACITY = 0.5
MAX_SIGMA = 0.15
# A voxel is solid with this many such splats in it; a solid patch smaller than this many voxels
# is a floater.
MIN_SPLATS = 2
MIN_PATCH = 30
WALL_SPAN = 0.6  # metres
# Splats lower than this above the floor are the floor's (the scan's floor is a few cm thick).
FLOOR_CLEARANCE = 0.06


@dataclass(frozen=True)
class ScanScene:
    mjcf: Path
    splats: Path  # a 3DGS PLY, levelled and in metres
    collision: Path  # .npz of the voxels' and floor's "vertices" and "faces", for the lidar
    floor_z: float
    spawns: list[tuple[float, float, float]]  # x, y, yaw; the most open first
    scale: float  # metres per unit of the scan


def scene_id(package: str) -> str | None:
    """The SuperSplat id in a package name like "supersplat/3f89bbd3", else None."""
    return package[len(PREFIX) :] if package.startswith(PREFIX) else None


def _cache(cache_dir: Path | None) -> Path:
    return (cache_dir or Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))) / (
        "dim_go2_sim"
    )


def load_scene(scene: str, cache_dir: Path | None = None, scale: float | None = None) -> ScanScene:
    """SuperSplat scene *scene*, downloaded (~30 MB) and cooked the first time."""
    folder = _cache(cache_dir) / "supersplat" / scene
    scale = scale or SCALES.get(scene, 1.0)

    def splats() -> Splats:
        meta = _download_sog(scene, folder / "sog")
        return _concatenated([load_splats(m) for m in meta])

    return _cooked(folder, f"supersplat-{scene}", scale, splats)


def load_scan(path: Path, cache_dir: Path | None = None, scale: float = 1.0) -> ScanScene:
    """A local scan (a 3DGS PLY or SOG meta.json), cooked the first time."""
    path = Path(path).resolve()
    stat = path.stat()
    key = hashlib.sha256(f"{path} {stat.st_size} {stat.st_mtime_ns}".encode()).hexdigest()[:12]
    folder = _cache(cache_dir) / "scans" / f"{path.stem}-{key}"
    return _cooked(folder, path.stem, scale, lambda: load_splats(path))


def _cooked(folder: Path, name: str, scale: float, splats: object) -> ScanScene:
    info_path = folder / "scene.json"
    info = json.loads(info_path.read_text()) if info_path.is_file() else {}
    if info.get("cook_version") != COOK_VERSION or info.get("scale") != scale:
        folder.mkdir(parents=True, exist_ok=True)
        info = cook(splats(), folder, name, scale)  # type: ignore[operator]
        info_path.write_text(json.dumps(info))
    return ScanScene(
        mjcf=folder / "scene.xml",
        splats=folder / "splats.ply",
        collision=folder / "collision.npz",
        floor_z=0.0,
        spawns=[tuple(s) for s in info["spawns"]],
        scale=scale,
    )


def _download_sog(scene: str, folder: Path) -> list[Path]:
    """The scene's SOG files (each a meta.json beside its images), at LOD_LEVEL if it has levels."""
    done = folder / "files.json"
    if done.is_file():
        return [folder / f for f in json.loads(done.read_text())]
    with urllib.request.urlopen(VIEWER_URL.format(id=scene)) as response:
        page = response.read().decode()
    found = re.search(r'"contentUrl":"([^"]+)"', page)
    if found is None:
        raise RuntimeError(f"SuperSplat scene {scene}: no contentUrl on its viewer page")
    url = found.group(1)
    base = url.rsplit("/", 1)[0] + "/"
    if url.endswith("lod-meta.json"):
        with urllib.request.urlopen(url) as response:
            levels = json.loads(response.read())
        level = min(LOD_LEVEL, levels["lodLevels"] - 1)
        metas = [f for f in levels["filenames"] if f.split("_")[0] == str(level)]
    elif url.endswith("meta.json"):
        metas = [url[len(base) :]]
    else:
        raise RuntimeError(f"SuperSplat scene {scene}: {url} is not a SOG this reads")
    for meta in metas:
        path = folder / meta
        _download(base + meta, path)
        info = json.loads(path.read_text())
        names = [
            n
            for key in ("means", "scales", "quats", "sh0", "shN")
            for n in info.get(key, {}).get("files", [])
        ]
        for name in names:
            _download(base + str(Path(meta).parent / name), path.parent / name)
    done.write_text(json.dumps(metas))
    return [folder / m for m in metas]


def _concatenated(parts: list[Splats]) -> Splats:
    return Splats(
        *(
            np.concatenate([getattr(p, f) for p in parts])
            for f in ("positions", "rotations", "log_scales", "opacities", "sh")
        )
    )


def cook(splats: Splats, folder: Path, name: str, scale: float) -> dict[str, object]:
    """Levels, scales and voxelises *splats*; writes the scene's files into *folder*."""
    rotation, floor = levelling(splats)
    world = transformed(splats, rotation, floor, scale)
    save_ply(world, folder / "splats.ply")
    occupied, origin = voxelize(world)
    boxes = merged_boxes(occupied[:, :, : math.ceil(COLLISION_HEIGHT / VOXEL)])
    (folder / "scene.xml").write_text(mjcf(boxes, origin, 0.0, name))
    has_floor = floor_cells(world, occupied.shape[:2], origin)
    vertices, faces = surface_mesh(occupied, origin, has_floor)
    np.savez(folder / "collision.npz", vertices=vertices, faces=faces)
    spawns = open_floor_spawns(has_floor, occupied, origin)
    return {
        "cook_version": COOK_VERSION,
        "scale": scale,
        "spawns": spawns,
        "rotation": rotation.tolist(),
        "floor": floor,
    }


def _normals(splats: Splats) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Each splat's thinnest axis, and how thin it is next to its middle one."""
    from scipy.spatial.transform import Rotation

    w, x, y, z = splats.rotations.astype(np.float64).T
    matrices = Rotation.from_quat(np.stack([x, y, z, w], axis=1)).as_matrix()
    scales = splats.log_scales.astype(np.float64)
    thinnest = np.argmin(scales, axis=1)
    normals = matrices[np.arange(len(scales)), :, thinnest]
    ordered = np.sort(scales, axis=1)
    return normals, np.exp(ordered[:, 0] - ordered[:, 1])


def levelling(splats: Splats, seed: int = 0) -> tuple[NDArray[np.float64], float]:
    """The rotation (rows: x, y, up) that turns the scan's floor level and its walls square to x
    and y, and the floor's height along up, in scan units."""
    rng = np.random.default_rng(seed)
    solid = np.flatnonzero(splats.opacities.astype(np.float32) > MIN_OPACITY)
    sample = rng.choice(solid, min(len(solid), 400_000), replace=False)
    points = splats.positions[sample].astype(np.float64)
    # A home is far wider than it is tall: up is roughly its thinnest direction, pointing away
    # from the side its furniture crowds (the floor's).
    centered = points - points.mean(axis=0)
    up = np.linalg.eigh(centered.T @ centered)[1][:, 0]
    heights = points @ up
    low, high = np.percentile(heights, [2, 98])
    if np.median(heights) > (low + high) / 2:
        up, heights, low, high = -up, -heights, -high, -low
    extent = high - low
    # The floor: the densest thin layer in the bottom third, fitted by RANSAC.
    bins = np.arange(low - 0.1 * extent, low + extent / 3, extent / 200)
    counts, edges = np.histogram(heights, bins)
    peak = edges[np.argmax(counts)] + extent / 400
    near = points[np.abs(heights - peak) < 0.04 * extent]
    tolerance = 0.006 * extent
    best, best_count = up, -1
    for _ in range(300):
        a, b, c = near[rng.choice(len(near), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        if np.linalg.norm(normal) < 1e-12:
            continue
        normal /= np.linalg.norm(normal)
        normal *= np.sign(normal @ up)
        if normal @ up < math.cos(math.radians(20)):
            continue
        count = int((np.abs((near - a) @ normal) < tolerance).sum())
        if count > best_count:
            best, best_count, anchor = normal, count, a
    inliers = near[np.abs((near - anchor) @ best) < tolerance]
    center = inliers.mean(axis=0)
    normal = np.linalg.svd(inliers - center, full_matrices=False)[2][2]
    up = normal * np.sign(normal @ up)
    floor = float(center @ up)
    # The walls: thin, upright splats; their normals' heading, folded into 90 degrees, peaks at
    # the walls' heading.
    normals, thinness = _normals(splats)
    upright = (thinness[sample] < 0.3) & (np.abs(normals[sample] @ up) < 0.2)
    across = np.cross(up, [1.0, 0, 0] if abs(up[0]) < 0.9 else [0, 1.0, 0])
    across /= np.linalg.norm(across)
    along = np.cross(up, across)
    wall = normals[sample][upright]
    heading = np.arctan2(wall @ along, wall @ across)
    fourfold = np.exp(4j * heading)
    histogram = np.histogram(np.angle(fourfold), 180, (-math.pi, math.pi))
    top = histogram[1][np.argmax(histogram[0])] + math.pi / 180
    close = np.abs(np.angle(fourfold * np.exp(-1j * top))) < math.radians(20)
    yaw = float(np.angle(fourfold[close].mean())) / 4
    x = math.cos(yaw) * across + math.sin(yaw) * along
    return np.stack([x, np.cross(up, x), up]), floor


def transformed(
    splats: Splats, rotation: NDArray[np.float64], floor: float, scale: float
) -> Splats:
    """*splats* turned by *rotation*, the floor at z = 0, centred in x and y, in metres."""
    from scipy.spatial.transform import Rotation

    positions = splats.positions.astype(np.float64) @ rotation.T
    positions[:, 2] -= floor
    solid = splats.opacities.astype(np.float32) > MIN_OPACITY
    low, high = np.percentile(positions[solid, :2], [1, 99], axis=0)
    positions[:, :2] -= (low + high) / 2
    w, x, y, z = splats.rotations.astype(np.float64).T
    turned = (
        Rotation.from_matrix(rotation) * Rotation.from_quat(np.stack([x, y, z, w], 1))
    ).as_quat()
    # Higher spherical harmonics would have to turn with the scene: only the view-independent
    # degree 0 is kept.
    return Splats(
        positions=(positions * scale).astype(np.float32),
        rotations=np.roll(turned, 1, axis=1).astype(np.float16),  # w x y z
        log_scales=(splats.log_scales.astype(np.float32) + math.log(scale)).astype(np.float16),
        opacities=splats.opacities,
        sh=splats.sh[:, :1],
    )


def voxelize(splats: Splats) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """Voxels solid with confident splats, from FLOOR_CLEARANCE up, floaters dropped; and the
    grid's corner."""
    from scipy import ndimage

    sigma = np.exp(splats.log_scales.astype(np.float32).max(axis=1))
    confident = (splats.opacities.astype(np.float32) > MIN_OPACITY) & (sigma < MAX_SIGMA)
    points = splats.positions[confident & (splats.positions[:, 2] > FLOOR_CLEARANCE)]
    low = np.percentile(points, 0.01, axis=0) - VOXEL
    origin = np.array([low[0], low[1], FLOOR_CLEARANCE])
    cell = np.floor((points - origin) / VOXEL).astype(np.int64)
    shape = np.percentile(cell, 99.99, axis=0).astype(np.int64) + 2
    cell = cell[(cell >= 0).all(axis=1) & (cell < shape).all(axis=1)]
    counts = np.zeros(shape, np.int32)
    np.add.at(counts, tuple(cell.T), 1)
    solid = counts >= MIN_SPLATS
    labels, n = ndimage.label(solid, np.ones((3, 3, 3)))
    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    keep = sizes >= MIN_PATCH
    keep[0] = False
    solid = keep[labels]
    # A wall's foot is often thin in a scan (behind furniture, in shadow), so a column solid for
    # WALL_SPAN of the metre below COLLISION_HEIGHT is a wall, solid down to the floor.
    top = math.ceil((COLLISION_HEIGHT - FLOOR_CLEARANCE) / VOXEL)
    band = solid[:, :, max(0, top - round(1.0 / VOXEL)) : top]
    solid[band.sum(axis=2) * VOXEL >= WALL_SPAN, :top] = True
    return solid, origin


def floor_cells(
    splats: Splats, shape: tuple[int, ...], origin: NDArray[np.float64]
) -> NDArray[np.bool_]:
    """Grid columns over the scan's floor, holes under furniture filled."""
    from scipy import ndimage

    on_floor = (np.abs(splats.positions[:, 2]) < FLOOR_CLEARANCE) & (
        splats.opacities.astype(np.float32) > MIN_OPACITY
    )
    cell = np.floor((splats.positions[on_floor, :2] - origin[:2]) / VOXEL).astype(np.int64)
    cell = cell[(cell >= 0).all(axis=1) & (cell < shape).all(axis=1)]
    has_floor = np.zeros(shape, bool)
    has_floor[cell[:, 0], cell[:, 1]] = True
    has_floor = ndimage.binary_closing(has_floor, iterations=2)
    return ndimage.binary_fill_holes(has_floor)


def surface_mesh(
    occupied: NDArray[np.bool_], origin: NDArray[np.float64], has_floor: NDArray[np.bool_]
) -> tuple[NDArray[np.float32], NDArray[np.uint32]]:
    """The solid voxels' outer faces, and a quad of floor (z = 0) over each floor column."""
    quads = []
    padded = np.pad(occupied, 1)
    for axis in range(3):
        for side in (0, 1):
            neighbour = np.roll(padded, -1 if side else 1, axis=axis)[1:-1, 1:-1, 1:-1]
            cells = np.argwhere(occupied & ~neighbour).astype(np.float64)
            # The face's four corners, in the cell's unit cube.
            u, v = [a for a in range(3) if a != axis]
            corners = np.zeros((4, 3))
            corners[:, axis] = side
            corners[:, u] = (0, 1, 1, 0)
            corners[:, v] = (0, 0, 1, 1)
            quads.append(origin + (cells[:, None, :] + corners) * VOXEL)
    cells = np.argwhere(has_floor).astype(np.float64)
    floor = np.zeros((len(cells), 4, 3))
    floor[:, :, :2] = origin[:2] + (cells[:, None, :] + [(0, 0), (1, 0), (1, 1), (0, 1)]) * VOXEL
    quads.append(floor)
    corners = np.concatenate(quads).reshape(-1, 3).astype(np.float32)
    first = np.arange(0, len(corners), 4, dtype=np.uint32)[:, None, None]
    faces = (first + np.array([[0, 1, 2], [0, 2, 3]], np.uint32)).reshape(-1, 3)
    return corners, faces
