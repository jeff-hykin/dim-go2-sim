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

"""Camera frames drawn from a scene's textured glTF with pygfx (after PimSim's mesh renderer).

Cooked scenes' MJCFs are collision-only, so MuJoCo would draw grey boxes. Only the static scene
is drawn, not the robot: a camera must not see its own body. Rendering runs in a fresh child
process, since GPU device state cannot cross dimOS's forkserver.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import pickle
import subprocess
import sys
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dim_go2_sim.config import CameraConfig

# Most vertices in one merged mesh: 12 bytes each keeps a buffer far under 128 MB.
MAX_VERTICES = 2_000_000
# Triangles simplified away on a software GPU (e.g. lavapipe in Docker), and the fewest a merged
# mesh needs before it is simplified at all.
SOFTWARE_REDUCTION = 0.95
# How near small objects must be to be drawn, metres, on a real GPU (all of them: the image is
# exact) and on a software one.
DETAIL_DISTANCE, SOFTWARE_DETAIL_DISTANCE = math.inf, 4.0
MIN_SIMPLIFY_TRIANGLES = 200
# Merged meshes are cut into square cells this wide, metres, so far ones can be skipped.
CHUNK_CELL = 4.0
# Objects smaller than this bounding radius, metres, are detail (products, signs).
SMALL_RADIUS = 0.3
# A sign is modelled a millimetre or two in front of its wall; seen from across a store, the wall
# must still lose the depth test everywhere. That takes two things:
# - Depth resolution. Each frame is drawn in two slices, [near, DEPTH_SPLIT] then [DEPTH_SPLIT,
#   far], metres, each with the whole depth buffer: one slice from 5 cm resolves ~1 mm at 30 m;
#   from 4 m, ~0.03 mm at 42 m.
# - Small triangles. The GPU interpolates depth across a triangle, and across one many metres wide
#   seen at an angle that is off by millimetres; triangles are split until no edge is longer than
#   MAX_EDGE, metres.
DEPTH_SPLIT = 4.0
MAX_EDGE = 4.0
# glTF is Y-up; the world is Z-up.
Y_UP_TO_Z_UP = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)


def focal_length(config: CameraConfig) -> float:
    return config.height / (2 * math.tan(math.radians(config.fovy_deg) / 2))


class _Renderer:
    def __init__(self, visual: str, y_up: bool, config: CameraConfig) -> None:
        import pygfx as gfx  # type: ignore[import-untyped]
        from rendercanvas.offscreen import OffscreenRenderCanvas  # type: ignore[import-untyped]

        root = gfx.Group()
        root.add(gfx.load_gltf(visual, quiet=True, remote_ok=False).scene)
        if y_up:
            root.local.matrix = Y_UP_TO_Z_UP
        if config.max_texture_size is not None:
            _cap_textures(gfx, root, config.max_texture_size)
        self._config = config
        software = gfx.renderers.wgpu.get_shared().adapter.info["adapter_type"] == "CPU"
        reduction = config.simplify
        if reduction is None:
            reduction = SOFTWARE_REDUCTION if software else 0.0
        detail = config.detail_distance
        if detail is None:
            detail = SOFTWARE_DETAIL_DISTANCE if software else DETAIL_DISTANCE
        self._chunks = _chunks(gfx, root, reduction)
        self._centers = np.array([c.center for c in self._chunks]).reshape(-1, 3)
        self._extents = np.array([c.extent for c in self._chunks])
        self._reach = np.array([detail if c.small else config.far for c in self._chunks])
        self._scene = gfx.Scene()
        self._background = gfx.Background(None, gfx.BackgroundMaterial(config.background))
        self._scene.add(self._background, *[c.mesh for c in self._chunks])
        self._scene.add(gfx.AmbientLight("#ffffff", config.ambient))
        sun = gfx.DirectionalLight("#fff4e6", config.sun)
        sun.local.position = (4.0, -6.0, 10.0)
        self._scene.add(sun)
        canvas = OffscreenRenderCanvas(size=(config.width, config.height), pixel_ratio=1.0)
        self._renderer = gfx.WgpuRenderer(canvas, pixel_ratio=1.0)
        # pygfx's fov spans the mean of the view's height and width; match the vertical fovy.
        aspect = config.width / config.height
        fov = math.degrees(2 * math.atan(config.height * (1 + aspect) / (4 * focal_length(config))))
        split = min(max(DEPTH_SPLIT, config.near), config.far)
        self._slices = [
            (gfx.PerspectiveCamera(fov=fov, aspect=aspect, depth_range=(split, config.far)), split),
            (gfx.PerspectiveCamera(fov=fov, aspect=aspect, depth_range=(config.near, split)), 0.0),
        ]

    def capture(
        self, position: NDArray[np.float64], forward: NDArray[np.float64], up: NDArray[np.float64]
    ) -> NDArray[np.uint8]:
        # Skip chunks wholly behind the camera, which can't change the image; on a software GPU
        # also small ones out of reach, as a store's tens of millions of vertices are too many.
        offset = self._centers - position
        along = offset @ forward  # depth of each chunk's centre, as the slices measure it
        distance = np.linalg.norm(offset, axis=1) - self._extents
        visible = (distance < self._reach) & (along > -self._extents)
        # The far slice first; the near one then draws over it with a fresh depth buffer.
        for i, (camera, start) in enumerate(self._slices):
            reaches = visible & (along + self._extents > start)
            if i:
                reaches &= along - self._extents < self._slices[0][1]
                self._renderer.clear(depth=True)
            for chunk, show in zip(self._chunks, reaches, strict=True):
                chunk.mesh.visible = bool(show)
            self._background.visible = not i
            camera.world.position = tuple(position)
            camera.world.reference_up = tuple(up)
            camera.look_at(tuple(position + forward))
            self._renderer.render(self._scene, camera, clear=not i, flush=False)
        self._renderer.flush()
        return np.ascontiguousarray(np.asarray(self._renderer.snapshot())[:, :, :3])


def _cap_textures(gfx: Any, root: Any, max_size: int) -> None:
    """Every texture map under *root* drawn from a copy no wider or taller than *max_size*."""
    capped: dict[int, Any] = {}  # by id of the original: materials share textures
    meshes: list[Any] = []
    root.traverse(lambda node: meshes.append(node) if isinstance(node, gfx.Mesh) else None)
    for mesh in meshes:
        material = mesh.material
        for name in dir(material):
            if not name.endswith("map") or name.startswith("_"):
                continue
            texture_map = getattr(material, name, None)
            if not isinstance(texture_map, gfx.TextureMap):
                continue
            texture = texture_map.texture
            if id(texture) not in capped:
                data = np.asarray(texture.data)
                small = _downsampled(data, max_size)
                capped[id(texture)] = texture
                if small is not data:
                    capped[id(texture)] = gfx.Texture(
                        small,
                        dim=texture.dim,
                        colorspace=texture.colorspace,
                        generate_mipmaps=texture._generate_mipmaps,  # glTF's sampler may set it
                    )
            texture_map.texture = capped[id(texture)]


def _downsampled(data: NDArray[Any], max_size: int) -> NDArray[Any]:
    """An image's rows and columns halved, each pair averaged, until neither is over *max_size*.

    *data* itself when it already fits. An odd last row or column is dropped.
    """
    rows, columns = data.shape[:2]
    row_factor = column_factor = 1
    while rows // row_factor > max_size:
        row_factor *= 2
    while columns // column_factor > max_size:
        column_factor *= 2
    if row_factor == column_factor == 1:
        return data
    rows, columns = rows // row_factor, columns // column_factor
    blocks = data[: rows * row_factor, : columns * column_factor].reshape(
        rows, row_factor, columns, column_factor, *data.shape[2:]
    )
    mean = blocks.mean(axis=(1, 3), dtype=np.float32)
    if np.issubdtype(data.dtype, np.integer):
        mean = np.rint(mean)
    return np.ascontiguousarray(mean.astype(data.dtype))


def _simplified(attributes: dict[str, NDArray[Any]], reduction: float) -> dict[str, NDArray[Any]]:
    """Quadric decimation; each kept vertex takes the normal and UV of the nearest original."""
    import fast_simplification  # type: ignore[import-untyped]
    from scipy.spatial import cKDTree

    positions = attributes["positions"].astype(np.float64)
    points, faces = fast_simplification.simplify(
        positions, attributes["indices"].astype(np.int64), target_reduction=reduction
    )
    nearest = cKDTree(positions).query(points)[1]
    result = {
        "positions": points.astype(np.float32),
        "indices": faces.astype(np.uint32).reshape(-1, 3),
    }
    for name in ("normals", "texcoords"):
        if name in attributes:
            result[name] = attributes[name][nearest]
    return result


@dataclass
class _Chunk:
    mesh: Any
    center: NDArray[np.float64]
    extent: float  # bounding radius about center
    small: bool  # made of small objects, drawn only near the camera


def _chunks(gfx: Any, root: Any, reduction: float) -> list[_Chunk]:
    """The scene's meshes merged by material, CHUNK_CELL cell, and size, in world coordinates.

    A store is ~47k nodes, one draw call each; merged, it is a few thousand chunks that can be
    culled by distance. Each stays under MAX_VERTICES, as GPUs cap a buffer (lavapipe: 128 MB).
    With reduction > 0, that share of each chunk's triangles is simplified away.
    """
    meshes: list[Any] = []
    root.traverse(lambda node: meshes.append(node) if isinstance(node, gfx.Mesh) else None)
    groups: dict[tuple[int, int, int, bool], list[tuple[Any, NDArray[np.float64]]]] = {}
    for mesh in meshes:
        matrix = mesh.world.matrix
        local = mesh.geometry.positions.data.reshape(-1, 3).astype(np.float64)
        world = local @ matrix[:3, :3].T + matrix[:3, 3]
        low, high = world.min(axis=0), world.max(axis=0)
        center = (low + high) / 2
        small = float(np.linalg.norm(high - low)) / 2 < SMALL_RADIUS
        cell = np.floor(center[:2] / CHUNK_CELL).astype(int)
        key = (id(mesh.material), int(cell[0]), int(cell[1]), small)
        groups.setdefault(key, []).append((mesh, world))
    chunks: list[_Chunk] = []
    for key, members in groups.items():
        batch: list[tuple[Any, NDArray[np.float64]]] = []
        count = 0
        for member in members:
            if batch and count + len(member[1]) > MAX_VERTICES:
                chunks.append(_merge(gfx, batch, key[3], reduction))
                batch, count = [], 0
            batch.append(member)
            count += len(member[1])
        chunks.append(_merge(gfx, batch, key[3], reduction))
    return chunks


def _merge(
    gfx: Any, members: list[tuple[Any, NDArray[np.float64]]], small: bool, reduction: float
) -> _Chunk:
    """One mesh of meshes that share a material, given their world-frame vertices."""
    positions, normals, texcoords, indices, offset = [], [], [], [], 0
    for mesh, world in members:
        geometry, matrix = mesh.geometry, mesh.world.matrix
        positions.append(world.astype(np.float32))
        normal = getattr(geometry, "normals", None)
        if normal is not None:
            rotated = normal.data.reshape(-1, 3) @ np.linalg.inv(matrix[:3, :3])
            rotated /= np.maximum(np.linalg.norm(rotated, axis=1, keepdims=True), 1e-12)
            normals.append(rotated.astype(np.float32))
        uv = getattr(geometry, "texcoords", None)
        if uv is not None:
            texcoords.append(uv.data.reshape(-1, 2).astype(np.float32))
        index = getattr(geometry, "indices", None)
        index = np.arange(len(world)) if index is None else index.data.reshape(-1)
        indices.append(index.astype(np.uint32) + offset)
        offset += len(world)
    vertices = np.concatenate(positions)
    attributes: dict[str, NDArray[Any]] = {
        "positions": vertices,
        "indices": np.concatenate(indices).reshape(-1, 3),
    }
    # A partial attribute can't be merged; drop it rather than misalign vertices.
    if len(normals) == len(members):
        attributes["normals"] = np.concatenate(normals)
    if len(texcoords) == len(members):
        attributes["texcoords"] = np.concatenate(texcoords)
    if reduction > 0 and len(attributes["indices"]) >= MIN_SIMPLIFY_TRIANGLES:
        attributes = _simplified(attributes, reduction)
    attributes = _subdivided(attributes, MAX_EDGE)
    vertices = attributes["positions"]
    low, high = vertices.min(axis=0).astype(np.float64), vertices.max(axis=0).astype(np.float64)
    return _Chunk(
        mesh=gfx.Mesh(gfx.Geometry(**attributes), members[0][0].material),
        center=(low + high) / 2,
        extent=float(np.linalg.norm(high - low)) / 2,
        small=small,
    )


def _subdivided(attributes: dict[str, NDArray[Any]], max_edge: float) -> dict[str, NDArray[Any]]:
    """The same surface, with every edge longer than *max_edge* split at its midpoint.

    Whether an edge is split depends only on its length, so both triangles sharing it split it
    at the same midpoint and no cracks open.
    """
    attributes = dict(attributes)
    while True:
        positions, triangles = attributes["positions"], attributes["indices"]
        corners = positions[triangles]  # (T, 3 corners, 3)
        # Edge k joins corner k to corner k+1.
        long = np.linalg.norm(corners - np.roll(corners, -1, axis=1), axis=2) > max_edge
        if not long.any():
            return attributes
        split = triangles[long.any(axis=1)]
        pairs = np.stack([split, np.roll(split, -1, axis=1)], axis=2)[long[long.any(axis=1)]]
        pairs = np.unique(np.sort(pairs, axis=1), axis=0)
        midpoint = {(int(a), int(b)): len(positions) + i for i, (a, b) in enumerate(pairs)}
        for name, values in list(attributes.items()):
            if name == "indices":
                continue
            middle = (values[pairs[:, 0]] + values[pairs[:, 1]]) / 2
            if name == "normals":
                middle /= np.maximum(np.linalg.norm(middle, axis=1, keepdims=True), 1e-12)
            attributes[name] = np.concatenate([values, middle.astype(values.dtype)])
        kept = [triangles[~long.any(axis=1)]]
        for a, b, c in split:
            edges = [(a, b), (b, c), (c, a)]
            mids = [midpoint.get((min(u, v), max(u, v))) for u, v in edges]
            kept.append(np.array(_fan(a, b, c, *mids), np.uint32).reshape(-1, 3))
        attributes["indices"] = np.concatenate(kept)


def _fan(a: int, b: int, c: int, ab: int | None, bc: int | None, ca: int | None) -> list[int]:
    """Triangle abc split at the midpoints given (None where an edge stays whole), same winding."""
    if ab is not None and bc is not None and ca is not None:
        return [a, ab, ca, ab, b, bc, ca, bc, c, ab, bc, ca]
    if ab is not None and bc is not None:
        return [a, ab, c, ab, bc, c, ab, b, bc]
    if bc is not None and ca is not None:
        return [b, bc, a, bc, ca, a, bc, c, ca]
    if ca is not None and ab is not None:
        return [c, ca, b, ca, ab, b, ca, a, ab]
    if ab is not None:
        return [a, ab, c, ab, b, c]
    if bc is not None:
        return [b, bc, a, bc, c, a]
    assert ca is not None
    return [c, ca, b, ca, a, b]


def _serve(visual: str, y_up: bool, config: CameraConfig) -> None:
    renderer = _Renderer(visual, y_up, config)
    out = sys.stdout.buffer
    pickle.dump(None, out)
    out.flush()
    while True:
        try:
            request = pickle.load(sys.stdin.buffer)
        except EOFError:  # the parent exited without saying goodbye
            break
        if request is None:
            break
        pickle.dump(renderer.capture(*request), out, protocol=pickle.HIGHEST_PROTOCOL)
        out.flush()


class SceneCamera:
    """Renders *visual* in a child process. Loading a large scene there takes a while."""

    def __init__(self, visual: str, y_up: bool, config: CameraConfig) -> None:
        self._process = subprocess.Popen(
            [sys.executable, "-m", __name__, visual, json.dumps(y_up), config.model_dump_json()],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        assert self._process.stdout is not None
        try:
            pickle.load(self._process.stdout)
        except EOFError:
            raise RuntimeError(
                f"the camera renderer exited loading {visual}; see its stderr"
            ) from None

    def capture(
        self, position: NDArray[np.float64], forward: NDArray[np.float64], up: NDArray[np.float64]
    ) -> NDArray[np.uint8]:
        assert self._process.stdin is not None and self._process.stdout is not None
        pickle.dump((position, forward, up), self._process.stdin)
        self._process.stdin.flush()
        rgb: NDArray[np.uint8] = pickle.load(self._process.stdout)
        return rgb

    def close(self) -> None:
        try:
            assert self._process.stdin is not None
            pickle.dump(None, self._process.stdin)
            self._process.stdin.close()
        except OSError:
            pass
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()


if __name__ == "__main__":
    _serve(sys.argv[1], json.loads(sys.argv[2]), CameraConfig.model_validate_json(sys.argv[3]))
