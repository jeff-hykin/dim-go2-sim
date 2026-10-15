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

import json
import math
import pickle
import subprocess
import sys
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dim_go2_sim.config import CameraConfig

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
        self._scene = gfx.Scene()
        background = gfx.Background(None, gfx.BackgroundMaterial(config.background))
        self._scene.add(background, _merged(gfx, root))
        self._scene.add(gfx.AmbientLight("#ffffff", config.ambient))
        sun = gfx.DirectionalLight("#fff4e6", config.sun)
        sun.local.position = (4.0, -6.0, 10.0)
        self._scene.add(sun)
        canvas = OffscreenRenderCanvas(size=(config.width, config.height), pixel_ratio=1.0)
        self._renderer = gfx.WgpuRenderer(canvas, pixel_ratio=1.0)
        # pygfx's fov spans the mean of the view's height and width; match the vertical fovy.
        aspect = config.width / config.height
        fov = math.degrees(2 * math.atan(config.height * (1 + aspect) / (4 * focal_length(config))))
        self._camera = gfx.PerspectiveCamera(
            fov=fov, aspect=aspect, depth_range=(config.near, config.far)
        )

    def capture(
        self, position: NDArray[np.float64], forward: NDArray[np.float64], up: NDArray[np.float64]
    ) -> NDArray[np.uint8]:
        self._camera.world.position = tuple(position)
        self._camera.world.reference_up = tuple(up)
        self._camera.look_at(tuple(position + forward))
        self._renderer.render(self._scene, self._camera)
        self._renderer.flush()
        return np.ascontiguousarray(np.asarray(self._renderer.snapshot())[:, :, :3])


def _merged(gfx: Any, root: Any) -> Any:
    """One mesh per material, in world coordinates: a store is ~47k nodes, one draw call each."""
    meshes: list[Any] = []
    root.traverse(lambda node: meshes.append(node) if isinstance(node, gfx.Mesh) else None)
    by_material: dict[int, list[Any]] = {}
    for mesh in meshes:
        by_material.setdefault(id(mesh.material), []).append(mesh)
    group = gfx.Group()
    for same in by_material.values():
        positions, normals, texcoords, indices, offset = [], [], [], [], 0
        for mesh in same:
            geometry, matrix = mesh.geometry, mesh.world.matrix
            local = geometry.positions.data.reshape(-1, 3).astype(np.float64)
            positions.append((local @ matrix[:3, :3].T + matrix[:3, 3]).astype(np.float32))
            normal = getattr(geometry, "normals", None)
            if normal is not None:
                rotated = normal.data.reshape(-1, 3) @ np.linalg.inv(matrix[:3, :3])
                rotated /= np.maximum(np.linalg.norm(rotated, axis=1, keepdims=True), 1e-12)
                normals.append(rotated.astype(np.float32))
            uv = getattr(geometry, "texcoords", None)
            if uv is not None:
                texcoords.append(uv.data.reshape(-1, 2).astype(np.float32))
            index = getattr(geometry, "indices", None)
            index = np.arange(len(local)) if index is None else index.data.reshape(-1)
            indices.append(index.astype(np.uint32) + offset)
            offset += len(local)
        attributes: dict[str, NDArray[Any]] = {
            "positions": np.concatenate(positions),
            "indices": np.concatenate(indices).reshape(-1, 3),
        }
        # A partial attribute can't be merged; drop it rather than misalign vertices.
        if len(normals) == len(same):
            attributes["normals"] = np.concatenate(normals)
        if len(texcoords) == len(same):
            attributes["texcoords"] = np.concatenate(texcoords)
        group.add(gfx.Mesh(gfx.Geometry(**attributes), same[0].material))
    return group


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
