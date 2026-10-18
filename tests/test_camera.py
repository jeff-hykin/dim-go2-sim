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


import json
from pathlib import Path
import struct

import numpy as np

from dim_go2_sim.camera import SceneCamera, _cap_textures, _downsampled
from dim_go2_sim.config import CameraConfig


def write_quads_glb(path: Path, quads: list[tuple[float, float, list[float]]]) -> None:
    """Squares facing the origin across the world's x axis: (x, half width, RGBA) each."""
    blob, nodes, meshes, materials, views, accessors = b"", [], [], [], [], []
    for x, half, rgba in quads:
        # World (x, y, z) is glTF (x, z, -y): Y-up.
        corners = np.array(
            [[x, -half, half], [x, half, half], [x, half, -half], [x, -half, -half]], np.float32
        )
        normals = np.tile(np.array([-1.0, 0.0, 0.0], np.float32), (4, 1))  # toward the origin
        faces = np.array([[0, 1, 2], [0, 2, 3]], np.uint16)
        arrays = ((corners, "VEC3", 5126), (normals, "VEC3", 5126), (faces, "SCALAR", 5123))
        for array, kind, component in arrays:
            views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": array.nbytes})
            accessor = {
                "bufferView": len(views) - 1,
                "componentType": component,
                "count": len(array) if kind == "VEC3" else array.size,
                "type": kind,
            }
            if kind == "VEC3":
                accessor |= {"min": array.min(0).tolist(), "max": array.max(0).tolist()}
            accessors.append(accessor)
            blob += array.tobytes() + b"\0" * (-array.nbytes % 4)
        i = len(meshes)
        attributes = {"POSITION": 3 * i, "NORMAL": 3 * i + 1}
        primitive = {"attributes": attributes, "indices": 3 * i + 2, "material": i}
        meshes.append({"primitives": [primitive]})
        pbr = {"baseColorFactor": rgba, "metallicFactor": 0.0}  # metal shows only reflections
        materials.append({"pbrMetallicRoughness": pbr, "doubleSided": True})
        nodes.append({"mesh": i})
    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "materials": materials,
        "meshes": meshes,
        "buffers": [{"byteLength": len(blob)}],
        "bufferViews": views,
        "accessors": accessors,
    }
    text = json.dumps(gltf).encode()
    text += b" " * (-len(text) % 4)
    chunks = struct.pack("<I4s", len(text), b"JSON") + text
    chunks += struct.pack("<I4s", len(blob), b"BIN\0") + blob
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, 12 + len(chunks)) + chunks)


def test_a_sign_a_millimetre_off_a_far_wall_is_drawn_in_front_of_it(tmp_path: Path) -> None:
    # A store's signs sit a millimetre or two off its walls; seen from across the store, the
    # wall must not win the depth test.
    visual = tmp_path / "wall.glb"
    # One big triangle pair each, as a store's walls and signs are; seen 50 m off, at 37 degrees.
    write_quads_glb(visual, [(40.0, 60.0, [1, 0, 0, 1]), (39.9984, 20.0, [0, 1, 0, 1])])
    camera = SceneCamera(str(visual), True, CameraConfig(width=64, height=48, ambient=1.0))
    position = np.array([0.0, 30.0, 0.0])
    forward = np.array([40.0, 0.0, 0.0]) - position
    try:
        rgb = camera.capture(position, forward / 50.0, np.array([0.0, 0.0, 1.0]))
    finally:
        camera.close()
    sign = rgb[20:28, 28:36].reshape(-1, 3).astype(int)  # the middle of the frame: all sign
    assert (sign[:, 1] > sign[:, 0]).all(), "the wall shows through the sign"


def test_a_texture_over_the_cap_is_averaged_down_once_and_stays_shared() -> None:
    import pygfx as gfx  # type: ignore[import-untyped]

    tall = np.arange(8 * 4 * 4, dtype=np.uint8).reshape(8, 4, 4)
    shared, small = gfx.Texture(tall, dim=2), gfx.Texture(np.zeros((2, 2, 4), np.uint8), dim=2)
    root = gfx.Group()
    for texture in (shared, shared):
        root.add(gfx.Mesh(gfx.box_geometry(), gfx.MeshBasicMaterial(map=gfx.TextureMap(texture))))
    standard = gfx.MeshStandardMaterial(
        map=gfx.TextureMap(small), normal_map=gfx.TextureMap(shared)
    )
    root.add(gfx.Mesh(gfx.box_geometry(), standard))

    _cap_textures(gfx, root, 4)

    first, second, third = (mesh.material for mesh in root.children)
    assert first.map.texture is second.map.texture is third.normal_map.texture
    assert third.map.texture is small  # already fits: drawn as it was
    # Only the rows were over the cap; each pair of rows is averaged.
    expected = np.rint(tall.reshape(4, 2, 4, 4).mean(axis=1)).astype(np.uint8)
    np.testing.assert_array_equal(first.map.texture.data, expected)


def test_downsampling_halves_each_side_until_it_fits_and_drops_an_odd_last_row() -> None:
    image = np.arange(5 * 9, dtype=np.float32).reshape(5, 9)
    small = _downsampled(image, 2)
    # 5 rows: one halving, the fifth dropped; 9 columns: two halvings, the ninth dropped.
    expected = image[:4, :8].reshape(2, 2, 2, 4).mean(axis=(1, 3))
    np.testing.assert_allclose(small, expected)
    assert _downsampled(image, 9) is image
