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

import gzip
from pathlib import Path

import msgpack
import numpy as np

from dim_go2_sim.camera import SceneCamera
from dim_go2_sim.config import CameraConfig

SH_C0 = 0.28209479177387814  # degree-0 basis: colour = SH_C0 * albedo + 0.5


def write_nurec(path: Path, positions: list, scales: list, colors: list) -> None:
    """Round, opaque Gaussians in 3DGRUT's NuRec layout: (x, y, z), sigma metres, RGB."""
    count = len(positions)
    arrays = {
        "positions": np.array(positions, np.float16),
        "rotations": np.tile(np.array([1, 0, 0, 0], np.float16), (count, 1)),
        "scales": np.log(np.array(scales, np.float32))[:, None].repeat(3, 1).astype(np.float16),
        "densities": np.full((count, 1), 8.0, np.float16),  # a logit: opacity ~1
        "features_albedo": ((np.array(colors, np.float32) - 0.5) / SH_C0).astype(np.float16),
        "features_specular": np.zeros((count, 45), np.float16),
    }
    prefix = ".gaussians_nodes.gaussians."
    state = {prefix + k: v.tobytes() for k, v in arrays.items()}
    state |= {f"{prefix}{k}.shape": list(v.shape) for k, v in arrays.items()}
    with gzip.open(path, "wb") as file:
        file.write(msgpack.packb({"nre_data": {"state_dict": state}}))


def capture(visual: Path, forward: tuple[float, float, float]) -> np.ndarray:
    camera = SceneCamera(
        str(visual), False, CameraConfig(width=64, height=48, background="#000000")
    )
    try:
        return camera.capture(np.zeros(3), np.array(forward, float), np.array([0.0, 0.0, 1.0]))
    finally:
        camera.close()


def test_the_nearer_splat_hides_the_one_behind_it(tmp_path: Path) -> None:
    # A big red splat 2 m ahead, a small green one 1 m ahead in front of it, blue off to the left.
    visual = tmp_path / "three.nurec"
    write_nurec(
        visual,
        positions=[(2, 0, 0), (1, 0, 0), (1.9, 0.8, 0)],
        scales=[0.5, 0.08, 0.1],
        colors=[(1, 0, 0), (0, 1, 0), (0, 0, 1)],
    )
    rgb = capture(visual, (1, 0, 0)).astype(int)
    # The focal length is 41.6 px: green is 3.3 px across a sigma, red 10.4.
    center = rgb[23:25, 31:33].reshape(-1, 3)
    assert (center[:, 1] > 200).all() and (center[:, 0] < 50).all(), "green is not in front"
    ring = rgb[24, 24]  # 8 px left of centre: 0.8 sigma out on red, 2.4 on green
    assert ring[0] > 150 and ring[1] < 30, f"red does not surround green: {ring}"
    # +y is to the left when looking along +x with z up: blue is 17.5 px left of centre.
    left = rgb[23:25, 14:16].reshape(-1, 3)
    assert (left[:, 2] > 150).all(), "blue is not to the left"
    assert rgb[:, 40:, 1:].max() < 30, "green or blue drawn right of centre"


def test_nothing_behind_the_camera_is_drawn(tmp_path: Path) -> None:
    visual = tmp_path / "behind.nurec"
    write_nurec(visual, positions=[(-2, 0, 0)], scales=[0.5], colors=[(1, 1, 1)])
    assert capture(visual, (1, 0, 0)).max() == 0
