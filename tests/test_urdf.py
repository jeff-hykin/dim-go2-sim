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

"""The URDF a controller draws the Go2 and its scene from, and the files it asks for."""

import gzip
import hashlib
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from test_interiorgs import room

from dim_go2_sim.config import UrdfConfig
from dim_go2_sim.splat import Splats, spz
from dim_go2_sim.urdf import (
    MeshFile,
    SceneSource,
    body_in_parent,
    leg_bodies,
    share_urdf,
    stl,
    tiles,
)
from dim_go2_sim.world import go2_spec


def read_spz(data: bytes) -> dict[str, np.ndarray]:
    """SPZ version 2 as Spark reads it, written from the format's description."""
    raw = gzip.decompress(data)
    magic, version, count, degree, fraction, _, _ = struct.unpack_from("<IIIBBBB", raw)
    assert (magic, version) == (0x5053474E, 2)
    offset = 16

    def take(size: int) -> np.ndarray:
        nonlocal offset
        chunk = np.frombuffer(raw, np.uint8, size, offset)
        offset += size
        return chunk.astype(np.float64)

    packed = take(count * 9).astype(np.int64).reshape(count, 3, 3)
    fixed = packed[..., 0] | packed[..., 1] << 8 | packed[..., 2] << 16
    fixed -= (fixed & 0x800000) << 1  # 24-bit two's complement
    alpha = take(count) / 255
    color = (take(count * 3).reshape(count, 3) / 255 - 0.5) / 0.15
    scale = take(count * 3).reshape(count, 3) / 16 - 10
    xyz = take(count * 3).reshape(count, 3) / 127.5 - 1
    w = np.sqrt(np.maximum(0, 1 - (xyz**2).sum(1)))
    rest = (degree + 1) ** 2 - 1
    sh = (take(count * rest * 3).reshape(count, rest, 3) - 128) / 128
    assert offset == len(raw)
    return {
        "positions": fixed / (1 << fraction),
        "opacities": alpha,
        "dc": color,
        "log_scales": scale,
        "rotations": np.column_stack([w, xyz]),
        "sh": sh,
    }


def test_spz_keeps_each_splat_to_its_quantisation() -> None:
    rng = np.random.default_rng(0)
    count = 500
    rotations = Rotation.random(count, random_state=1).as_quat()[:, [3, 0, 1, 2]]
    splats = Splats(
        positions=rng.uniform(-20, 20, (count, 3)).astype(np.float32),
        rotations=rotations.astype(np.float16),
        log_scales=rng.uniform(-6, 0, (count, 3)).astype(np.float16),
        opacities=rng.uniform(0, 1, count).astype(np.float16),
        sh=rng.uniform(-0.9, 0.9, (count, 16, 3)).astype(np.float16),
    )
    back = read_spz(spz(splats, 3))
    assert np.abs(back["positions"] - splats.positions).max() <= 0.5 / 4096 + 1e-6
    assert np.abs(back["opacities"] - splats.opacities).max() <= 0.5 / 255 + 1e-3
    assert np.abs(back["dc"] - splats.sh[:, 0]).max() <= 0.5 / (0.15 * 255) + 1e-3
    assert np.abs(back["log_scales"] - splats.log_scales).max() <= 0.5 / 16 + 1e-2
    same = np.abs((back["rotations"] * rotations.astype(np.float16)).sum(1))  # q and -q agree
    assert same.min() > 0.99
    assert np.abs(back["sh"][:, :3] - splats.sh[:, 1:4]).max() <= 4 / 128 + 1e-2  # 5 bits
    assert np.abs(back["sh"][:, 3:] - splats.sh[:, 4:]).max() <= 8 / 128 + 1e-2  # 4 bits
    assert read_spz(spz(splats, 0))["sh"].shape == (count, 0, 3)


def test_tiles_keep_every_triangle_under_the_limit() -> None:
    vertices, faces = room()
    pieces = tiles(vertices, faces, 5)
    assert all(len(f) <= 5 for _, f in pieces)
    assert sum(len(f) for _, f in pieces) == len(faces)
    corners = sorted(map(tuple, np.concatenate([v[f] for v, f in pieces]).reshape(-1, 9).tolist()))
    assert corners == sorted(map(tuple, vertices[faces].reshape(-1, 9).tolist()))


def test_stl_is_binary_stl() -> None:
    vertices, faces = room()
    data = stl(vertices, faces)
    assert struct.unpack_from("<I", data, 80)[0] == len(faces) and len(data) == 84 + 50 * len(faces)


@pytest.fixture
def shared(tmp_path: Path):
    vertices, faces = room()
    scene = SceneSource("room", "1", lambda: (vertices, faces))
    return share_urdf(go2_spec(False).compile(), "go2/", scene, UrdfConfig(tile_faces=8), tmp_path)


def test_urdf_names_each_mesh_by_its_hash_and_poses_the_legs_by_tf(shared) -> None:
    robot = ET.fromstring(shared.xml)
    links = {link.get("name") for link in robot.iter("link")}
    joints = {j.find("child").get("link"): j for j in robot.iter("joint")}
    # One root, world; every other link hangs off it, the robot's named as the sim's tf frames.
    assert links - set(joints) == {"go2/world"}
    assert {"go2/base_link", "go2/FL_hip", "go2/RR_calf", "go2/scene_mesh"} <= links
    assert joints["go2/base_link"].get("type") == "floating"
    tiles_in_urdf = [m for m in robot.iter("mesh") if "/scenes/room/mesh_" in m.get("filename")]
    assert len(tiles_in_urdf) == 4  # 22 triangles halved until at most 8 a file
    for mesh in robot.iter("mesh"):
        name, _, query = mesh.get("filename").partition("?")
        path, digest = shared.files[name]
        assert name.startswith("package://dim_go2_sim/") and path.suffix == ".stl"
        assert query == f"hash={digest}" == f"hash={hashlib.sha256(path.read_bytes()).hexdigest()}"
    assert len(shared.files) == len(list(robot.iter("mesh")))


def test_the_scene_and_the_people_are_each_a_group_the_robot_in_none(tmp_path: Path) -> None:
    vertices, faces = room()
    scene = SceneSource("room", "1", lambda: (vertices, faces))
    xml = share_urdf(
        go2_spec(False).compile(), "go2/", scene, UrdfConfig(tile_faces=8), tmp_path, people=2
    ).xml
    groups = {link.get("name"): link.findtext("group") for link in ET.fromstring(xml).iter("link")}
    assert {name: group for name, group in groups.items() if group} == {
        "go2/scene_mesh": "scene",
        "go2/person_0": "people",
        "go2/person_1": "people",
    }
    assert groups["go2/base_link"] is None and groups["go2/world"] is None


def test_a_mesh_request_gets_the_file_unless_the_asker_has_it(shared) -> None:
    name, (path, digest) = next(iter(shared.files.items()))
    for request in (name, f"{name}?hash={digest}", f"{name}?hash={digest}&have={'0' * 64}"):
        mesh = MeshFile.lcm_decode(shared.answer(request).lcm_encode())
        assert (mesh.uri, mesh.data) == (f"{name}?hash={digest}", path.read_bytes())
    assert shared.answer(f"{name}?hash={digest}&have={digest}") is None
    with pytest.raises(KeyError):
        shared.answer("package://dim_go2_sim/../../etc/passwd")


def test_the_urdf_poses_each_leg_where_the_sim_has_it(shared) -> None:
    """Forward kinematics from the URDF's joints match the sim's bodies, at any joint angles."""
    model = go2_spec(False).compile()
    data = mujoco.MjData(model)
    data.qpos[7:] = np.random.default_rng(2).uniform(-0.8, 0.8, model.nq - 7)
    mujoco.mj_forward(model, data)
    joints = {j.get("name"): j for j in ET.fromstring(shared.xml).iter("joint")}
    for body, parent in leg_bodies(model, model.body("base").id):
        joint = joints[model.joint(int(model.body_jntadr[body])).name]
        origin = joint.find("origin")
        rpy = [float(v) for v in origin.get("rpy").split()]
        axis = np.array([float(v) for v in joint.find("axis").get("xyz").split()])
        angle = data.qpos[model.jnt_qposadr[model.body_jntadr[body]]]
        rotation = Rotation.from_euler("xyz", rpy) * Rotation.from_rotvec(axis * angle)
        position, sim_rotation = body_in_parent(data, body, parent)
        assert np.allclose([float(v) for v in origin.get("xyz").split()], position, atol=1e-5)
        assert np.allclose(rotation.as_matrix(), sim_rotation, atol=1e-5)
        assert joint.find("parent").get("link").removeprefix("go2/") in (
            model.body(parent).name,
            "base_link",
        )
