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

import itertools
import math
import os
from pathlib import Path

import numpy as np
import pytest

from dim_go2_sim import interiorgs
from dim_go2_sim.camera import SceneCamera
from dim_go2_sim.config import CameraConfig, PhysicsConfig, SceneConfig
from dim_go2_sim.contacts import Contact, ContactSensor
from dim_go2_sim.splat import load_nurec
from dim_go2_sim.world import Go2World, load_world

# A home the real-data tests cook (~190 MB the first time): opt in with DIM_GO2_SIM_DOWNLOAD=1,
# or have it cooked already.
HOME = "839920"


def quad(a: tuple, b: tuple, c: tuple, d: tuple) -> np.ndarray:
    return np.array([[a, b, c], [a, c, d]], np.float32)


def room() -> tuple[np.ndarray, np.ndarray]:
    """A 6 x 3 m room with 2.5 m walls, as open surfaces, and a 0.6 m cube in one corner."""
    h = 2.5
    triangles = [
        quad((0, 0, 0), (6, 0, 0), (6, 3, 0), (0, 3, 0)),  # floor
        quad((0, 0, 0), (6, 0, 0), (6, 0, h), (0, 0, h)),
        quad((0, 3, 0), (6, 3, 0), (6, 3, h), (0, 3, h)),
        quad((0, 0, 0), (0, 3, 0), (0, 3, h), (0, 0, h)),
        quad((6, 0, 0), (6, 3, 0), (6, 3, h), (6, 0, h)),
    ]
    cube = np.array(list(itertools.product((0.0, 0.6), (2.2, 2.8), (5.2, 5.8))))[:, ::-1]
    for face in (
        (0, 1, 3, 2),
        (4, 5, 7, 6),
        (0, 1, 5, 4),
        (2, 3, 7, 6),
        (0, 2, 6, 4),
        (1, 3, 7, 5),
    ):
        triangles.append(quad(*(tuple(cube[i]) for i in face)))
    soup = np.concatenate(triangles).reshape(-1, 3)
    return soup, np.arange(len(soup), dtype=np.uint32).reshape(-1, 3)


def test_boxes_cover_exactly_the_voxels_a_surface_passes_through() -> None:
    vertices, faces = room()
    occupied, origin = interiorgs.voxelize(
        vertices, faces, interiorgs.floor_height(vertices, faces)
    )
    assert interiorgs.floor_height(vertices, faces) == 0.0
    assert origin[2] == pytest.approx(interiorgs.FLOOR_CLEARANCE)
    covered = np.zeros(occupied.shape, int)
    for x0, y0, z0, x1, y1, z1 in interiorgs.merged_boxes(occupied):
        covered[x0:x1, y0:y1, z0:z1] += 1
    assert (covered == occupied).all(), "the boxes overlap, or miss or add a voxel"
    assert not occupied[:, :, 0][20:100, 20:40].any(), "the floor became an obstacle"


def test_a_spawn_stands_in_the_open_facing_down_the_room() -> None:
    vertices, faces = room()
    occupied, origin = interiorgs.voxelize(vertices, faces, 0.0)
    spawns = interiorgs.spawn_poses(vertices, faces, 0.0, occupied, origin)
    x, y, yaw = spawns[0]
    assert 0.5 <= x <= 5.2 - 0.5 and 1.4 <= y <= 1.6, "not as far from the walls as possible"
    assert abs(math.cos(yaw)) > 0.9, "facing a side wall, not down the room"
    for x, y, _ in spawns:
        assert math.dist((x, y), (5.5, 2.5)) > 0.3 + interiorgs.SPAWN_CLEARANCE
        assert min(x, 6 - x, y, 3 - y) >= interiorgs.SPAWN_CLEARANCE


@pytest.fixture(scope="module")
def home() -> interiorgs.InteriorScene:
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    cooked = (cache / "dim_go2_sim" / "interiorgs" / HOME / "scene.json").is_file()
    if not (cooked or os.environ.get("DIM_GO2_SIM_DOWNLOAD")):
        pytest.skip(f"InteriorGS {HOME} isn't cooked; set DIM_GO2_SIM_DOWNLOAD=1 to fetch it")
    return interiorgs.load_scene(HOME)


def test_the_robot_stands_at_every_spawn_of_a_real_home(home: interiorgs.InteriorScene) -> None:
    physics = PhysicsConfig(realtime=False)
    world = Go2World(load_world(home.mjcf, physics), physics)
    robot = {b for b in range(world.model.nbody) if world.is_robot(b)}
    assert len(home.spawns) >= 4
    for x, y, yaw in home.spawns:
        world.reset(x, y, yaw, home.floor_z)
        sensor = ContactSensor(
            world.model, world.data, robot, world.trunk, world.feet, home.floor_z + 0.05
        )
        for _ in range(50):
            world.tick(np.zeros(3))
        position, _ = world.base_pose()
        assert sensor.contacts() == [Contact("foot", "floor")], (x, y)
        assert position[2] - home.floor_z > 0.25, f"fell over at {(x, y)}"


def test_a_package_name_resolves_to_the_cooked_home(home: interiorgs.InteriorScene) -> None:
    from dim_go2_sim.module import resolve_scene

    scene = resolve_scene(SceneConfig(package=f"interiorgs/{HOME}"))
    assert (scene.mjcf, scene.visual, scene.y_up) == (home.mjcf, home.splats, False)
    assert scene.spawns[0] == home.spawns[0] and scene.lidar_mesh is not None


def test_the_splats_lie_on_the_collision_surfaces(home: interiorgs.InteriorScene) -> None:
    # The camera's splats and the lidar's meshes must share a frame and a scale.
    import open3d as o3d

    splats = load_nurec(home.splats)
    opaque = np.flatnonzero(splats.opacities > 0.9)
    sample = splats.positions[np.random.default_rng(0).choice(opaque, 20_000, replace=False)]
    mesh = np.load(home.collision)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(mesh["vertices"]), o3d.core.Tensor(mesh["faces"]))
    distance = scene.compute_distance(o3d.core.Tensor(sample)).numpy()
    assert np.median(distance) < 0.03, f"splats sit {np.median(distance):.3f} m off the meshes"


def test_the_camera_sees_a_room_from_a_spawn(home: interiorgs.InteriorScene) -> None:
    x, y, yaw = home.spawns[0]
    camera = SceneCamera(str(home.splats), False, CameraConfig(width=160, height=120))
    try:
        rgb = camera.capture(
            np.array([x, y, home.floor_z + 0.35]),
            np.array([math.cos(yaw), math.sin(yaw), 0.0]),
            np.array([0.0, 0.0, 1.0]),
        )
    finally:
        camera.close()
    background = np.array([0x1B, 0x22, 0x30])
    assert (np.abs(rgb.astype(int) - background).sum(axis=2) > 30).mean() > 0.99, "gaps"
    assert rgb.std() > 20, "a blur, not a room"
