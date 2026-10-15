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

import math
from pathlib import Path

import numpy as np
import pytest

from dim_go2_sim.config import LidarConfig, PhysicsConfig
from dim_go2_sim.contacts import Contact, ContactSensor
from dim_go2_sim.lidar import Raycaster, RollingLidar, mount_rotation
from dim_go2_sim.mid360 import Mid360
from dim_go2_sim.module import Go2SimConfig
from dim_go2_sim.world import Go2World, compose, load_world

# A floor and one wall across x = 3, 2 m tall.
SCENE = """<mujoco><worldbody>
  <geom type="plane" size="0 0 0.05"/>
  <geom name="wall" type="box" pos="3 0 1" size="0.05 5 1"/>
</worldbody></mujoco>"""


@pytest.fixture
def physics(tmp_path: Path) -> PhysicsConfig:
    return PhysicsConfig(cache_dir=tmp_path / "cache", realtime=False)


@pytest.fixture
def scene(tmp_path: Path) -> Path:
    path = tmp_path / "scene.xml"
    path.write_text(SCENE)
    return path


def walk(world: Go2World, command: tuple[float, float, float], seconds: float) -> None:
    for _ in range(round(seconds / 0.02)):
        world.tick(np.array(command))


def yaw(world: Go2World) -> float:
    _, rotation = world.base_pose()
    return math.atan2(rotation[1, 0], rotation[0, 0])


def test_stands_still_without_drifting(physics: PhysicsConfig) -> None:
    world = Go2World(load_world(None, physics), physics)
    world.reset(1.0, 2.0, 0.5, 0.0)
    walk(world, (0, 0, 0), 2.0)  # the policy's first steps settle it, turning it a few degrees
    settled_position, _ = world.base_pose()
    settled_yaw = yaw(world)
    walk(world, (0, 0, 0), 10.0)
    position, rotation = world.base_pose()
    assert np.hypot(*(position - settled_position)[:2]) < 0.02
    assert abs(yaw(world) - settled_yaw) < math.radians(1)
    assert np.hypot(position[0] - 1.0, position[1] - 2.0) < 0.05
    assert rotation[2, 2] > 0.98  # upright


@pytest.mark.parametrize("command", [(0.1, 0, 0), (0.4, 0, 0), (0, 0, 0.5)])
def test_follows_velocity_commands(
    physics: PhysicsConfig, command: tuple[float, float, float]
) -> None:
    world = Go2World(load_world(None, physics), physics)
    world.reset(0.0, 0.0, 0.0, 0.0)
    walk(world, command, 3.0)  # settle into the command, then measure
    averaged = np.zeros(3)
    for _ in range(100):
        world.tick(np.array(command))
        averaged += world.body_velocity() / 100
    assert averaged == pytest.approx(np.array(command), abs=0.04)


def test_compiled_world_is_cached(physics: PhysicsConfig, scene: Path) -> None:
    first = load_world(scene, physics)
    assert list((physics.cache_dir / "dim_go2_sim").glob("scene-*.mjb"))  # type: ignore[operator]
    assert load_world(scene, physics).ngeom == first.ngeom


def test_contacts_are_feet_on_floor_until_the_robot_walks_into_the_wall(
    physics: PhysicsConfig, scene: Path
) -> None:
    world = Go2World(load_world(scene, physics), physics)
    world.reset(2.2, 0.0, 0.0, 0.0)
    model = world.model
    robot = {b for b in range(model.nbody) if world.is_robot(b)}
    sensor = ContactSensor(model, world.data, robot, world.trunk, world.feet, 0.05)
    assert sensor.contacts() == [Contact("foot", "floor")]
    touched = set()
    for _ in range(150):
        world.tick(np.array([0.5, 0.0, 0.0]))
        touched |= set(sensor.contacts())
    assert any(c.kind == "clutter" for c in touched)


def test_lidar_sees_the_wall_from_a_rolling_scan(physics: PhysicsConfig, scene: Path) -> None:
    world = Go2World(load_world(scene, physics), physics)
    world.reset(0.0, 0.0, 0.0, 0.0)
    model = world.model
    robot = {b for b in range(model.nbody) if world.is_robot(b)}
    raycaster = Raycaster(
        model,
        world.data,
        robot,
        compose(None, physics, ground=False),
        int(model.jnt_qposadr[model.body_jntadr[world.trunk]]),
    )
    config = LidarConfig(noise=False, dropout=False)
    mount = mount_rotation(config.pitch_deg)

    def pose() -> tuple[np.ndarray, np.ndarray]:
        position, rotation = world.base_pose()
        return position + rotation @ np.array(config.xyz), rotation @ mount

    walk(world, (0, 0, 0), 2.0)  # settled, so one pose places every return
    lidar = RollingLidar(config, raycaster, pose)
    scans = []

    def cast() -> None:
        if (scan := lidar.advance(world.time - 2.0)) is not None:
            scans.append(scan)

    while not scans:
        world.tick(np.zeros(3), cast)
    scan = scans[0]
    assert len(scan.points) > 5_000  # rays above the wall miss
    assert np.all(np.diff(scan.offsets) >= 0) and scan.offsets[-1] < 0.1
    origin, rotation = pose()
    world_points = scan.points @ rotation.T + origin
    on_wall = np.abs(world_points[:, 0] - 2.95) < 0.02
    on_floor = np.abs(world_points[:, 2]) < 0.02
    assert on_wall.sum() > 500 and on_floor.sum() > 500
    behind = (world_points[:, 0] >= 2.96) & (np.abs(world_points[:, 1]) < 4.5)
    assert not behind.any()  # nothing through the wall


def test_mid360_scan_is_a_tenth_of_a_second_of_four_laser_groups() -> None:
    scan = Mid360().scan(0.0, 0.1)
    assert len(scan.directions) == 20_000
    assert np.allclose(np.linalg.norm(scan.directions, axis=1), 1)
    assert set(np.unique(scan.lines)) == {0, 1, 2, 3}


def test_config_takes_nested_overrides() -> None:
    config = Go2SimConfig(
        scene={"mjcf": "store.xml", "start": (1, 2, 3)}, camera={"width": 320}, odom_hz=10
    )
    assert config.scene.start == (1, 2, 3)
    assert config.camera.width == 320
    assert config.lidar.rate_hz == 10
