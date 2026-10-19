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

from itertools import pairwise
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from dim_go2_sim.camera import SceneCamera
from dim_go2_sim.config import CameraConfig, PeopleConfig, PhysicsConfig, UrdfConfig
from dim_go2_sim.contacts import ContactSensor
from dim_go2_sim.lidar import Raycaster
from dim_go2_sim.people import CLEARANCE, SHIRTS, People, plan_routes
from dim_go2_sim.splat import Splats, save_ply
from dim_go2_sim.urdf import share_urdf
from dim_go2_sim.world import Go2World, compose, go2_spec, load_world

# A 6 x 4 m room, its walls 2 m tall, with a 1 m cube in it.
ROOM = """<mujoco><worldbody>
  <geom type="plane" size="0 0 0.05"/>
  <geom type="box" pos="3 -0.05 1" size="3.1 0.05 1"/>
  <geom type="box" pos="3 4.05 1" size="3.1 0.05 1"/>
  <geom type="box" pos="-0.05 2 1" size="0.05 2 1"/>
  <geom type="box" pos="6.05 2 1" size="0.05 2 1"/>
  <geom type="box" pos="3 2 0.5" size="0.5 0.5 0.5"/>
</worldbody></mujoco>"""


@pytest.fixture
def physics(tmp_path: Path) -> PhysicsConfig:
    return PhysicsConfig(cache_dir=tmp_path / "cache", realtime=False)


@pytest.fixture
def room(tmp_path: Path) -> Path:
    path = tmp_path / "room.xml"
    path.write_text(ROOM)
    return path


def world_with(
    physics: PhysicsConfig, scene: Path | None, routes: list, start=(0.0, 0.0)
) -> Go2World:
    world = Go2World(load_world(scene, physics, len(routes)), physics)
    world.people = People(routes, PeopleConfig(count=len(routes)), start)
    world.reset(*start, 0.0, 0.0)
    return world


def test_a_person_waits_while_the_robot_is_in_their_way_then_turns_back() -> None:
    people = People([[(0, 0), (4, 0)]], PeopleConfig(count=1, speed=0.4), robot_start=(-10, 0))
    far = np.array([-10.0, 0.0])
    for _ in range(250):  # 5 s
        people.step(0.02, far)
    x, y, yaw = people.poses()[0]
    assert (x, y) == pytest.approx((2.0, 0.0), abs=0.01) and yaw == pytest.approx(0.0)
    for _ in range(100):  # 2 s with the robot just ahead of them, side-on, its nose in the way
        people.step(0.02, np.array([x + 0.8, 0.55]))
    assert people.poses()[0][0] == pytest.approx(x) and people.waiting == [True]
    for _ in range(100):  # out of patience at 3 s: they turn back
        people.step(0.02, np.array([x + 0.8, 0.55]))
    assert people.waiting == [False] and people.poses()[0][0] < x
    for _ in range(100):
        people.step(0.02, np.array([x + 0.8, 0.55]))
    x, _, yaw = people.poses()[0]
    assert x == pytest.approx(2.0 - 0.4 * 3.0, abs=0.05) and abs(yaw) > 3.0  # walking away


def test_planned_loops_keep_clear_of_the_walls_and_the_cube(
    physics: PhysicsConfig, room: Path
) -> None:
    model = load_world(room, physics)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    robot = {b for b in range(model.nbody) if model.body(b).name.startswith("go2/")}
    raycaster = Raycaster(model, data, robot, None, 0)
    routes = plan_routes(raycaster.cast_static, raycaster.bounds, (1.0, 1.0), 2, seed=3)
    assert len(routes) == 2 and routes[0] != routes[1]
    for route in routes:
        loop = np.array(route + route[:1])
        walked = np.concatenate([np.linspace(a, b, 50) for a, b in pairwise(loop)])
        margin = CLEARANCE - 0.1  # the grid's half step, either way
        assert np.all((walked > margin) & (walked < np.array([6, 4]) - margin)), "in a wall"
        assert np.all(np.abs(walked - [3, 2]).max(axis=1) > 0.5 + margin), "in the cube"
        assert sum(np.linalg.norm(np.diff(loop, axis=0), axis=1)) > 4.0


def test_the_robot_walking_into_a_person_is_a_person_contact(physics: PhysicsConfig) -> None:
    # A person pacing 10 cm across the robot's way, 1.2 m ahead.
    world = world_with(physics, None, [[(1.2, -0.05), (1.2, 0.05)]])
    model = world.model
    robot = {b for b in range(model.nbody) if world.is_robot(b)}
    sensor = ContactSensor(model, world.data, robot, world.trunk, world.feet, 0.05)
    touched = set()
    for _ in range(200):
        world.tick(np.array([0.5, 0.0, 0.0]))
        touched |= {c.kind for c in sensor.contacts()}
    assert "person" in touched
    position, _ = world.base_pose()
    assert position[0] < 1.0, "walked through them"


@pytest.mark.parametrize("self_occlusion", [True, False])
def test_the_lidar_sees_a_person(physics: PhysicsConfig, self_occlusion: bool) -> None:
    world = world_with(physics, None, [[(2.0, 0.0), (2.02, 0.0)]], start=(-1.0, 0.0))
    model = world.model
    robot = {b for b in range(model.nbody) if world.is_robot(b)}
    raycaster = Raycaster(
        model,
        world.data,
        robot,
        compose(None, physics, ground=False, people=1),
        int(model.jnt_qposadr[model.body_jntadr[world.trunk]]),
        robot_visible=self_occlusion,
    )
    origin = np.array([0.0, 0.0, 1.15])  # torso height, 2 m from them
    ranges, _ = raycaster.cast(origin, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), 10.0)
    assert ranges[0] == pytest.approx(2.0 - 0.11, abs=0.02)  # the torso's front
    assert ranges[1] < 0


def wall_of_splats(path: Path) -> None:
    """A grey wall of opaque splats across x = 3, 4 m wide and 2 m tall."""
    y, z = np.meshgrid(np.arange(-2, 2, 0.02), np.arange(0, 2, 0.02))
    count = y.size
    save_ply(
        Splats(
            positions=np.column_stack([np.full(count, 3.0), y.ravel(), z.ravel()]).astype(
                np.float32
            ),
            rotations=np.tile(np.array([1, 0, 0, 0], np.float16), (count, 1)),
            log_scales=np.full((count, 3), math.log(0.015), np.float16),
            opacities=np.full(count, 0.99, np.float16),
            sh=np.zeros((count, 1, 3), np.float16),  # grey: 0.5
        ),
        path,
    )


def test_the_camera_draws_a_person_in_front_of_a_wall_and_not_behind_it(tmp_path: Path) -> None:
    wall_of_splats(tmp_path / "wall.ply")
    config = CameraConfig(width=80, height=60, background="#000000")
    camera = SceneCamera(str(tmp_path / "wall.ply"), False, config, people=1)
    try:

        def center(x: float) -> np.ndarray:
            eye, forward, up = (
                np.array([0.0, 0, 1.15]),
                np.array([1.0, 0, 0]),
                np.array([0, 0, 1.0]),
            )
            rgb = camera.capture(eye, forward, up, np.array([[x, 0.0, math.pi]]))
            return rgb[25:35, 37:43].reshape(-1, 3).mean(axis=0) / 255

        shirt = np.array(SHIRTS[0])
        in_front, behind = center(1.5), center(4.5)
    finally:
        camera.close()
    assert np.abs(in_front - shirt).max() < 0.25, f"{in_front} isn't the shirt"
    assert np.abs(behind - 0.5).max() < 0.05, f"{behind} isn't the grey wall"


def test_the_urdf_carries_each_person(tmp_path: Path) -> None:
    robot = go2_spec(False).compile()
    shared = share_urdf(robot, "", None, UrdfConfig(), tmp_path, people=2)
    xml = ET.fromstring(shared.xml)
    links = {link.get("name") for link in xml.iter("link")}
    assert {"person_0", "person_1"} <= links
    joints = {j.get("name"): j for j in xml.iter("joint")}
    assert joints["person_1_joint"].get("type") == "floating"
    assert joints["person_1_joint"].find("parent").get("link") == "world"
    names = [m.get("filename").partition("?")[0] for m in xml.iter("mesh")]
    people = {n for n in names if "/people/" in n}
    assert len(people) == 3 and all(shared.answer(n).data for n in people)
