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
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
from scipy.spatial.transform import Rotation

from dim_go2_sim import supersplat
from dim_go2_sim.camera import SceneCamera
from dim_go2_sim.config import CameraConfig, PhysicsConfig, SceneConfig
from dim_go2_sim.contacts import Contact, ContactSensor
from dim_go2_sim.splat import Splats, load_ply, load_sog, save_ply
from dim_go2_sim.world import Go2World, load_world

# The NJ ranch home the real-data tests cook (~31 MB the first time): opt in with
# DIM_GO2_SIM_DOWNLOAD=1, or have it cooked already.
RANCH = "3f89bbd3"


def splats(positions: np.ndarray, normals: np.ndarray, sigma: float = 0.02) -> Splats:
    """Opaque grey discs, *sigma* across and a tenth of that thick, facing *normals*."""
    count = len(positions)
    # Each disc's thin axis (z) turned onto its normal (none face straight down).
    axis = np.cross([0.0, 0.0, 1.0], normals)
    angle = np.arccos(np.clip(normals[:, 2], -1, 1))
    length = np.linalg.norm(axis, axis=1, keepdims=True)
    rotvec = np.where(length > 1e-9, axis / np.maximum(length, 1e-9), 0) * angle[:, None]
    quats = Rotation.from_rotvec(rotvec).as_quat()
    return Splats(
        positions=positions.astype(np.float32),
        rotations=np.roll(quats, 1, axis=1).astype(np.float16),
        log_scales=np.log(np.tile([sigma, sigma, sigma / 10], (count, 1))).astype(np.float16),
        opacities=np.full(count, 0.95, np.float16),
        sh=np.zeros((count, 1, 3), np.float16),
    )


def room_scan() -> Splats:
    """A 6 x 4 m room with 2.4 m walls and a 0.6 m cube near one corner, as surface splats 2 cm
    apart, plus two lone floaters in the air."""
    step = 0.02
    parts = []

    def plane(origin: tuple, u: tuple, v: tuple, normal: tuple) -> None:
        a, b = np.meshgrid(
            np.arange(0, np.linalg.norm(u), step), np.arange(0, np.linalg.norm(v), step)
        )
        unit_u, unit_v = np.array(u) / np.linalg.norm(u), np.array(v) / np.linalg.norm(v)
        points = np.array(origin) + a.reshape(-1, 1) * unit_u + b.reshape(-1, 1) * unit_v
        parts.append((points, np.tile(normal, (len(points), 1))))

    h = 2.4
    plane((0, 0, 0), (6, 0, 0), (0, 4, 0), (0, 0, 1))
    plane((0, 0, 0), (6, 0, 0), (0, 0, h), (0, 1, 0))
    plane((0, 4, 0), (6, 0, 0), (0, 0, h), (0, -1, 0))
    plane((0, 0, 0), (0, 4, 0), (0, 0, h), (1, 0, 0))
    plane((6, 0, 0), (0, 4, 0), (0, 0, h), (-1, 0, 0))
    for corner, u, v, n in (
        ((5.0, 3.0, 0.6), (0.6, 0, 0), (0, 0.6, 0), (0, 0, 1)),
        ((5.0, 3.0, 0), (0.6, 0, 0), (0, 0, 0.6), (0, -1, 0)),
        ((5.0, 3.0, 0), (0, 0.6, 0), (0, 0, 0.6), (-1, 0, 0)),
    ):
        plane(corner, u, v, n)
    parts.append((np.array([[2.0, 2.0, 1.0], [3.5, 1.0, 1.5]]), np.array([[0, 0, 1.0]] * 2)))
    return splats(np.concatenate([p for p, _ in parts]), np.concatenate([n for _, n in parts]))


# The scan's frame: tilted 4 degrees, turned 25 degrees, Y down as 3DGS's often are, in units
# of 1/1.7 m, off somewhere.
SCAN_SCALE = 1.7
SCAN_TURN = Rotation.from_euler("xyz", [-90, 4, 25], degrees=True)


def in_scan_frame(room: Splats) -> Splats:
    positions = SCAN_TURN.apply(room.positions.astype(np.float64)) * SCAN_SCALE + [3, -1, 7]
    w, x, y, z = room.rotations.astype(np.float64).T
    turned = (SCAN_TURN * Rotation.from_quat(np.stack([x, y, z, w], 1))).as_quat()
    return Splats(
        positions=positions.astype(np.float32),
        rotations=np.roll(turned, 1, axis=1).astype(np.float16),
        log_scales=(room.log_scales.astype(np.float32) + math.log(SCAN_SCALE)).astype(np.float16),
        opacities=room.opacities,
        sh=room.sh,
    )


def test_a_ply_reads_back_as_written(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    count = 50
    original = Splats(
        positions=rng.normal(size=(count, 3)).astype(np.float32),
        rotations=rng.normal(size=(count, 4)).astype(np.float16),
        log_scales=rng.normal(-3, 1, (count, 3)).astype(np.float16),
        opacities=rng.uniform(0.01, 0.99, count).astype(np.float16),
        sh=rng.normal(size=(count, 4, 3)).astype(np.float16),  # degree 1
    )
    save_ply(original, tmp_path / "s.ply")
    read = load_ply(tmp_path / "s.ply")
    for field in ("positions", "rotations", "log_scales", "sh"):
        np.testing.assert_allclose(getattr(read, field), getattr(original, field), atol=1e-3)
    np.testing.assert_allclose(read.opacities, original.opacities, atol=2e-3)
    assert read.sh_degree == 1


def test_a_sog_reads_as_splat_transform_writes_it(tmp_path: Path) -> None:
    # Two splats, by SOG version 2's layout: 16-bit log-positions over two images, a smallest-
    # three quaternion, codebook scales and colours, and the opacity in sh0's alpha.
    def image(name: str, pixels: list) -> None:
        Image.fromarray(np.array(pixels, np.uint8).reshape(1, 2, 4), "RGBA").save(
            tmp_path / name, lossless=True
        )

    codebook = np.linspace(-5, 5, 256).tolist()
    # Position 0 is the minimum (log -1 -> -(e - 1)); 1 is the maximum (log 2 -> e^2 - 1).
    image("means_l.webp", [[0, 0, 0, 255], [255, 255, 255, 255]])
    image("means_u.webp", [[0, 0, 0, 255], [255, 255, 255, 255]])
    # Splat 0: the identity (w largest, tag 252); splat 1: 90 degrees about z, w = z = 0.7071,
    # the largest stored as z (tag 255), w kept as 0.7071 * sqrt(2) = 1 -> byte 255.
    image("quats.webp", [[127.5, 127.5, 127.5, 252], [255, 127.5, 127.5, 255]])
    image("scales.webp", [[0, 128, 255, 255], [10, 10, 10, 255]])
    image("sh0.webp", [[255, 0, 128, 255], [128, 128, 128, 51]])
    meta = {
        "version": 2,
        "count": 2,
        "means": {
            "mins": [-1, -1, -1],
            "maxs": [2, 2, 2],
            "files": ["means_l.webp", "means_u.webp"],
        },
        "scales": {"codebook": codebook, "files": ["scales.webp"]},
        "quats": {"files": ["quats.webp"]},
        "sh0": {"codebook": codebook, "files": ["sh0.webp"]},
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    s = load_sog(tmp_path / "meta.json")
    np.testing.assert_allclose(s.positions[0], [1 - math.e] * 3, atol=1e-3)
    np.testing.assert_allclose(s.positions[1], [math.e**2 - 1] * 3, atol=1e-2)
    w, x, y, z = s.rotations.astype(np.float32).T
    np.testing.assert_allclose([w[0], x[0], y[0], z[0]], [1, 0, 0, 0], atol=0.01)
    np.testing.assert_allclose(
        [abs(w[1]), x[1], y[1], abs(z[1])], [0.7071, 0, 0, 0.7071], atol=0.01
    )
    np.testing.assert_allclose(
        s.log_scales[0], [codebook[0], codebook[128], codebook[255]], atol=0.01
    )
    np.testing.assert_allclose(s.sh[0, 0], [codebook[255], codebook[0], codebook[128]], atol=0.01)
    np.testing.assert_allclose(s.opacities, [1.0, 0.2], atol=0.01)


def test_levelling_finds_the_floor_and_squares_the_walls() -> None:
    scan = in_scan_frame(room_scan())
    rotation, floor = supersplat.levelling(scan)
    # Turned back, the room's up is up and its walls face x or y.
    level = rotation @ SCAN_TURN.as_matrix()
    assert level[2] @ [0, 0, 1] > math.cos(math.radians(0.3)), "the floor isn't level"
    heading = math.degrees(math.atan2(level[1, 0], level[0, 0])) % 90
    assert min(heading, 90 - heading) < 0.5, f"the walls are {heading:.1f} degrees off square"
    world = supersplat.transformed(scan, rotation, floor, 1 / SCAN_SCALE)
    assert np.percentile(world.positions[:, 2], 30) == pytest.approx(0.0, abs=0.01)
    assert world.positions[:, 2].max() == pytest.approx(2.4, abs=0.05)


def test_the_room_collides_where_its_surfaces_are(tmp_path: Path) -> None:
    room = room_scan()
    occupied, origin = supersplat.voxelize(room)
    v = supersplat.VOXEL

    def solid(x: float, y: float, z: float) -> bool:
        i, j, k = np.floor((np.array([x, y, z]) - origin) / v).astype(int)
        return bool(occupied[i, j, k])

    assert solid(0.0, 2.0, 1.0) and solid(6.0, 2.0, 0.3), "a wall isn't solid"
    assert solid(5.3, 3.01, 0.3), "the cube isn't solid"
    assert not solid(2.0, 2.0, 1.0) and not solid(3.5, 1.0, 1.5), "a floater is solid"
    assert not occupied[:, :, 0][30:90, 30:60].any(), "the floor became an obstacle"
    info = supersplat.cook(room, tmp_path, "room", 1.0)
    x, y, _ = info["spawns"][0]  # the cooked room is centred, its walls square to x and y
    assert math.hypot(x, y) < 0.6, "not out in the open"
    mesh = np.load(tmp_path / "collision.npz")
    assert mesh["vertices"][:, 2].min() == pytest.approx(0.0), "no floor for the lidar"


@pytest.fixture(scope="module")
def ranch() -> supersplat.ScanScene:
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    cooked = (cache / "dim_go2_sim" / "supersplat" / RANCH / "scene.json").is_file()
    if not (cooked or os.environ.get("DIM_GO2_SIM_DOWNLOAD")):
        pytest.skip(f"SuperSplat {RANCH} isn't cooked; set DIM_GO2_SIM_DOWNLOAD=1 to fetch it")
    return supersplat.load_scene(RANCH)


def test_the_robot_stands_at_every_spawn_of_the_ranch(ranch: supersplat.ScanScene) -> None:
    physics = PhysicsConfig(realtime=False)
    world = Go2World(load_world(ranch.mjcf, physics), physics)
    robot = {b for b in range(world.model.nbody) if world.is_robot(b)}
    assert len(ranch.spawns) >= 4
    for x, y, yaw in ranch.spawns:
        world.reset(x, y, yaw, ranch.floor_z)
        sensor = ContactSensor(world.model, world.data, robot, world.trunk, world.feet, 0.05)
        for _ in range(50):
            world.tick(np.zeros(3))
        position, _ = world.base_pose()
        assert sensor.contacts() == [Contact("foot", "floor")], (x, y)
        assert position[2] > 0.25, f"fell over at {(x, y)}"


def test_a_package_name_resolves_to_the_cooked_ranch(ranch: supersplat.ScanScene) -> None:
    from dim_go2_sim.module import resolve_scene

    scene = resolve_scene(SceneConfig(package=f"supersplat/{RANCH}"))
    assert (scene.mjcf, scene.visual, scene.y_up) == (ranch.mjcf, ranch.splats, False)
    assert scene.spawns[0] == ranch.spawns[0] and scene.lidar_mesh is not None


def test_the_ranch_splats_lie_on_its_collision_surfaces(ranch: supersplat.ScanScene) -> None:
    import open3d as o3d

    s = load_ply(ranch.splats)
    sigma = np.exp(s.log_scales.astype(np.float32).max(axis=1))
    sure = np.flatnonzero((s.opacities > 0.9) & (sigma < 0.05))
    sample = s.positions[np.random.default_rng(0).choice(sure, 20_000, replace=False)]
    mesh = np.load(ranch.collision)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(mesh["vertices"]), o3d.core.Tensor(mesh["faces"]))
    distance = scene.compute_distance(o3d.core.Tensor(sample)).numpy()
    assert np.median(distance) < supersplat.VOXEL, f"{np.median(distance):.3f} m off the voxels"


def test_the_camera_sees_a_room_from_a_ranch_spawn(ranch: supersplat.ScanScene) -> None:
    x, y, yaw = ranch.spawns[0]
    camera = SceneCamera(str(ranch.splats), False, CameraConfig(width=160, height=120))
    try:
        rgb = camera.capture(
            np.array([x, y, 0.35]),
            np.array([math.cos(yaw), math.sin(yaw), 0.0]),
            np.array([0.0, 0.0, 1.0]),
        )
    finally:
        camera.close()
    background = np.array([0x1B, 0x22, 0x30])
    # The scan has no ceiling, and nothing outside its windows: the background shows there.
    lower = rgb[60:].astype(int)
    assert (np.abs(lower - background).sum(axis=2) > 30).mean() > 0.95, "gaps"
    assert rgb.std() > 20, "a blur, not a room"
