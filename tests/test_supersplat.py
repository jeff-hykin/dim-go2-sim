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
from xml.etree import ElementTree

import numpy as np
from numpy.typing import NDArray
from PIL import Image
import pytest
from scipy.spatial.transform import Rotation

from dim_go2_sim import supersplat
from dim_go2_sim.camera import SceneCamera
from dim_go2_sim.config import CameraConfig, PhysicsConfig, SceneConfig
from dim_go2_sim.contacts import Contact, ContactSensor
from dim_go2_sim.interiorgs import ROBOT_HEIGHT
from dim_go2_sim.splat import Splats, load_ply, load_sog, save_ply
from dim_go2_sim.world import Go2World, load_world

# The NJ ranch home the real-data tests cook (~31 MB the first time): opt in with
# DIM_GO2_SIM_DOWNLOAD=1, or have it cooked already.
RANCH = "3f89bbd3"
# The ranch's two bottom (south) bedrooms, a point on open floor in each, and how far the Go2's
# centre keeps from anything in its way (half its width, and a little).
BOTTOM_BEDROOMS = {"south-west": (0.0, -9.0), "south-east": (4.5, -9.0)}
DOG_CLEARANCE = 0.17
# The north-south hallway from the dining area down to the bottom bedrooms: its walls' faces
# (x), and its ends (y), as the scan has them.
HALLWAY_WALLS = (2.33, 3.28)
HALLWAY_ENDS = (-3.58, -6.88)
# A point on open floor in each of the ranch's rooms (its edits rebuild them room by room).
RANCH_ROOMS = {
    "living room": (4.5, 3.5),
    "entry": (5.0, -3.2),
    "dining room": (1.2, -2.0),
    "kitchen": (0.3, 3.0),
    "family room": (-5.0, 1.0),
    "hall bathroom": (1.6, -4.1),
    "hallway": (2.8, -5.5),
    "cross hall": (3.0, -7.3),
    "office": (5.5, -6.5),
    "south-west bedroom": (0.0, -9.0),
    "south-east bedroom": (4.5, -9.0),
    "north hall": (-2.55, 5.0),
    "main bathroom": (-5.0, 4.5),
    "north bedroom": (-5.0, 8.8),
}
# The plants an eval grades, each at (x, y, the height of its leaves), and how near the robot must
# get to one, seeing it.
RANCH_PLANTS = {
    "palm on the bedroom floor": (-6.6, 8.1, 0.9),
    "kitchen shelf plants": (-1.75, 3.2, 1.65),
    "sideboard plant": (-0.35, -3.35, 1.05),
    "stove-ledge plant": (5.45, -2.4, 1.4),
    "entry-table plant": (4.27, -3.8, 0.95),
}
PLANT_REACH = 1.2
# The ranch's beds' footprints: solid to the floor, so the robot can't get under one or into it.
RANCH_BEDS = {
    "north": (-6.25, 9.35, -4.35, 11.8),
    "south-west": (-1.0, -11.65, 0.65, -9.35),
    "south-east": (4.4, -11.55, 6.0, -9.3),
}


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


def test_a_cleared_region_empties_the_voxels_inside_it_and_no_others(tmp_path: Path) -> None:
    occupied, origin = supersplat.voxelize(room_scan())
    v = supersplat.VOXEL
    before = occupied.copy()
    # The cube's voxels, by their faces (as boxes merged from them give them), to the mm.
    i = np.argwhere(occupied)
    centres = origin + (i + 0.5) * v
    cube = (centres[:, :2] > [4.9, 2.9]).all(axis=1) & (centres[:, 2] < 0.7)
    cube &= (centres[:, :2] < [5.7, 3.7]).all(axis=1)
    low = np.round(origin + i[cube].min(axis=0) * v, 3)
    high = np.round(origin + (i[cube].max(axis=0) + 1) * v, 3)
    supersplat.clear(occupied, origin, [(*low, *high)])
    expected = before.copy()
    expected[tuple(i[cube].T)] = False
    assert cube.sum() > 100 and (occupied == expected).all()
    # Cooked with the whole room cleared, only the floor is left to collide with.
    supersplat.cook(room_scan(), tmp_path, "room", 1.0, supersplat.Edits(((-9, -9, -9, 9, 9, 9),)))
    assert np.load(tmp_path / "collision.npz")["vertices"][:, 2].max() == 0.0


def test_an_added_solid_fills_the_voxels_inside_it_even_where_it_is_cleared(
    tmp_path: Path,
) -> None:
    room = room_scan()
    wall = (2.0, 0.5, 0.0, 2.1, 3.5, 1.5)  # thinner than two voxels, across the open floor
    edits = supersplat.Edits(cleared=((1.5, 0.0, 0.0, 2.5, 4.0, 2.0),), added=(wall,))
    supersplat.cook(room, tmp_path, "room", 1.0, edits)
    boxes = np.array(
        [
            [float(n) for n in g.get("pos").split() + g.get("size").split()]
            for g in ElementTree.parse(tmp_path / "scene.xml").iter("geom")
            if g.get("type") == "box"
        ]
    )
    # The cooked room is centred: find the wall by its length along y, where nothing else is.
    low, high = boxes[:, :3] - boxes[:, 3:], boxes[:, :3] + boxes[:, 3:]
    long = boxes[(high[:, 1] - low[:, 1] > 2.5) & (high[:, 2] - low[:, 2] > 1.0)]
    assert len(long) >= 1, "the added wall isn't a box"
    thickness = 2 * long[:, 3].sum()
    assert supersplat.VOXEL <= thickness <= 3 * supersplat.VOXEL


def test_the_floor_covers_the_whole_house_as_a_few_rectangles(tmp_path: Path) -> None:
    room = room_scan()
    # A room whose scan saw no floor in its middle (a dim hallway) is floored all the same.
    keep = ~((np.abs(room.positions[:, 2]) < 0.03) & (np.abs(room.positions[:, 0] - 3) < 1.5))
    fields = ("positions", "rotations", "log_scales", "opacities", "sh")
    supersplat.cook(Splats(*(getattr(room, f)[keep] for f in fields)), tmp_path, "r", 1.0)
    mesh = np.load(tmp_path / "collision.npz")
    triangles = mesh["vertices"][mesh["faces"]]
    floor = triangles[(np.abs(triangles[:, :, 2]) < 1e-6).all(axis=1)]
    area = 0.5 * np.abs(np.cross(floor[:, 1] - floor[:, 0], floor[:, 2] - floor[:, 0])[:, 2])
    span = floor[:, :, :2].reshape(-1, 2)
    assert area.sum() >= 0.95 * np.prod(span.max(axis=0) - span.min(axis=0)), "holes"
    assert len(floor) < 100, f"{len(floor)} floor triangles"


def _ranch_boxes(ranch: supersplat.ScanScene) -> NDArray[np.float64]:
    """The cooked ranch's MuJoCo boxes: centre and half-size, x y z."""
    return np.array(
        [
            [float(n) for n in g.get("pos").split() + g.get("size").split()]
            for g in ElementTree.parse(ranch.mjcf).iter("geom")
            if g.get("type") == "box"
        ]
    )


def _ranch_grid(
    ranch: supersplat.ScanScene,
) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.bool_]]:
    """The cooked ranch's grid corner, its floor columns and its columns blocked below the
    robot's height, from its MuJoCo boxes and the lidar's floor quads."""
    v = supersplat.VOXEL
    boxes = _ranch_boxes(ranch)
    mesh = np.load(ranch.collision)
    floor_low, has_floor = _floor_grid(mesh["vertices"], mesh["faces"])
    low = np.minimum(floor_low, (boxes[:, :2] - boxes[:, 3:5]).min(axis=0)) - 1
    shape = tuple(np.round((floor_low - low) / v).astype(int) + has_floor.shape + 40)
    has_floor = _floor_grid(mesh["vertices"], mesh["faces"], low, shape)[1]
    blocked = np.zeros(shape, bool)
    for x, y, _, sx, sy, _ in boxes[boxes[:, 2] - boxes[:, 5] < ROBOT_HEIGHT]:
        a = np.round((np.array([x - sx, y - sy]) - low) / v).astype(int)
        b = np.round((np.array([x + sx, y + sy]) - low) / v).astype(int)
        blocked[a[0] : b[0], a[1] : b[1]] = True
    return low, has_floor, blocked


def _floor_grid(
    vertices: NDArray[np.float32],
    faces: NDArray[np.uint32],
    low: NDArray[np.float64] | None = None,
    shape: tuple[int, ...] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """The VOXEL columns under the level triangles at z = 0 (the floor's rectangles, each two
    triangles that span it), on a grid from *low* (by default the floor's corner)."""
    v = supersplat.VOXEL
    triangles = vertices[faces]
    floor = triangles[(np.abs(triangles[:, :, 2]) < 1e-6).all(axis=1)][:, :, :2]
    if low is None:
        low = floor.reshape(-1, 2).min(axis=0)
    if shape is None:
        shape = tuple(np.ceil((floor.reshape(-1, 2).max(axis=0) - low) / v).astype(int))
    has_floor = np.zeros(shape, bool)
    for a, b in zip(floor.min(axis=1), floor.max(axis=1), strict=True):
        i, j = np.round((a - low) / v).astype(int), np.round((b - low) / v).astype(int)
        has_floor[i[0] : j[0], i[1] : j[1]] = True
    return low, has_floor


def test_the_ranch_cleared_regions_are_open(ranch: supersplat.ScanScene) -> None:
    edits = supersplat.bundled_edits(RANCH)
    regions, added = np.array(edits.cleared), np.array(edits.added)
    assert len(regions) > 0 and len(added) > 0
    centres = (regions[:, :3] + regions[:, 3:]) / 2
    # A region an added solid walls over again (to within the half voxel it takes) is solid.
    v = supersplat.VOXEL / 2
    walled = (centres[:, None] > added[None, :, :3] - v) & (
        centres[:, None] < added[None, :, 3:] + v
    )
    centres = centres[~walled.all(axis=2).any(axis=1)]
    boxes = _ranch_boxes(ranch)
    inside = (np.abs(centres[:, None, :] - boxes[None, :, :3]) < boxes[None, :, 3:]).all(axis=2)
    assert not inside.any(), f"{inside.any(axis=1).sum()} cleared regions still collide"


def test_the_ranch_hallway_is_clear_wall_to_wall(ranch: supersplat.ScanScene) -> None:
    from scipy import ndimage

    v = supersplat.VOXEL
    low, has_floor, blocked = _ranch_grid(ranch)
    room = ndimage.distance_transform_edt(has_floor & ~blocked) * v >= DOG_CLEARANCE
    # The robot's centre fits anywhere a voxel more than DOG_CLEARANCE from either wall.
    left, right = HALLWAY_WALLS[0] + DOG_CLEARANCE + v, HALLWAY_WALLS[1] - DOG_CLEARANCE - v
    for x in np.arange(left, right, v):
        for y in np.arange(HALLWAY_ENDS[0] - 0.1, HALLWAY_ENDS[1] - 0.1, -v):
            cell = tuple(np.floor((np.array([x, y]) - low) / v).astype(int))
            assert room[cell], f"the hallway is blocked at ({x:.2f}, {y:.2f})"


def test_the_go2_walks_straight_down_the_ranch_hallway(ranch: supersplat.ScanScene) -> None:
    physics = PhysicsConfig(realtime=False)
    world = Go2World(load_world(ranch.mjcf, physics), physics)
    robot = {b for b in range(world.model.nbody) if world.is_robot(b)}
    sensor = ContactSensor(world.model, world.data, robot, world.trunk, world.feet, 0.05)
    middle = sum(HALLWAY_WALLS) / 2
    world.reset(middle, HALLWAY_ENDS[0] + 0.2, -math.pi / 2, ranch.floor_z)
    touched = set()
    for _ in range(600):
        position, rotation = world.base_pose()
        if position[1] < HALLWAY_ENDS[1] - 0.2:
            break
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        # Straight south, held to the middle as a driver would.
        steer = -2.0 * (yaw + math.pi / 2) - 2.0 * (middle - position[0])
        world.tick(np.array([0.5, 0.0, steer]))
        touched.update(sensor.contacts())
    assert position[1] < HALLWAY_ENDS[1] - 0.2, f"stopped at {position[:2].round(2)}"
    assert touched <= {Contact("foot", "floor")}, touched


def test_a_ray_across_the_ranch_hallway_hits_its_walls(ranch: supersplat.ScanScene) -> None:
    import open3d as o3d

    mesh = np.load(ranch.collision)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(mesh["vertices"]), o3d.core.Tensor(mesh["faces"]))
    middle = sum(HALLWAY_WALLS) / 2
    ys = np.linspace(HALLWAY_ENDS[1] + 0.3, -4.8, 12)
    rays = [[middle, y, z, d, 0, 0] for y in ys for z in (0.3, 1.0, 1.7) for d in (-1, 1)]
    hits = scene.cast_rays(o3d.core.Tensor(np.array(rays, np.float32)))["t_hit"].numpy()
    wall = np.array([middle - HALLWAY_WALLS[0], HALLWAY_WALLS[1] - middle] * (len(rays) // 2))
    assert np.abs(hits - wall).max() < supersplat.VOXEL, np.abs(hits - wall).max()


def test_the_ranch_floor_is_shared_down_the_hallway(
    ranch: supersplat.ScanScene, tmp_path: Path
) -> None:
    from dim_go2_sim.config import UrdfConfig
    from dim_go2_sim.urdf import SceneSource, _write_scene

    mesh = np.load(ranch.collision)
    source = SceneSource("ranch", "", lambda: (mesh["vertices"], mesh["faces"]))
    _write_scene(source, UrdfConfig(), tmp_path)
    tiles = [_read_stl(path.read_bytes()) for path in sorted(tmp_path.glob("mesh_*.stl"))]
    vertices = np.concatenate([t.reshape(-1, 3) for t in tiles])
    faces = np.arange(len(vertices), dtype=np.uint32).reshape(-1, 3)
    low = np.array([HALLWAY_WALLS[0], HALLWAY_ENDS[1]])
    shape = tuple(
        np.floor((np.array([HALLWAY_WALLS[1], HALLWAY_ENDS[0]]) - low) / 0.05).astype(int)
    )
    covered = _floor_grid(vertices, faces, low, shape)[1]
    assert covered.all(), f"{(~covered).sum()} of {covered.size} hallway cells have no floor"


def _read_stl(data: bytes) -> NDArray[np.float32]:
    """A binary STL's triangles' corners."""
    count = int.from_bytes(data[80:84], "little")
    records = np.frombuffer(data[84 : 84 + 50 * count], np.dtype([("f", "<f4", 12), ("a", "<u2")]))
    return records["f"][:, 3:].reshape(-1, 3, 3)


def test_the_ranch_bottom_bedrooms_are_reachable_from_the_start(
    ranch: supersplat.ScanScene,
) -> None:
    from scipy import ndimage

    v = supersplat.VOXEL
    low, has_floor, blocked = _ranch_grid(ranch)
    # Where the robot's centre can be: open floor DOG_CLEARANCE from anything in its way.
    room = ndimage.distance_transform_edt(has_floor & ~blocked) * v >= DOG_CLEARANCE
    labels, _ = ndimage.label(room)
    start = labels[tuple(np.floor((np.array(ranch.spawns[0][:2]) - low) / v).astype(int))]
    assert start > 0
    for name, (x, y) in BOTTOM_BEDROOMS.items():
        cell = tuple(np.floor((np.array([x, y]) - low) / v).astype(int))
        assert labels[cell] == start, f"the {name} bedroom isn't reachable from the start"


def _reachable(ranch: supersplat.ScanScene) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """The ranch's grid corner, and the columns the robot's centre can reach from the start: open
    floor DOG_CLEARANCE from anything in its way, connected to the first spawn."""
    from scipy import ndimage

    v = supersplat.VOXEL
    low, has_floor, blocked = _ranch_grid(ranch)
    room = ndimage.distance_transform_edt(has_floor & ~blocked) * v >= DOG_CLEARANCE
    labels, _ = ndimage.label(room)
    start = labels[tuple(np.floor((np.array(ranch.spawns[0][:2]) - low) / v).astype(int))]
    assert start > 0, "the start isn't on open floor"
    return low, labels == start


def _ranch_raycaster(ranch: supersplat.ScanScene):
    import open3d as o3d

    mesh = np.load(ranch.collision)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(mesh["vertices"]), o3d.core.Tensor(mesh["faces"]))
    return scene


def test_every_ranch_room_and_spawn_is_reachable_from_the_start(
    ranch: supersplat.ScanScene,
) -> None:
    v = supersplat.VOXEL
    low, reachable = _reachable(ranch)
    points = {**RANCH_ROOMS, **{f"spawn {i}": s[:2] for i, s in enumerate(ranch.spawns)}}
    for name, (x, y) in points.items():
        cell = tuple(np.floor((np.array([x, y]) - low) / v).astype(int))
        assert reachable[cell], f"the {name} isn't reachable from the start"


def test_the_robot_can_get_near_each_ranch_plant_and_see_it(ranch: supersplat.ScanScene) -> None:
    import open3d as o3d

    v = supersplat.VOXEL
    low, reachable = _reachable(ranch)
    scene = _ranch_raycaster(ranch)
    cells = low + (np.argwhere(reachable) + 0.5) * v
    for name, (x, y, height) in RANCH_PLANTS.items():
        near = cells[np.hypot(cells[:, 0] - x, cells[:, 1] - y) < PLANT_REACH]
        assert len(near), f"the robot can't get within {PLANT_REACH} m of the {name}"
        # From each, a ray at the plant's height towards it hits nothing before the plant itself.
        starts = np.column_stack([near, np.full(len(near), height)])
        ahead = np.array([x, y, height]) - starts
        distance = np.linalg.norm(ahead, axis=1)
        rays = np.column_stack([starts, ahead / distance[:, None]]).astype(np.float32)
        hits = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
        seen = hits > distance - 0.3
        assert seen.any(), f"the {name} is hidden from everywhere the robot can get near it"


def test_a_ranch_bed_blocks_a_low_ray_from_every_side(ranch: supersplat.ScanScene) -> None:
    import open3d as o3d

    scene = _ranch_raycaster(ranch)
    height = 0.2  # under a bed frame's rail, where the scan saw nothing
    for name, (x0, y0, x1, y1) in RANCH_BEDS.items():
        rays = []
        # Rays from 0.4 m off each side, square towards the bed, every 10 cm along it (a side that
        # stands against a wall is reached from inside the wall, which the ray leaves at once).
        for along in np.arange(0.1, 1.0, 0.1):
            x, y = x0 + along * (x1 - x0), y0 + along * (y1 - y0)
            for start, direction in (
                ((x0 - 0.4, y), (1, 0)),
                ((x1 + 0.4, y), (-1, 0)),
                ((x, y0 - 0.4), (0, 1)),
                ((x, y1 + 0.4), (0, -1)),
            ):
                rays.append([*start, height, *direction, 0])
        hits = scene.cast_rays(o3d.core.Tensor(np.array(rays, np.float32)))["t_hit"].numpy()
        gaps = hits > 0.4 + supersplat.VOXEL
        assert not gaps.any(), (
            f"the {name} bed lets a ray under it at {np.array(rays)[gaps][:3, :2]}"
        )


def test_nothing_floats_in_the_ranch(ranch: supersplat.ScanScene) -> None:
    from scipy import ndimage

    v = supersplat.VOXEL
    boxes = _ranch_boxes(ranch)
    low = (boxes[:, :3] - boxes[:, 3:]).min(axis=0)
    first = np.round((boxes[:, :3] - boxes[:, 3:] - low) / v).astype(int)
    last = np.round((boxes[:, :3] + boxes[:, 3:] - low) / v).astype(int)
    solid = np.zeros(last.max(axis=0), bool)
    for a, b in zip(first, last, strict=True):
        solid[a[0] : b[0], a[1] : b[1], a[2] : b[2]] = True
    labels, _ = ndimage.label(solid, np.ones((3, 3, 3)))
    # Every patch stands on the floor or reaches up past the boxes (a wall, a hanging lamp).
    held = set(np.unique(labels[:, :, 0])) | set(np.unique(labels[:, :, -1]))
    floating = sorted(set(np.unique(labels)) - held)
    sizes = [int((labels == k).sum()) for k in floating]
    assert not floating, f"{len(floating)} patches float in the air, {sizes} voxels"


def test_a_speck_an_edit_leaves_is_dropped() -> None:
    occupied = np.zeros((20, 20, 20), bool)
    occupied[2:12, 2:12, 0:10] = True  # a 1000-voxel block
    occupied[15:17, 15:17, 5:7] = True  # an 8-voxel speck
    supersplat.drop_floaters(occupied)
    assert occupied.sum() == 1000 and not occupied[15:17, 15:17, 5:7].any()


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
