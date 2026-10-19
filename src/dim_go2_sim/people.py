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

"""People walking slow loops through a scene, as kinematic (mocap) bodies the robot bumps into.

A person is a few primitives, 1.69 m tall and 0.47 m across the arms: legs, torso, arms, neck
and head. MuJoCo collides them with the robot only (not the scene, nor each other), the lidar
casts them, the camera draws them (as opaque splats among a scene's, or as meshes), and the URDF
carries them. Each walks its loop at a steady pace, and waits while the robot is in its way.

No dimOS imports: the camera's child process loads this.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import heapq
import math
from typing import Literal

import mujoco
import numpy as np
from numpy.typing import NDArray

from dim_go2_sim.config import PeopleConfig

# The geom group people are in, which the lidar casts (the robot's are 2 and 3, the lens guard 5).
GROUP = 4
# People collide with geoms whose contype has this bit: only the robot's.
COLLISION_BIT = 2
BODY = "person_{}"
Material = Literal["skin", "shirt", "pants"]
SKIN, PANTS = (0.80, 0.62, 0.50), (0.20, 0.24, 0.32)
SHIRTS = ((0.75, 0.20, 0.18), (0.20, 0.45, 0.70), (0.30, 0.60, 0.30), (0.85, 0.65, 0.20))
# The robot is in a person's way within this far either side of their path: half its length (it
# may stand side-on) plus half a person's width.
IN_THE_WAY = 0.6
# Nor do they walk on with the robot this close beside them, unless it is behind them: half its
# length plus half their width, and a margin.
BESIDE = 0.7
TURN_RATE = 2.5  # rad/s: a person turns toward their path no faster
# Route planning: grid step, a person's clearance from anything, and the heights checked.
PLAN_STEP = 0.1
CLEARANCE = 0.4
CLEARANCE_HEIGHTS = (0.3, 1.0, 1.6)
WAYPOINTS = 4
WAYPOINT_SPACING = 3.0  # metres between a loop's waypoints
START_DISTANCE = 2.5  # a person starts at least this far from the robot


@dataclass(frozen=True)
class Part:
    shape: Literal["capsule", "box", "sphere"]
    size: tuple[float, float, float]  # radius and half-length (capsule), half-extents (box)
    pos: tuple[float, float, float]  # in the person's frame: x forward, z up, feet at z = 0
    material: Material


PARTS = (
    Part("capsule", (0.07, 0.365, 0.0), (0.0, 0.09, 0.445), "pants"),
    Part("capsule", (0.07, 0.365, 0.0), (0.0, -0.09, 0.445), "pants"),
    Part("box", (0.11, 0.15, 0.28), (0.0, 0.0, 1.15), "shirt"),
    Part("capsule", (0.045, 0.255, 0.0), (0.0, 0.19, 1.13), "shirt"),
    Part("capsule", (0.045, 0.255, 0.0), (0.0, -0.19, 1.13), "shirt"),
    Part("sphere", (0.05, 0.0, 0.0), (0.0, 0.0, 1.46), "skin"),
    Part("sphere", (0.105, 0.0, 0.0), (0.0, 0.0, 1.585), "skin"),
)


def colors(index: int) -> dict[Material, tuple[float, float, float]]:
    return {"skin": SKIN, "shirt": SHIRTS[index % len(SHIRTS)], "pants": PANTS}


def add_people(spec: mujoco.MjSpec, count: int) -> None:
    """*count* people, as mocap bodies far below the floor until they are placed."""
    shapes = {
        "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
        "box": mujoco.mjtGeom.mjGEOM_BOX,
        "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
    }
    for i in range(count):
        body = spec.worldbody.add_body(name=BODY.format(i), mocap=True, pos=[0, 0, -10.0 - 3 * i])
        for part in PARTS:
            body.add_geom(
                type=shapes[part.shape],
                size=list(part.size),
                pos=list(part.pos),
                group=GROUP,
                contype=0,
                conaffinity=COLLISION_BIT,
                rgba=[*colors(i)[part.material], 1.0],
            )


def collide_with_people(spec: mujoco.MjSpec) -> None:
    """Every geom of *spec* that collides at all collides with people too."""
    for geom in spec.geoms:
        if geom.contype:
            geom.contype |= COLLISION_BIT


def mesh(material: Material, segments: int = 16) -> tuple[NDArray[np.float32], NDArray[np.uint32]]:
    """A person's surfaces of one material, as triangles, in the person's frame."""
    vertices, faces = [], []
    for part in PARTS:
        if part.material != material:
            continue
        v, f = _part_mesh(part, segments)
        faces.append(f + sum(len(x) for x in vertices))
        vertices.append(v)
    return np.concatenate(vertices).astype(np.float32), np.concatenate(faces).astype(np.uint32)


def _part_mesh(part: Part, segments: int) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    if part.shape == "box":
        corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
        quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
        faces = np.array([t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))])
        return corners * part.size + part.pos, faces
    radius, half = part.size[0], part.size[1] if part.shape == "capsule" else 0.0
    # A sphere of rings, its upper half raised and its lower half lowered by half.
    rings = segments // 2 + 1
    theta = np.linspace(0, math.pi, rings)
    phi = np.linspace(0, 2 * math.pi, segments, endpoint=False)
    t, p = np.meshgrid(theta, phi, indexing="ij")
    points = radius * np.stack([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)], -1)
    points[:, :, 2] += np.where(t <= math.pi / 2, half, -half)
    index = np.arange(rings * segments).reshape(rings, segments)
    a, b = index[:-1], np.roll(index[:-1], -1, axis=1)
    c, d = index[1:], np.roll(index[1:], -1, axis=1)
    faces = np.concatenate([np.stack([a, c, d], -1), np.stack([a, d, b], -1)]).reshape(-1, 3)
    return points.reshape(-1, 3) + part.pos, faces


def splats(index: int, spacing: float = 0.02) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Person *index* as opaque round splats over their surfaces, *spacing* apart: positions in
    the person's frame and RGB, shaded by a light from above in front."""
    rng = np.random.default_rng(index)
    light = np.array([0.4, -0.3, 0.85]) / np.linalg.norm([0.4, -0.3, 0.85])
    positions, rgb = [], []
    for material, color in colors(index).items():
        vertices, faces = mesh(material, segments=24)
        corners = vertices[faces].astype(np.float64)
        normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        areas = np.linalg.norm(normals, axis=1) / 2
        normals /= np.maximum(2 * areas[:, None], 1e-12)
        count = math.ceil(areas.sum() / spacing**2)
        chosen = rng.choice(len(faces), count, p=areas / areas.sum())
        u, v = rng.random((2, count))
        flip = u + v > 1
        u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
        tri = corners[chosen]
        positions.append(
            tri[:, 0] + u[:, None] * (tri[:, 1] - tri[:, 0]) + v[:, None] * (tri[:, 2] - tri[:, 0])
        )
        shade = 0.55 + 0.45 * np.clip(normals[chosen] @ light, 0, 1)
        rgb.append(np.asarray(color) * shade[:, None])
    return np.concatenate(positions).astype(np.float32), np.concatenate(rgb).astype(np.float32)


def yaw_rotation(yaw: float) -> NDArray[np.float64]:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class Route:
    """A closed loop of straight legs through (x, y) waypoints."""

    def __init__(self, waypoints: Sequence[Sequence[float]]) -> None:
        self.points = np.asarray(waypoints, np.float64).reshape(-1, 2)
        legs = np.roll(self.points, -1, axis=0) - self.points
        self.lengths = np.linalg.norm(legs, axis=1)
        self.ends = np.cumsum(self.lengths)
        self.length = float(self.ends[-1])
        if self.length <= 0:
            raise ValueError(f"a route needs two distinct waypoints: {waypoints}")

    def at(self, distance: float) -> tuple[NDArray[np.float64], float]:
        """The point *distance* along the loop, and the heading of the leg it is on."""
        distance %= self.length
        leg = min(int(np.searchsorted(self.ends, distance, side="right")), len(self.ends) - 1)
        while self.lengths[leg] == 0:
            leg = (leg + 1) % len(self.lengths)
        start = self.ends[leg] - self.lengths[leg]
        a, b = self.points[leg], self.points[(leg + 1) % len(self.points)]
        point = a + (b - a) * (distance - start) / self.lengths[leg]
        return point, math.atan2(b[1] - a[1], b[0] - a[0])


class People:
    """Where each person is (x, y, yaw), walking their route; they wait for the robot when it is
    in their way, and after config.patience_s turn back and walk their loop the other way."""

    def __init__(
        self,
        routes: Sequence[Sequence[Sequence[float]]],
        config: PeopleConfig,
        robot_start: tuple[float, float],
    ) -> None:
        self.config = config
        self.routes = [Route(r) for r in routes]
        self._robot_start = np.asarray(robot_start, np.float64)
        self.reset()

    def reset(self) -> None:
        """Each person at the first point of their loop START_DISTANCE from the robot's start."""
        self._along = []
        for route in self.routes:
            steps = np.arange(0.0, route.length, 0.1)
            far = [
                s
                for s in steps
                if np.hypot(*(route.at(s)[0] - self._robot_start)) >= START_DISTANCE
            ]
            self._along.append(far[0] if far else 0.0)
        self.yaws = [route.at(s)[1] for route, s in zip(self.routes, self._along, strict=True)]
        self.waiting = [False] * len(self.routes)
        self._waited = [0.0] * len(self.routes)
        self._direction = [1.0] * len(self.routes)  # along their loop, or back the other way

    def step(self, dt: float, robot_xy: NDArray[np.float64]) -> None:
        for i, route in enumerate(self.routes):
            point, heading = route.at(self._along[i])
            heading += math.pi if self._direction[i] < 0 else 0.0
            ahead = np.array([math.cos(heading), math.sin(heading)])  # the way they're walking
            offset = robot_xy[:2] - point
            along = float(offset @ ahead)
            across = abs(float(offset[0] * ahead[1] - offset[1] * ahead[0]))
            ahead_of_them = 0 < along < self.config.yield_distance and across < IN_THE_WAY
            beside_them = along > -0.2 and math.hypot(along, across) < BESIDE
            self.waiting[i] = ahead_of_them or beside_them
            self._waited[i] = self._waited[i] + dt if self.waiting[i] else 0.0
            if self._waited[i] >= self.config.patience_s:
                self._direction[i], self._waited[i] = -self._direction[i], 0.0
            if not self.waiting[i]:
                self._along[i] += self._direction[i] * self.config.speed * dt
            turn = (heading - self.yaws[i] + math.pi) % (2 * math.pi) - math.pi
            self.yaws[i] += float(np.clip(turn, -TURN_RATE * dt, TURN_RATE * dt))

    def poses(self) -> NDArray[np.float64]:
        """(N, 3): x, y, yaw of each person."""
        rows = [
            [*route.at(s)[0], yaw]
            for route, s, yaw in zip(self.routes, self._along, self.yaws, strict=True)
        ]
        return np.asarray(rows, np.float64).reshape(-1, 3)


Cast = Callable[[NDArray[np.float64], NDArray[np.float64], float], NDArray[np.float64]]


def plan_routes(
    cast: Cast,
    bounds: tuple[NDArray[np.float64], NDArray[np.float64]],
    start: tuple[float, float],
    count: int,
    seed: int,
) -> list[list[tuple[float, float]]]:
    """*count* loops through WAYPOINTS spots on the open floor reachable from *start*, each leg the
    shortest way between them that keeps CLEARANCE from anything. *cast(origins, directions,
    max_range)* is each ray's distance to the static scene, -1 for a miss."""
    free, origin = open_floor(cast, bounds)
    reachable = _flood(free, _cell(start, origin))
    cells = np.argwhere(reachable)
    if not len(cells):
        raise RuntimeError(f"no open floor for people around the robot's start {start}")
    rng = np.random.default_rng(seed)
    routes = []
    for _ in range(count):
        chosen: list[NDArray[np.int64]] = []
        for candidate in rng.permutation(cells):
            if all(np.hypot(*(candidate - c)) * PLAN_STEP >= WAYPOINT_SPACING for c in chosen):
                chosen.append(candidate)
                if len(chosen) == WAYPOINTS:
                    break
        if len(chosen) < 2:
            raise RuntimeError(f"too little open floor around {start} for people to walk a loop")
        corners: list[tuple[int, int]] = []
        for a, b in zip(chosen, chosen[1:] + chosen[:1], strict=True):
            corners += _straightened(reachable, _shortest(reachable, tuple(a), tuple(b)))[:-1]
        routes.append(
            [tuple(float(v) for v in origin + (np.array(c) + 0.5) * PLAN_STEP) for c in corners]
        )
    return routes


def open_floor(
    cast: Cast, bounds: tuple[NDArray[np.float64], NDArray[np.float64]]
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """Grid cells (PLAN_STEP) with floor under them and nothing within CLEARANCE at any of
    CLEARANCE_HEIGHTS, and the grid's corner."""
    low, high = bounds
    origin = low[:2]
    shape = np.maximum(np.ceil((high[:2] - low[:2]) / PLAN_STEP).astype(int), 1)
    centers = origin + (np.indices(shape).reshape(2, -1).T + 0.5) * PLAN_STEP
    top = CLEARANCE_HEIGHTS[0]
    floor = cast(
        np.column_stack([centers, np.full(len(centers), top)]),
        np.tile([0.0, 0.0, -1.0], (len(centers), 1)),
        top + 0.1,
    )
    free = np.abs(floor - top) < 0.05  # the floor (z = 0), not something on it
    angles = np.linspace(0, 2 * math.pi, 16, endpoint=False)
    around = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(16)])
    for height in CLEARANCE_HEIGHTS:
        candidates = centers[free]
        origins = np.repeat(np.column_stack([candidates, np.full(len(candidates), height)]), 16, 0)
        hits = cast(origins, np.tile(around, (len(candidates), 1)), CLEARANCE).reshape(-1, 16)
        still = np.flatnonzero(free)[(hits < 0).all(axis=1)]
        free[:] = False
        free[still] = True
    return free.reshape(shape), origin


def _cell(xy: Sequence[float], origin: NDArray[np.float64]) -> tuple[int, int]:
    i, j = np.floor((np.asarray(xy) - origin) / PLAN_STEP).astype(int)
    return int(i), int(j)


def _flood(free: NDArray[np.bool_], start: tuple[int, int]) -> NDArray[np.bool_]:
    """The free cells 4-connected to the free cell nearest *start*."""
    from scipy import ndimage

    labels, _ = ndimage.label(free)
    cells = np.argwhere(free)
    if not len(cells):
        return free
    nearest = cells[np.argmin(np.hypot(*(cells - start).T))]
    reachable: NDArray[np.bool_] = labels == labels[tuple(nearest)]
    return reachable


def _shortest(
    free: NDArray[np.bool_], a: tuple[int, int], b: tuple[int, int]
) -> list[tuple[int, int]]:
    """A* over 8-connected free cells, from a to b inclusive."""
    steps = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj]
    came: dict[tuple[int, int], tuple[int, int]] = {}
    cost = {a: 0.0}
    frontier = [(0.0, a)]
    while frontier:
        _, cell = heapq.heappop(frontier)
        if cell == b:
            break
        for di, dj in steps:
            nxt = (cell[0] + di, cell[1] + dj)
            if not (0 <= nxt[0] < free.shape[0] and 0 <= nxt[1] < free.shape[1]) or not free[nxt]:
                continue
            new = cost[cell] + math.hypot(di, dj)
            if new < cost.get(nxt, math.inf):
                cost[nxt], came[nxt] = new, cell
                heapq.heappush(frontier, (new + math.hypot(b[0] - nxt[0], b[1] - nxt[1]), nxt))
    path = [b]
    while path[-1] != a:
        path.append(came[path[-1]])
    return path[::-1]


def _straightened(free: NDArray[np.bool_], path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """The path's corners: from each kept cell, straight to the farthest cell in clear sight."""

    def clear(a: tuple[int, int], b: tuple[int, int]) -> bool:
        n = max(abs(b[0] - a[0]), abs(b[1] - a[1])) * 2 + 1
        cells = np.rint(np.linspace(a, b, n)).astype(int)
        return bool(free[cells[:, 0], cells[:, 1]].all())

    corners, i = [path[0]], 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not clear(path[i], path[j]):
            j -= 1
        corners.append(path[j])
        i = j
    return corners
