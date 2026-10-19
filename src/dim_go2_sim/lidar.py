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

"""A rolling Mid-360 on the Go2: each slice of a scan is cast from where the head is then.

``mj_multiRay`` tests every ray against every geom, hopeless for a cooked store (~48k geoms).
The static scene never moves, so it is triangulated once into Embree; MuJoCo casts only
against a model of the robot alone, for its own body's occlusion. Nearest hit wins.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import NDArray
import open3d as o3d  # type: ignore[import-untyped]
from scipy.spatial.transform import Rotation

from dim_go2_sim.config import LidarConfig
from dim_go2_sim.mid360 import Mid360, Scan
from dim_go2_sim.world import LENS_GUARD_GROUP

ROUND_RESOLUTION = 12


def _primitive(geom_type: int, size: NDArray[np.float64]) -> o3d.geometry.TriangleMesh | None:
    """A centred triangle mesh for a primitive geom, in its own frame."""
    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        return o3d.geometry.TriangleMesh.create_box(*(2 * size[:3])).translate(-size[:3])
    if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        return o3d.geometry.TriangleMesh.create_sphere(size[0], resolution=ROUND_RESOLUTION // 2)
    if geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return o3d.geometry.TriangleMesh.create_cylinder(size[0], 2 * size[1], ROUND_RESOLUTION)
    if geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
        # A cylinder covering the caps; the ends are a few centimetres too square.
        return o3d.geometry.TriangleMesh.create_cylinder(
            size[0], 2 * (size[1] + size[0]), ROUND_RESOLUTION
        )
    if geom_type == mujoco.mjtGeom.mjGEOM_PLANE:
        half = 1000.0
        plane = o3d.geometry.TriangleMesh.create_box(2 * half, 2 * half, 0.001)
        return plane.translate((-half, -half, -0.001))
    return None


def static_geoms(model: mujoco.MjModel, robot_bodies: set[int]) -> NDArray[np.bool_]:
    """Geoms welded to the world, not on a mocap body or the robot."""
    body = model.geom_bodyid
    static = (model.body_weldid[body] == 0) & (model.body_mocapid[body] < 0)
    static &= ~np.isin(body, list(robot_bodies))
    return static


def static_triangles(
    model: mujoco.MjModel, data: mujoco.MjData, geoms: NDArray[np.bool_]
) -> tuple[NDArray[np.float32], NDArray[np.uint32]]:
    """World-frame vertices and faces of *geoms*."""
    vertices: list[NDArray[np.float64]] = []
    faces: list[NDArray[np.int64]] = []
    count = 0
    for i in np.flatnonzero(geoms):
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[i]
            v0, nv = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            f0, nf = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
            local, face = model.mesh_vert[v0 : v0 + nv], model.mesh_face[f0 : f0 + nf]
        else:
            primitive = _primitive(model.geom_type[i], model.geom_size[i])
            if primitive is None:
                continue  # height fields: none in cooked scenes
            local, face = np.asarray(primitive.vertices), np.asarray(primitive.triangles)
        rotation = data.geom_xmat[i].reshape(3, 3)
        vertices.append(local @ rotation.T + data.geom_xpos[i])
        faces.append(face + count)
        count += len(local)
    if not vertices:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint32)
    return np.concatenate(vertices).astype(np.float32), np.concatenate(faces).astype(np.uint32)


class Raycaster:
    """The static world through Embree, the robot through MuJoCo; nearest hit wins.

    *robot* is the robot alone, whose qpos is the world's from *robot_qpos* on. *static_mesh*
    (world-frame vertices and faces) stands in for the world's static geoms, when a scene's
    collision geometry is coarser than its surfaces.
    """

    def __init__(
        self,
        world: mujoco.MjModel,
        world_data: mujoco.MjData,
        robot_bodies: set[int],
        robot: mujoco.MjModel | None,
        robot_qpos: int,
        static_mesh: tuple[NDArray[np.float32], NDArray[np.uint32]] | None = None,
    ) -> None:
        vertices, faces = static_mesh or static_triangles(
            world, world_data, static_geoms(world, robot_bodies)
        )
        self._scene = o3d.t.geometry.RaycastingScene()
        if len(faces):
            self._scene.add_triangles(o3d.core.Tensor(vertices), o3d.core.Tensor(faces))
        self._world_data = world_data
        self._robot = robot
        self._robot_data = mujoco.MjData(robot) if robot is not None else None
        self._robot_qpos = robot_qpos
        self._groups = np.ones(6, dtype=np.uint8)
        self._groups[LENS_GUARD_GROUP] = 0  # nothing a laser can hit

    def cast(
        self, origin: NDArray[np.float64], directions: NDArray[np.float64], max_range: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Distance and hit normal (facing the ray) per world-frame unit direction; -1 is a miss."""
        rays = np.empty((len(directions), 6), np.float32)
        rays[:, :3] = origin
        rays[:, 3:] = directions
        hit = self._scene.cast_rays(o3d.core.Tensor(rays))
        dist = hit["t_hit"].numpy().astype(np.float64)
        normals = hit["primitive_normals"].numpy().astype(np.float64)
        dist[~np.isfinite(dist) | (dist > max_range)] = -1.0
        if self._robot is not None and self._robot_data is not None:
            d = self._robot_data
            d.qpos[:] = self._world_data.qpos[self._robot_qpos : self._robot_qpos + len(d.qpos)]
            mujoco.mj_kinematics(self._robot, d)
            own = np.full(len(directions), -1.0)
            own_normals = np.zeros_like(directions)
            mujoco.mj_multiRay(
                self._robot,
                d,
                np.asarray(origin, dtype=np.float64),
                np.ascontiguousarray(directions, dtype=np.float64).ravel(),
                self._groups,
                1,
                -1,
                np.full(len(directions), -1, dtype=np.int32),
                own,
                own_normals.ravel(),
                len(directions),
                max_range,
            )
            nearer = (own >= 0) & ((dist < 0) | (own < dist))
            dist[nearer] = own[nearer]
            normals[nearer] = own_normals[nearer]
        # Embree's normals follow the triangle winding; make them face the ray.
        away = np.einsum("ij,ij->i", normals, directions) > 0
        normals[away] *= -1
        return dist, normals


@dataclass(frozen=True)
class LidarScan:
    """Returns in the sensor frame where each ray fired, as the device reports them."""

    points: NDArray[np.float32]
    offsets: NDArray[np.float64]  # seconds from the scan's start
    lines: NDArray[np.uint8]
    start: float  # sim time


class RollingLidar:
    """Casts each motion bin of a scan from the sensor's pose at that moment."""

    def __init__(
        self,
        config: LidarConfig,
        raycaster: Raycaster,
        sensor_pose: Callable[[], tuple[NDArray[np.float64], NDArray[np.float64]]],
    ) -> None:
        self.config = config
        self.device = Mid360(
            downsample=config.downsample,
            point_rate_hz=config.point_rate_hz,
            min_range=config.min_range,
            max_range=config.max_range,
            noise=config.noise,
            dropout=config.dropout,
            seed=config.seed,
        )
        self._raycaster = raycaster
        self._sensor_pose = sensor_pose
        self._period = 1.0 / config.rate_hz
        self._begin(0.0)

    def _begin(self, start: float) -> None:
        self._start = start
        self._scan: Scan = self.device.scan(start, self._period)
        bins = np.floor(self._scan.offsets * self.config.motion_bins_hz).astype(np.int64)
        self._bins = [np.flatnonzero(bins == b) for b in np.unique(bins)]
        self._bin_times = [start + float(np.mean(self._scan.offsets[i])) for i in self._bins]
        self._next = 0
        self._ranges = np.full(len(self._scan.offsets), -1.0)
        self._cosines = np.zeros(len(self._scan.offsets))

    def advance(self, now: float) -> LidarScan | None:
        """Cast every bin due by sim time *now*; the scan once its last bin is cast."""
        while self._next < len(self._bins) and self._bin_times[self._next] <= now:
            indices = self._bins[self._next]
            origin, rotation = self._sensor_pose()
            directions = self._scan.directions[indices] @ rotation.T
            ranges, normals = self._raycaster.cast(origin, directions, self.config.max_range)
            self._ranges[indices] = ranges
            self._cosines[indices] = np.einsum("ij,ij->i", directions, normals)
            self._next += 1
        if self._next < len(self._bins):
            return None
        measured = self.device.measure(self._ranges, self._cosines, self._start)
        hit = measured >= 0
        scan = LidarScan(
            points=(self._scan.directions[hit] * measured[hit, None]).astype(np.float32),
            offsets=self._scan.offsets[hit],
            lines=self._scan.lines[hit],
            start=self._start,
        )
        self._begin(self._start + self._period)
        return scan


def mount_rotation(pitch_deg: float) -> NDArray[np.float64]:
    """A sensor pitched nose-down by pitch_deg, in the body frame."""
    rotation: NDArray[np.float64] = Rotation.from_euler("y", math.radians(pitch_deg)).as_matrix()
    return rotation


def scene_raycaster(mjcf: Path, cache_dir: Path | None = None) -> Raycaster:
    """Casts against a scene alone, without the robot: for surveying what the lidar could see."""
    from dim_go2_sim.config import PhysicsConfig
    from dim_go2_sim.world import PREFIX, load_world

    model = load_world(mjcf, PhysicsConfig(cache_dir=cache_dir))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    robot = {b for b in range(model.nbody) if model.body(b).name.startswith(PREFIX)}
    return Raycaster(model, data, robot, None, 0)
