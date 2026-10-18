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

"""Go2Sim: a simulated Unitree Go2 with a Mid-360 and a front camera, driven by cmd_vel."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import threading
import time
from typing import Any

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.msgs.sensor_msgs.Imu import Imu
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.msgs.tf2_msgs.TFMessage import TFMessage
from dimos.utils.logging_config import setup_logger
from dimos_lcm.std_msgs import String
import mujoco
import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from reactivex.disposable import Disposable
from scipy.spatial.transform import Rotation

from dim_go2_sim import interiorgs, supersplat
from dim_go2_sim.config import CameraConfig, LidarConfig, PhysicsConfig, SceneConfig, UrdfConfig
from dim_go2_sim.contacts import Contacts, ContactSensor
from dim_go2_sim.lidar import (
    LidarScan,
    Raycaster,
    RollingLidar,
    mount_rotation,
    static_geoms,
    static_triangles,
)
from dim_go2_sim.splat import is_splats
from dim_go2_sim.urdf import (
    MeshFile,
    SceneSource,
    SharedUrdf,
    body_in_parent,
    leg_bodies,
    share_urdf,
)
from dim_go2_sim.world import PREFIX, Go2World, compose, go2_spec, load_world

logger = setup_logger()

# The optical frame's axes (z forward, x right, y down) as columns, in a forward-looking body frame.
OPTICAL_IN_BODY = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


class Go2SimConfig(ModuleConfig):
    scene: SceneConfig = Field(default_factory=SceneConfig)
    physics: PhysicsConfig = Field(default_factory=PhysicsConfig)
    lidar: LidarConfig = Field(default_factory=LidarConfig)
    camera: CameraConfig = Field(default_factory=CameraConfig)
    urdf: UrdfConfig = Field(default_factory=UrdfConfig)
    odom_hz: float = 50.0
    imu_hz: float = 50.0
    frame_prefix: str = ""  # namespaces every frame published, e.g. "go2/"


class Go2Sim(Module):
    """cmd_vel in; the Go2's lidar, camera, IMU, odometry, tf, and contacts out.

    Odometry is the true body pose; contacts are ground truth, for grading.
    """

    config: Go2SimConfig
    dedicated_worker = True

    cmd_vel: In[Twist]
    lidar: Out[PointCloud2]
    color_image: Out[Image]
    camera_info: Out[CameraInfo]
    imu: Out[Imu]
    odom: Out[PoseStamped]
    tf: Out[TFMessage]
    contacts: Out[Contacts]
    # The robot and scene as a URDF, its meshes named package://dim_go2_sim/<path>?hash=<sha256>;
    # any of those names on mesh_request gets the file on mesh_info, unless it has &have=<sha256>.
    urdf_info: Out[String]
    mesh_request: In[String]
    mesh_info: Out[MeshFile]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._command = np.zeros(3)
        self._command_time = -math.inf
        self._pose: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None
        self._clock_origin = 0.0
        self._shared_urdf: SharedUrdf | None = None
        self._mesh_requests: list[str] = []
        self._mesh_wanted = threading.Condition()

    @rpc
    def start(self) -> None:
        super().start()
        cfg = self.config
        scene = resolve_scene(cfg.scene, cfg.physics.cache_dir)
        self.world = Go2World(load_world(scene.mjcf, cfg.physics), cfg.physics)
        floor_z = scene.floor_z if cfg.scene.floor_z is None else cfg.scene.floor_z
        x, y, yaw = cfg.scene.start or (scene.spawns[0] if scene.spawns else (0.0, 0.0, 0.0))
        self.world.reset(x, y, yaw, floor_z)
        model, data = self.world.model, self.world.data
        robot_bodies = {b for b in range(model.nbody) if self.world.is_robot(b)}
        floor_top_z = floor_z + 0.05 if cfg.scene.floor_top_z is None else cfg.scene.floor_top_z
        self.sensor = ContactSensor(
            model, data, robot_bodies, self.world.trunk, self.world.feet, floor_top_z
        )
        blocked = [c for c in self.sensor.contacts() if c.kind != "floor"]
        if blocked:
            raise RuntimeError(f"the robot starts inside scene geometry: {blocked}")
        self.frames = {
            name: cfg.frame_prefix + name
            for name in ("world", "base_link", "mid360_link", "camera_optical")
        }
        self._lidar: RollingLidar | None = None
        if cfg.lidar.enabled:
            robot = compose(None, cfg.physics, ground=False) if cfg.lidar.self_occlusion else None
            raycaster = Raycaster(
                model,
                data,
                robot_bodies,
                robot,
                int(model.jnt_qposadr[model.body_jntadr[self.world.trunk]]),
                scene.lidar_mesh,
            )
            self._lidar_mount = mount_rotation(cfg.lidar.pitch_deg)
            self._lidar = RollingLidar(cfg.lidar, raycaster, self._lidar_pose)
        # Each leg link's tf edge from its parent's frame, published with the body pose.
        legs = leg_bodies(model, self.world.trunk) if cfg.urdf.enabled else []
        self._leg_edges = [(b, p, self._frame_of(b), self._frame_of(p)) for b, p in legs]
        self.register_disposable(Disposable(self.cmd_vel.subscribe(self._on_cmd_vel)))
        self._stop.clear()
        self._threads = [threading.Thread(target=self._run, name="go2-sim", daemon=True)]
        if cfg.urdf.enabled:
            self.register_disposable(Disposable(self.mesh_request.subscribe(self._on_mesh_request)))
            self._threads += [
                threading.Thread(target=self._share_urdf, args=(scene,), daemon=True),
                threading.Thread(target=self._answer_mesh_requests, daemon=True),
            ]
        if cfg.camera.enabled and scene.visual is not None:
            self._threads.append(
                threading.Thread(
                    target=self._run_camera, args=(str(scene.visual), scene.y_up), daemon=True
                )
            )
        for thread in self._threads:
            thread.start()

    @rpc
    def stop(self) -> None:
        self._stop.set()
        with self._mesh_wanted:
            self._mesh_wanted.notify_all()
        for thread in self._threads:
            thread.join(timeout=10)
        super().stop()

    @rpc
    def status(self) -> dict[str, float]:
        """Sim time against wall time since the physics started."""
        return {"sim_time": self.world.time, "wall_time": time.time() - self._clock_origin}

    def _on_cmd_vel(self, twist: Twist) -> None:
        with self._lock:
            self._command = np.array([twist.linear.x, twist.linear.y, twist.angular.z])
            self._command_time = time.monotonic()

    def _frame_of(self, body: int) -> str:
        if body == self.world.trunk:
            return self.frames["base_link"]
        return self.config.frame_prefix + self.world.model.body(body).name.removeprefix(PREFIX)

    def _share_urdf(self, scene: ResolvedScene) -> None:
        """Builds the URDF's files (once per scene: a store's take a minute), then re-sends it."""
        cfg = self.config
        source = None
        if scene.name is not None:

            def mesh() -> tuple[NDArray[np.float32], NDArray[np.uint32]]:
                if scene.lidar_mesh is not None:
                    return scene.lidar_mesh
                model, data = self.world.model, self.world.data
                robot = {b for b in range(model.nbody) if self.world.is_robot(b)}
                # A plane is 2 km of floor: a viewer draws its own.
                planes = model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE
                return static_triangles(model, data, static_geoms(model, robot) & ~planes)

            splats = scene.visual if scene.visual and is_splats(scene.visual) else None
            source = SceneSource(scene.name, scene.key, mesh, splats)
        started = time.monotonic()
        robot = go2_spec(False).compile()
        shared = share_urdf(robot, cfg.frame_prefix, source, cfg.urdf, cfg.physics.cache_dir)
        self._shared_urdf = shared
        logger.info(
            f"urdf_info: {len(shared.files)} mesh files, "
            f"{sum(p.stat().st_size for p, _ in shared.files.values()) / 1e6:.1f} MB, "
            f"ready in {time.monotonic() - started:.1f} s"
        )
        while not self._stop.is_set():
            self.urdf_info.publish(String(shared.xml))
            self._stop.wait(cfg.urdf.period_s)

    def _on_mesh_request(self, request: String) -> None:
        with self._mesh_wanted:
            self._mesh_requests.append(request.data)
            self._mesh_wanted.notify()

    def _answer_mesh_requests(self) -> None:
        """One at a time, off the transport's thread: a file can be tens of MB."""
        while not self._stop.is_set():
            with self._mesh_wanted:
                while not self._mesh_requests and not self._stop.is_set():
                    self._mesh_wanted.wait()
                requests, self._mesh_requests = self._mesh_requests, []
            for request in dict.fromkeys(requests):
                shared = self._shared_urdf
                try:
                    mesh = shared.answer(request) if shared is not None else None
                except KeyError:
                    logger.warning(f"mesh_request for {request!r}: not a mesh in urdf_info")
                    continue
                if mesh is not None:
                    self.mesh_info.publish(mesh)

    def _lidar_pose(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        position, rotation = self.world.base_pose()
        return position + rotation @ np.array(self.config.lidar.xyz), rotation @ self._lidar_mount

    def _run(self) -> None:
        cfg = self.config
        world = self.world
        self._clock_origin = time.time()
        wall_start = time.monotonic()
        next_odom = next_imu = 0.0
        last_contacts: list[Any] | None = None
        scans: list[LidarScan] = []

        def on_substep() -> None:
            if self._lidar is not None and (scan := self._lidar.advance(world.time)) is not None:
                scans.append(scan)

        while not self._stop.is_set():
            with self._lock:
                fresh = time.monotonic() - self._command_time <= cfg.physics.command_timeout
                command = self._command.copy() if fresh else np.zeros(3)
            world.tick(command, on_substep)
            stamp = self._clock_origin + world.time
            with self._lock:
                self._pose = world.base_pose()
            if world.time >= next_odom:
                next_odom += 1.0 / cfg.odom_hz
                self._publish_pose(stamp)
            if world.time >= next_imu:
                next_imu += 1.0 / cfg.imu_hz
                self._publish_imu(stamp)
            contacts = self.sensor.contacts()
            if contacts != last_contacts:
                last_contacts = contacts
                self.contacts.publish(Contacts(contacts, ts=stamp))
            for scan in scans:
                self._publish_scan(scan)
            scans.clear()
            if cfg.physics.realtime:
                ahead = world.time - (time.monotonic() - wall_start)
                if ahead > 0:
                    self._stop.wait(ahead)
                elif ahead < -1.0:  # fell behind (e.g. a slow start): don't race to catch up
                    wall_start += -ahead

    def _publish_pose(self, stamp: float) -> None:
        position, rotation = self.world.base_pose()
        x, y, z, w = Rotation.from_matrix(rotation).as_quat()
        f = self.frames
        self.odom.publish(
            PoseStamped(
                ts=stamp,
                frame_id=f["world"],
                position=Vector3(*position),
                orientation=Quaternion(x, y, z, w),
            )
        )
        transforms = [
            Transform(
                translation=Vector3(*position),
                rotation=Quaternion(x, y, z, w),
                frame_id=f["world"],
                child_frame_id=f["base_link"],
                ts=stamp,
            )
        ]
        lidar, camera = self.config.lidar, self.config.camera
        for child, xyz, rotation_in_body in (
            (f["mid360_link"], lidar.xyz, mount_rotation(lidar.pitch_deg)),
            (f["camera_optical"], camera.xyz, mount_rotation(camera.pitch_deg) @ OPTICAL_IN_BODY),
        ):
            qx, qy, qz, qw = Rotation.from_matrix(rotation_in_body).as_quat()
            transforms.append(
                Transform(
                    translation=Vector3(*xyz),
                    rotation=Quaternion(qx, qy, qz, qw),
                    frame_id=f["base_link"],
                    child_frame_id=child,
                    ts=stamp,
                )
            )
        for body, parent, child, parent_frame in self._leg_edges:
            translation, rotation = body_in_parent(self.world.data, body, parent)
            qx, qy, qz, qw = Rotation.from_matrix(rotation).as_quat()
            transforms.append(
                Transform(
                    translation=Vector3(*translation),
                    rotation=Quaternion(qx, qy, qz, qw),
                    frame_id=parent_frame,
                    child_frame_id=child,
                    ts=stamp,
                )
            )
        self.tf.publish(TFMessage(*transforms))

    def _publish_imu(self, stamp: float) -> None:
        w, x, y, z = self.world.base_quat()
        self.imu.publish(
            Imu(
                angular_velocity=Vector3(*self.world.angular_velocity()),
                linear_acceleration=Vector3(*self.world.specific_force()),
                orientation=Quaternion(x, y, z, w),
                frame_id=self.frames["base_link"],
                ts=stamp,
            )
        )

    def _publish_scan(self, scan: LidarScan) -> None:
        self.lidar.publish(
            PointCloud2.from_numpy(
                scan.points,
                frame_id=self.frames["mid360_link"],
                timestamp=self._clock_origin + scan.start,
                offset_times=np.rint(scan.offsets * 1e9).astype(np.uint32),
                lines=scan.lines,
            )
        )

    def _run_camera(self, visual: str, y_up: bool) -> None:
        from dim_go2_sim.camera import SceneCamera, focal_length

        cfg = self.config.camera
        camera = SceneCamera(visual, y_up, cfg)
        mount = mount_rotation(cfg.pitch_deg)
        focal = focal_length(cfg)
        frame = self.frames["camera_optical"]
        period = 1.0 / cfg.rate_hz
        next_frame = time.monotonic()
        try:
            while not self._stop.wait(max(0.0, next_frame - time.monotonic())):
                next_frame = max(next_frame + period, time.monotonic())
                with self._lock:
                    pose = self._pose
                if pose is None:
                    continue
                position, rotation = pose
                stamp = time.time()
                eye = rotation @ mount
                rgb = camera.capture(position + rotation @ np.array(cfg.xyz), eye[:, 0], eye[:, 2])
                info = CameraInfo.from_intrinsics(
                    focal, focal, cfg.width / 2, cfg.height / 2, cfg.width, cfg.height, frame
                )
                info.ts = stamp
                self.camera_info.publish(info)
                self.color_image.publish(
                    Image(data=rgb, format=ImageFormat.RGB, frame_id=frame, ts=stamp)
                )
        finally:
            camera.close()


@dataclass(frozen=True)
class ResolvedScene:
    mjcf: Path | None
    visual: Path | None
    y_up: bool
    floor_z: float = 0.0
    spawns: tuple[tuple[float, float, float], ...] = ()  # x, y, yaw
    # What the lidar casts against in place of the MJCF's static geoms: vertices and faces.
    lidar_mesh: tuple[NDArray[np.float32], NDArray[np.uint32]] | None = None
    name: str | None = None  # names the scene's files in urdf_info; None for bare ground
    key: str = ""  # changes whenever the files it is built from do


def resolve_scene(scene: SceneConfig, cache_dir: Path | None = None) -> ResolvedScene:
    """The scene's files, from a package, a splat scan, or explicit paths; an InteriorGS home or a
    SuperSplat scan is downloaded and cooked the first time."""
    home = supersplat.scene_id(scene.package) if scene.package else None
    if scene.scan is not None or home is not None:
        if home is not None:
            scan = supersplat.load_scene(home, cache_dir, scene.scan_scale)
            name = f"supersplat_{home}"
        else:
            scan = supersplat.load_scan(scene.scan, cache_dir, scene.scan_scale or 1.0)
            name = f"scan_{scan.splats.parent.name}"
        mesh = np.load(scan.collision)
        visual = scene.visual or scan.splats
        return ResolvedScene(
            scene.mjcf or scan.mjcf,
            visual,
            False,
            scan.floor_z,
            tuple(scan.spawns),
            (mesh["vertices"], mesh["faces"]),
            name=name,
            key=_files_key(scan.collision, visual),
        )
    if scene.package is None:
        return ResolvedScene(
            scene.mjcf,
            scene.visual,
            scene.visual_y_up,
            name=scene.mjcf.stem if scene.mjcf else None,
            key=_files_key(scene.mjcf, scene.visual),
        )
    if (home := interiorgs.scene_id(scene.package)) is not None:
        cooked = interiorgs.load_scene(home, cache_dir)
        mesh = np.load(cooked.collision)
        visual = scene.visual or cooked.splats
        return ResolvedScene(
            scene.mjcf or cooked.mjcf,
            visual,
            False,
            cooked.floor_z,
            tuple(cooked.spawns),
            (mesh["vertices"], mesh["faces"]),
            name=f"interiorgs_{home}",
            key=_files_key(cooked.collision, visual),
        )
    from dimos.simulation.scenes.catalog import resolve_scene_package

    package = resolve_scene_package(scene.package)
    a = package.alignment
    if a.scale != 1 or any(a.translation) or any(a.rotation_zyx_deg):
        raise ValueError(f"{scene.package}: only a Y-up/Z-up alignment is supported")
    mjcf = scene.mjcf or package.mujoco_scene_path
    return ResolvedScene(
        mjcf,
        scene.visual or package.visual_path,
        a.y_up,
        name=scene.package.replace("/", "_"),
        key=_files_key(mjcf),
    )


def _files_key(*paths: Path | None) -> str:
    return " ".join(f"{p}:{p.stat().st_mtime_ns}" for p in paths if p is not None)
