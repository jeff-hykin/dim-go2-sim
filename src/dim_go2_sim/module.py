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
import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from reactivex.disposable import Disposable
from scipy.spatial.transform import Rotation

from dim_go2_sim.config import CameraConfig, LidarConfig, PhysicsConfig, SceneConfig
from dim_go2_sim.contacts import Contacts, ContactSensor
from dim_go2_sim.lidar import LidarScan, Raycaster, RollingLidar, mount_rotation
from dim_go2_sim.world import Go2World, compose, load_world

# The optical frame's axes (z forward, x right, y down) as columns, in a forward-looking body frame.
OPTICAL_IN_BODY = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


class Go2SimConfig(ModuleConfig):
    scene: SceneConfig = Field(default_factory=SceneConfig)
    physics: PhysicsConfig = Field(default_factory=PhysicsConfig)
    lidar: LidarConfig = Field(default_factory=LidarConfig)
    camera: CameraConfig = Field(default_factory=CameraConfig)
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

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._command = np.zeros(3)
        self._command_time = -math.inf
        self._pose: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None
        self._clock_origin = 0.0

    @rpc
    def start(self) -> None:
        super().start()
        cfg = self.config
        mjcf, visual, y_up = _resolve_scene(cfg.scene)
        self.world = Go2World(load_world(mjcf, cfg.physics), cfg.physics)
        x, y, yaw = cfg.scene.start
        self.world.reset(x, y, yaw, cfg.scene.floor_z)
        model, data = self.world.model, self.world.data
        robot_bodies = {b for b in range(model.nbody) if self.world.is_robot(b)}
        self.sensor = ContactSensor(
            model, data, robot_bodies, self.world.trunk, self.world.feet, cfg.scene.floor_top_z
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
            )
            self._lidar_mount = mount_rotation(cfg.lidar.pitch_deg)
            self._lidar = RollingLidar(cfg.lidar, raycaster, self._lidar_pose)
        self.register_disposable(Disposable(self.cmd_vel.subscribe(self._on_cmd_vel)))
        self._stop.clear()
        self._threads = [threading.Thread(target=self._run, name="go2-sim", daemon=True)]
        if cfg.camera.enabled and visual is not None:
            self._threads.append(
                threading.Thread(target=self._run_camera, args=(str(visual), y_up), daemon=True)
            )
        for thread in self._threads:
            thread.start()

    @rpc
    def stop(self) -> None:
        self._stop.set()
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


def _resolve_scene(scene: SceneConfig) -> tuple[Path | None, Path | None, bool]:
    """The scene's MJCF, its visual, and whether that is Y-up; from a package or explicit paths."""
    if scene.package is None:
        return scene.mjcf, scene.visual, scene.visual_y_up
    from dimos.simulation.scenes.catalog import resolve_scene_package

    package = resolve_scene_package(scene.package)
    a = package.alignment
    if a.scale != 1 or any(a.translation) or any(a.rotation_zyx_deg):
        raise ValueError(f"{scene.package}: only a Y-up/Z-up alignment is supported")
    return (
        scene.mjcf or package.mujoco_scene_path,
        scene.visual or package.visual_path,
        a.y_up,
    )
