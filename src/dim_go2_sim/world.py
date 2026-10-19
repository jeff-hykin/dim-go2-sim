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

"""A scene with the Go2 in it, compiled once and cached, stepped one policy tick at a time."""

from __future__ import annotations

from collections.abc import Callable
import hashlib
from importlib.metadata import version
from importlib.resources import files
import math
import os
from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import NDArray

from dim_go2_sim.config import CameraConfig, PhysicsConfig
from dim_go2_sim.people import BODY, People, add_people, collide_with_people
from dim_go2_sim.policy import (
    COMMAND_SLEW,
    CONTROL_DT,
    DEFAULT_POSE,
    JOINT_NAMES,
    KD,
    KP,
    LEGS,
    OnnxPolicy,
    VelocityTracker,
)

PREFIX = "go2/"
ROOT = "base"
# go2-sim's fit to real Go2 runs under the bundled policy.
FITTED = {
    "armature": 0.00712,
    "damping": 0.2850,
    "frictionloss": 0.3650,
    "trunk_mass_scale": 0.9412,
    "trunk_inertia_scale": 0.8601,
    "foot_friction": 0.7860,
    "foot_friction_torsional": 0.006134,
    "trunk_com_x": -0.006845,
    "leg_mass_scale": 1.616,
}
# Bump when the composed model changes, so cached worlds are rebuilt.
MODEL_VERSION = 3
# A sphere around the nose camera keeps the scene out of its near plane, whose farthest corner
# is 1.4 near planes away (fovy 60, 4:3), plus what a contact sinks in.
LENS_GUARD = "lens_guard"
LENS_GUARD_RADIUS = 1.4 * CameraConfig().near + 0.02
LENS_GUARD_GROUP = 5  # the lidar's self-occlusion leaves it out


def go2_spec(fitted: bool) -> mujoco.MjSpec:
    model_dir = Path(str(files("dim_go2_sim") / "data/go2"))
    spec = mujoco.MjSpec.from_file(str(model_dir / "go2.xml"))
    spec.meshdir = str(model_dir / "assets")
    for key in list(spec.keys):
        spec.delete(key)
    spec.body(ROOT).add_geom(
        name=LENS_GUARD,
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[LENS_GUARD_RADIUS, 0, 0],
        pos=CameraConfig().xyz,
        group=LENS_GUARD_GROUP,
        condim=1,  # slides along a wall rather than catching on it
        rgba=[0, 0, 0, 0],
    )
    collide_with_people(spec)
    if not fitted:
        return spec
    p = FITTED
    for name in JOINT_NAMES:
        joint = spec.joint(name)
        joint.armature, joint.frictionloss = p["armature"], p["frictionloss"]
        joint.damping = [p["damping"], 0.0, 0.0]
    trunk = spec.body(ROOT)
    trunk.mass *= p["trunk_mass_scale"]
    trunk.inertia = np.asarray(trunk.inertia) * p["trunk_inertia_scale"]
    trunk.ipos = np.asarray(trunk.ipos) + np.array([p["trunk_com_x"], 0.0, 0.0])
    for leg in LEGS:
        for part in ("thigh", "calf"):
            body = spec.body(f"{leg}_{part}")
            body.mass *= p["leg_mass_scale"]
            body.inertia = np.asarray(body.inertia) * p["leg_mass_scale"]
        foot = spec.geom(leg)
        foot.friction = [p["foot_friction"], p["foot_friction_torsional"], foot.friction[2]]
    return spec


def compose(
    scene_mjcf: Path | None, physics: PhysicsConfig, *, ground: bool = True, people: int = 0
) -> mujoco.MjModel:
    """The scene (or bare ground, or nothing) with the Go2 under PREFIX and *people* people, in
    the fitted options."""
    if scene_mjcf:
        world = mujoco.MjSpec.from_file(str(scene_mjcf))
    else:
        world = _ground() if ground else mujoco.MjSpec()
    add_people(world, people)
    world.option.timestep = physics.timestep
    world.option.integrator = mujoco.mjtIntegrator.mjINT_EULER
    world.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    world.option.impratio = 100.0
    robot = go2_spec(physics.fitted)
    robot.option.integrator = world.option.integrator
    robot.option.cone = world.option.cone
    robot.option.impratio = world.option.impratio
    world.attach(robot, prefix=PREFIX, frame=world.worldbody.add_frame())
    return world.compile()


def _ground() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05])
    return spec


def load_world(scene_mjcf: Path | None, physics: PhysicsConfig, people: int = 0) -> mujoco.MjModel:
    """compose(), cached by everything it is built from: a cooked store takes minutes."""
    parts = [
        f"people {people}".encode(),
        scene_mjcf.read_bytes() if scene_mjcf else b"ground",
        str(scene_mjcf.stat().st_mtime_ns if scene_mjcf else 0).encode(),
        physics.model_dump_json(include={"timestep", "fitted"}).encode(),
        f"{mujoco.__version__} {version('dim-go2-sim')} {MODEL_VERSION}".encode(),
    ]
    key = hashlib.sha256(b"\0".join(parts)).hexdigest()[:16]
    cache = physics.cache_dir or Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    path = cache / "dim_go2_sim" / f"{scene_mjcf.stem if scene_mjcf else 'ground'}-{key}.mjb"
    if path.is_file():
        return mujoco.MjModel.from_binary_path(str(path))
    model = compose(scene_mjcf, physics, people=people)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(f".{os.getpid()}.tmp")
    mujoco.mj_saveModel(model, str(partial), None)
    partial.rename(path)
    return model


def yaw_quat(yaw: float) -> NDArray[np.float64]:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def projected_gravity(quat_wxyz: NDArray[np.float64]) -> NDArray[np.float64]:
    w, x, y, z = quat_wxyz
    return np.array([-2 * (x * z - w * y), -2 * (y * z + w * x), -(1 - 2 * (x * x + y * y))])


class Go2World:
    """The compiled world and the Go2 walking in it under the policy, from a velocity command,
    with any people in it walking their routes (set *people* before reset)."""

    def __init__(self, model: mujoco.MjModel, physics: PhysicsConfig) -> None:
        self.model = model
        self.people: People | None = None
        self.data = mujoco.MjData(model)
        self.physics = physics
        self.policy = OnnxPolicy(physics.policy)
        self.tracker = VelocityTracker() if physics.track_velocity else None
        joints = [model.joint(PREFIX + name).id for name in JOINT_NAMES]
        self._qpos = model.jnt_qposadr[joints]
        self._dofs = model.jnt_dofadr[joints]
        actuator_of = {int(model.actuator_trnid[a, 0]): a for a in range(model.nu)}
        self._actuators = np.array([actuator_of[j] for j in joints])
        self._torque_limit = model.actuator_ctrlrange[self._actuators, 1]
        self.trunk = model.body(PREFIX + ROOT).id
        free = model.body_jntadr[self.trunk]  # the trunk's free joint
        self._base_qpos = model.jnt_qposadr[free]
        self._base_dof = model.jnt_dofadr[free]
        self.feet = tuple(model.geom(PREFIX + leg).id for leg in LEGS)
        self.substeps = max(1, round(CONTROL_DT / model.opt.timestep))
        self.time = 0.0
        self._people_mocap: list[int] = []
        while (
            body := mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, BODY.format(len(self._people_mocap))
            )
        ) >= 0:
            self._people_mocap.append(int(model.body_mocapid[body]))

    def reset(self, x: float, y: float, yaw: float, floor_z: float) -> None:
        """Stand in the policy's default pose with the feet on floor_z, at rest."""
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        q = self._base_qpos
        d.qpos[q : q + 3] = [x, y, 0.5]
        d.qpos[q + 3 : q + 7] = yaw_quat(yaw)
        d.qpos[self._qpos] = DEFAULT_POSE
        if self.people is not None:
            self.people.reset()
            self._place_people()
        mujoco.mj_forward(m, d)
        lowest = min(float(d.geom_xpos[g][2] - m.geom_size[g][0]) for g in self.feet)
        d.qpos[q + 2] += floor_z - lowest
        mujoco.mj_forward(m, d)
        self._command = np.zeros(3)
        self._applied = np.zeros(len(JOINT_NAMES))
        self.policy.reset()
        if self.tracker is not None:
            self.tracker.reset()
        self.time = 0.0

    def tick(
        self, command: NDArray[np.float64], on_substep: Callable[[], None] | None = None
    ) -> None:
        """Advance one policy tick (CONTROL_DT) toward a (vx, vy, wz) command."""
        limit = np.array(self.physics.max_velocity)
        command = np.clip(command, -limit, limit)
        self._command += np.clip(command - self._command, -COMMAND_SLEW, COMMAND_SLEW)
        walk = self._command
        if self.tracker is not None:
            walk = self.tracker.update(self._command, self.body_velocity(), CONTROL_DT)
        d = self.data
        q, v = self._base_qpos, self._base_dof
        targets = self.policy.act(
            d.qvel[v + 3 : v + 6].copy(),
            projected_gravity(d.qpos[q + 3 : q + 7]),
            d.qpos[self._qpos].copy(),
            d.qvel[self._dofs].copy(),
            walk,
        )
        if self.people is not None:
            self.people.step(CONTROL_DT, d.qpos[q : q + 2])
            self._place_people()
        dt = self.model.opt.timestep
        tau = self.physics.actuator_tau
        alpha = 1.0 if tau <= 0 else dt / (tau + dt)
        for _ in range(self.substeps):
            torque = KP * (targets - d.qpos[self._qpos]) - KD * d.qvel[self._dofs]
            torque = np.clip(torque, -self._torque_limit, self._torque_limit)
            self._applied += alpha * (torque - self._applied)
            d.ctrl[self._actuators] = self._applied
            mujoco.mj_step(self.model, d)
            self.time += dt
            if on_substep is not None:
                on_substep()

    def _place_people(self) -> None:
        assert self.people is not None
        poses = self.people.poses()
        if len(poses) != len(self._people_mocap):
            raise ValueError(
                f"{len(poses)} routes for {len(self._people_mocap)} people in the model"
            )
        for mocap, (x, y, yaw) in zip(self._people_mocap, poses, strict=True):
            self.data.mocap_pos[mocap] = [x, y, 0.0]
            self.data.mocap_quat[mocap] = yaw_quat(yaw)

    def base_pose(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Position and rotation matrix of the body in the world."""
        return self.data.xpos[self.trunk].copy(), self.data.xmat[self.trunk].reshape(3, 3).copy()

    def base_quat(self) -> NDArray[np.float64]:
        """The body's orientation, w x y z."""
        q = self._base_qpos
        quat: NDArray[np.float64] = self.data.qpos[q + 3 : q + 7].copy()
        return quat

    def body_velocity(self) -> NDArray[np.float64]:
        """(vx, vy) in the heading frame and the yaw rate, as a velocity command means them."""
        v = self._base_dof
        linear, angular = self.data.qvel[v : v + 3], self.data.qvel[v + 3 : v + 6]
        _, rotation = self.base_pose()
        heading = math.atan2(rotation[1, 0], rotation[0, 0])
        c, s = math.cos(heading), math.sin(heading)
        return np.array(
            [c * linear[0] + s * linear[1], -s * linear[0] + c * linear[1], (rotation @ angular)[2]]
        )

    def angular_velocity(self) -> NDArray[np.float64]:
        v = self._base_dof
        gyro: NDArray[np.float64] = self.data.qvel[v + 3 : v + 6].copy()
        return gyro

    def specific_force(self) -> NDArray[np.float64]:
        """What an accelerometer on the body reads: acceleration minus gravity, body frame."""
        v = self._base_dof
        _, rotation = self.base_pose()
        world = self.data.qacc[v : v + 3] - self.model.opt.gravity
        force: NDArray[np.float64] = rotation.T @ world
        return force

    def is_robot(self, body: int) -> bool:
        return bool(self.model.body(body).name.startswith(PREFIX))
