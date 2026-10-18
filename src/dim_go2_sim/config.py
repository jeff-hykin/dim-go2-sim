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

"""Everything about a Go2Sim run: the scene, the physics, and each sensor (no dimOS imports:
the camera's child process loads this)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class SceneConfig(BaseModel):
    # A cooked dimOS scene package by name (resolved with dimOS's catalog), or explicit files.
    package: str | None = None
    mjcf: Path | None = None  # collision geometry; a robot-free MJCF
    visual: Path | None = None  # textured glTF/GLB the camera draws; None draws nothing
    visual_y_up: bool = True  # glTF's convention; the world is Z-up
    start: tuple[float, float, float] = (0.0, 0.0, 0.0)  # x, y, yaw of the robot
    floor_z: float = 0.0  # height the feet start on
    # Scene geometry whose top is below this counts as floor for contacts; the rest is clutter.
    floor_top_z: float = 0.05


class PhysicsConfig(BaseModel):
    timestep: float = 0.0025  # seconds per MuJoCo step; the policy runs every 0.02 s
    # The physics were fitted to real Go2 runs under the bundled policy (go2-sim, dimOS #4441).
    fitted: bool = True
    actuator_tau: float = 0.0151  # seconds of first-order lag between a torque and the joint
    policy: Path | None = None  # an rl_sar-layout ONNX walking policy; None is the bundled one
    # Trim the policy's command closed-loop on the true body velocity, as a Go2's sport mode
    # does: the policy alone ignores slow commands and drifts when told to stand.
    track_velocity: bool = True
    command_timeout: float = 0.5  # seconds a cmd_vel holds before the robot stands still
    max_velocity: tuple[float, float, float] = (1.0, 0.5, 1.5)  # vx, vy m/s and wz rad/s
    realtime: bool = True  # pace to the wall clock; False runs as fast as it can
    cache_dir: Path | None = None  # compiled worlds; None is the user cache directory


class LidarConfig(BaseModel):
    enabled: bool = True
    rate_hz: float = 10.0
    # The Mid-360 on a Go2's head, from the body origin, pitched nose-down.
    xyz: tuple[float, float, float] = (0.29515, -0.00003, 0.16297)
    pitch_deg: float = 60.0
    point_rate_hz: int = 200_000
    downsample: int = 1
    min_range: float = 0.16
    max_range: float = 40.0
    noise: bool = True
    dropout: bool = True
    self_occlusion: bool = True  # the robot's own body blocks rays
    motion_bins_hz: float = 200.0  # how often a scan's origin follows the moving body
    seed: int = 0


class CameraConfig(BaseModel):
    enabled: bool = True
    rate_hz: float = 10.0
    width: int = 640
    height: int = 480
    fovy_deg: float = 60.0
    # The Go2's nose camera, from the body origin, looking straight ahead.
    xyz: tuple[float, float, float] = (0.32715, -0.00003, 0.04297)
    pitch_deg: float = 0.0
    near: float = 0.05
    far: float = 100.0
    # Small objects (products, signs) are drawn only within this many metres, and this share of
    # the scene's triangles is simplified away: tens of millions of vertices are too many for a
    # software GPU (e.g. Docker's lavapipe). None draws everything, exactly, on a real GPU, and
    # 4 m and 0.95 on a software one.
    detail_distance: float | None = None
    simplify: float | None = None
    # Textures wider or taller than this many pixels are halved, by averaging, until they fit:
    # a store's full-size textures need more memory than an 8 GB GPU has. None keeps them whole.
    max_texture_size: int | None = Field(default=None, ge=1)
    ambient: float = 1.8
    sun: float = 2.5
    background: str = "#1b2230"
