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
    # A cooked dimOS scene package by name (resolved with dimOS's catalog), an InteriorGS home
    # as "interiorgs/<id>" or a SuperSplat scan as "supersplat/<id>" (downloaded on first use),
    # or explicit files.
    package: str | None = None
    # A real-world Gaussian-splat scan (a 3DGS .ply, or a SOG's meta.json), levelled and given
    # collisions from its splats the first time, as a SuperSplat one is.
    scan: Path | None = None
    # Metres per unit of the scan; None is the scale measured for a known SuperSplat scene, else 1.
    scan_scale: float | None = Field(default=None, gt=0.0)
    # A scan's edits: a JSON file {"edits": [{"note": ..., "cleared": [[xmin, ymin, zmin, xmax,
    # ymax, zmax], ...], "added": [...]}, ...]} of world-frame boxes (metres) emptied of, or
    # filled with, collision voxels; None is the edits shipped for a known SuperSplat scene.
    edits: Path | None = None
    mjcf: Path | None = None  # collision geometry; a robot-free MJCF
    # What the camera draws: a textured glTF/GLB, or Gaussian splats (.nurec, or a 3DGS .ply);
    # None draws nothing.
    visual: Path | None = None
    visual_y_up: bool = True  # glTF's convention; the world is Z-up
    # x, y, yaw of the robot; None is the scene's first spawn pose, or the origin.
    start: tuple[float, float, float] | None = None
    floor_z: float | None = None  # height the feet start on; None is the scene's floor, or 0
    # Scene geometry whose top is below this counts as floor for contacts; the rest is clutter.
    # None is 5 cm above floor_z.
    floor_top_z: float | None = None


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


class PeopleConfig(BaseModel):
    # People walking slow loops on open floor: kinematic bodies the robot collides with, drawn by
    # the camera, cast by the lidar, posed on the people stream, in robot_description and on tf.
    count: int = Field(default=0, ge=0)
    speed: float = Field(default=0.4, gt=0.0)  # m/s
    seed: int = 0  # picks the loops, when routes doesn't
    # Each person's loop, as (x, y) waypoints walked in straight lines; None picks loops through
    # the open floor reachable from the robot's start.
    routes: list[list[tuple[float, float]]] | None = None
    # A person waits while the robot's body is this close, metres, in the way ahead of them (or
    # right beside them), and after patience_s turns back and walks their loop the other way.
    yield_distance: float = 1.0
    patience_s: float = Field(default=3.0, gt=0.0)


class UrdfConfig(BaseModel):
    # The robot and its scene as one URDF on robot_description, each file on file_request ->
    # file_info, as dimOS's UrdfManager shares one (for dim-controller); the leg links go on tf too.
    enabled: bool = True
    period_s: float = Field(default=5.0, gt=0.0)  # robot_description is re-sent; nothing is latched
    # The scene's surfaces are simplified to about this many triangles, in STL files of at most
    # tile_faces (50 bytes each).
    scene_faces: int = Field(default=1_000_000, ge=1)
    tile_faces: int = Field(default=100_000, ge=1)
    # Spherical-harmonic degree of an InteriorGS home's splats, shared as one SPZ file.
    splats_sh_degree: int = Field(default=3, ge=0, le=3)
