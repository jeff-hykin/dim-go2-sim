# dim-go2-sim

A simulated Unitree Go2 for [dimOS](https://github.com/dimensionalOS/dimos), as one module:
`cmd_vel` in; a rolling Livox Mid-360, a textured front camera, IMU, odometry, tf and contacts
out. It drops into a blueprint where a real Go2 would be.

```bash
pip install dim-go2-sim
```

```python
from dimos.core.coordination.blueprints import autoconnect
from dim_go2_sim import Go2Sim

blueprint = autoconnect(
    Go2Sim.blueprint(scene={"package": "supermarket", "start": (18.5, 10.0, -0.17)}),
    MyNavigation.blueprint(),  # anything that reads lidar/odom and writes cmd_vel
)
```

With no scene it stands on bare ground. A scene is a cooked dimOS scene package by name
(`scene.package`), an [InteriorGS](#interiorgs-homes) home (`scene.package="interiorgs/839920"`),
or an MJCF of the collision geometry (`scene.mjcf`) and optionally what the camera draws
(`scene.visual`): a textured glTF, or Gaussian splats (`.nurec`).

### InteriorGS homes

```python
Go2Sim.blueprint(scene={"package": "interiorgs/839920"})  # any of the 1000 scene ids
```

A furnished home from SpatialVerse's [InteriorGS](https://huggingface.co/datasets/spatialverse/InteriorGS),
drawn by the camera from its Gaussian splats. The first run downloads its splats and collision
meshes (100-350 MB, from SAGE-3D's [USDZ](https://huggingface.co/datasets/spatialverse/SAGE-3D_InteriorGS_usdz)
and [collision mesh](https://huggingface.co/datasets/spatialverse/SAGE-3D_Collision_Mesh)
mirrors) and cooks them into `physics.cache_dir` in under a minute. The robot starts at the
most open spot on the floor, facing the longest clear view; the cooked scene keeps up to 8
such spawn poses (`dim_go2_sim.interiorgs.load_scene(id).spawns`).

## Streams

| stream | type | |
|---|---|---|
| `cmd_vel` (in) | `Twist` | held for `physics.command_timeout`, then the robot stands still |
| `lidar` | `PointCloud2` | one 0.1 s scan, in `mid360_link` where each ray fired, with per-point `offset_times` and `lines`, as the device reports it |
| `color_image`, `camera_info` | `Image`, `CameraInfo` | the nose camera, RGB |
| `imu` | `Imu` | body angular velocity, specific force and orientation |
| `odom` | `PoseStamped` | the true body pose in `world` |
| `tf` | `TFMessage` | `world` → `base_link` → `mid360_link`, `camera_optical` |
| `contacts` | `dim_go2_sim.Contacts` | (robot part, floor/clutter) pairs touching, whenever they change; ground truth for grading |

## Config

`Go2SimConfig` has a section per part, each overridable from a blueprint or the dimOS CLI:

- `scene`: `package`, `mjcf`, `visual`, `visual_y_up`, `start` (x, y, yaw; default the scene's
  first spawn), `floor_z`, `floor_top_z`
- `physics`: `timestep`, `fitted`, `actuator_tau`, `policy`, `track_velocity`,
  `command_timeout`, `max_velocity`, `realtime`, `cache_dir`
- `lidar`: `enabled`, `rate_hz`, `xyz`, `pitch_deg`, `point_rate_hz`, `downsample`,
  `min_range`, `max_range`, `noise`, `dropout`, `self_occlusion`, `motion_bins_hz`, `seed`
- `camera`: `enabled`, `rate_hz`, `width`, `height`, `fovy_deg`, `xyz`, `pitch_deg`, `near`,
  `far`, `detail_distance`, `simplify`, `max_texture_size`, `ambient`, `sun`, `background`
- `odom_hz`, `imu_hz`, `frame_prefix`

See `src/dim_go2_sim/config.py` for what each means and its default.

## How it works

- **Walking.** MuJoCo steps the Go2 under the open [rl_sar](https://github.com/fan-ziqi/rl_sar)
  robot_lab policy at 50 Hz, with joint damping, friction, armature, masses, foot friction and
  a 15 ms motor lag fitted to real Go2 runs under that policy. The policy alone ignores slow
  commands and drifts when told to stand, so an integral loop on the true body velocity trims
  its command, as the Go2's sport mode tracks it (`physics.track_velocity`).
- **Lidar.** The Mid-360's four-channel Fourier firing pattern, 200k points a second, cast in
  5 ms slices from wherever the head is then, with range noise and grazing-angle dropout. The
  static scene is cast through Embree; the robot's own body, through MuJoCo. A cooked store of
  48k geoms scans at the full 10 Hz.
- **Camera.** pygfx draws the scene's textured glTF in a child process, its meshes merged by
  material and 4 m cell (a 47k-node store: 0.6 → 60+ fps on an M-series Mac, every object
  drawn). It draws the scene only, not the robot. On a software GPU (Mesa's lavapipe,
  e.g. in Docker: install `mesa-vulkan-drivers`) it simplifies the scene and draws small objects
  only within 4 m, to keep a store at several frames a second (`camera.simplify`,
  `camera.detail_distance`).
  A store's full-size textures overflow an 8 GB GPU; `camera.max_texture_size` (e.g. 2048)
  halves larger ones until they fit.
- **Splat camera.** Gaussian splats are drawn with wgpu (Metal, Vulkan or lavapipe), sorted far
  to near each frame, each pixel's ray evaluated against the splat in 3D as 3DGUT (which made
  the InteriorGS splats) renders it, with degree-3 spherical harmonics. An InteriorGS home (1.2M
  splats) at 640x480: ~25 fps on an M-series Mac, ~45 on an RTX 5070 laptop, ~2 on lavapipe.
- **InteriorGS collisions.** MuJoCo collides only with convex shapes and the homes' meshes are
  open surfaces, so they are voxelised at 5 cm up to 2 m above the floor and the voxels merged
  into boxes (a few thousand per home); the floor is a plane. The lidar casts against the
  meshes themselves.
- The composed world is compiled once and cached (`physics.cache_dir`, default
  `~/.cache/dim_go2_sim`); a store takes about 2 minutes the first time.

## Development

```bash
uv venv --python 3.12 && uv pip install -e . pytest
pytest
```

The tests on a real InteriorGS home run once it is cooked, or with `DIM_GO2_SIM_DOWNLOAD=1`.

## Licenses

Apache-2.0. Bundled data keeps its own terms: the Go2 model (Unitree, BSD-3,
`data/go2/LICENSE`), the walking policy (rl_sar, Apache-2.0, `data/policy/NOTICE`), and the
Mid-360 pattern (`data/mid360/NOTICE.md`).

InteriorGS scenes are not bundled: they are downloaded on first use, and keep their own
license, [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (non-commercial use
only). InteriorGS is by SpatialVerse; the USDZ and collision-mesh conversions
are SAGE-3D's ([paper](https://huggingface.co/papers/2510.21307),
[code](https://github.com/Galery23/SAGE-3D_Official)).
