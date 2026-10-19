# dim-go2-sim

A simulated Unitree Go2 for [dimOS](https://github.com/dimensionalOS/dimos), as one module:
`cmd_vel` in; a rolling Livox Mid-360, a textured front camera, IMU, odometry, tf and contacts
out. It drops into a blueprint where a real Go2 would be.

## Quick start

```bash
pip install dim-go2-sim
```

[`examples/drive.py`](https://github.com/jeff-hykin/dim-go2-sim/blob/main/examples/drive.py) puts the Go2 in a scanned ranch home:

```python
import sys

from dimos.core.coordination.blueprint_config.parser import BlueprintConfigParser
from dimos.core.coordination.module_coordinator import ModuleCoordinator

from dim_go2_sim import Go2Sim

drive = Go2Sim.blueprint(scene={"package": "supersplat/3f89bbd3"}).global_config(transport="zenoh")

if __name__ == "__main__":
    ModuleCoordinator.build(drive, BlueprintConfigParser(drive).parse(sys.argv[1:])).loop()
```

```bash
python examples/drive.py
```

The first run downloads and cooks the scan (31 MB, under a minute). Then the sim publishes lidar at
10 Hz, the nose camera at ~9 fps on an M-series Mac, odom at 50 Hz and tf over zenoh, and walks
wherever `cmd_vel` says: drive it from dimOS Desktop's Controller app, or add a planner or teleop
module to the blueprint with `autoconnect`.

## Scenes

With no scene it stands on bare ground. A scene is a cooked dimOS scene package by name
(`scene.package`), an [InteriorGS](#interiorgs-homes) home (`scene.package="interiorgs/839920"`),
a [real-world splat scan](#splat-scans) (`scene.package="supersplat/3f89bbd3"`, or
`scene.scan=<a .ply>`), or an MJCF of the collision geometry (`scene.mjcf`) and optionally what
the camera draws (`scene.visual`): a textured glTF, or Gaussian splats (`.nurec`, a 3DGS `.ply`).

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

### Splat scans

```python
Go2Sim.blueprint(scene={"package": "supersplat/3f89bbd3"})  # a SuperSplat scene by id
Go2Sim.blueprint(scene={"scan": "my_home.ply", "scan_scale": 1.0})  # a 3DGS PLY or SOG meta.json
```

A Gaussian-splat scan of a real home: the camera draws its splats, and the robot and lidar collide
with geometry built from them. The default scan is [an NJ ranch home](https://superspl.at/scene/3f89bbd3)
(2.7M splats, 31 MB, downloaded from SuperSplat on first use). A scan has no collision mesh and an
arbitrary frame and scale, so the first run cooks it (a few seconds):

- **Level.** The floor, the densest layer at the bottom of the scan, is fitted by RANSAC and
  turned to z = 0; the walls (thin upright splats) are turned square to x and y.
- **Metres.** A scan's units are arbitrary: `scene.scan_scale` gives metres per unit. A known
  SuperSplat scene has a measured one (`dim_go2_sim.supersplat.SCALES`); the ranch home's is 0.92,
  from its US-standard fittings: its kitchen counters are 0.985 units high and stand 36 in
  (0.914 m), its interior door openings 2.31 units and an 80 in door's ~2.05 m (0.928 and 0.89),
  and its walls, cut off at the ceiling, then reach 2.3 m, under an 8 ft ceiling.
- **Collisions.** Opaque splats under 15 cm across are voxelised at 5 cm (two to a voxel), specks
  under 30 voxels dropped as floaters, and a column solid for 0.6 m of the metre below 2 m filled
  down to the floor (walls' feet are thin in a scan). As with InteriorGS, the voxels up to 2 m are
  merged into boxes for MuJoCo over a floor plane, and the lidar casts against the voxels' faces
  and the house's floor: one flat floor over everything the scan's walls enclose (doorways
  closed), whether or not the scan saw floor there, as a few rectangles. Spawns are picked on open
  floor.
- **Edits.** Where a scan's noise or a piece of furniture's underside blocks floor the robot
  should cross, or a wall came out ragged, a scene can list world-frame boxes whose voxels are
  emptied (cleared) or filled (added, after the clearing) before the boxes, lidar surfaces, spawns
  and shared collision meshes are built (the splats are drawn as they are); specks under 30
  voxels an edit leaves are dropped. The ranch home ships its own
  (`data/supersplat/3f89bbd3_edits.json`): rebuilt room by room, each room's scan voxels up to 2 m
  cleared and replaced by straight 10 cm wall slabs (doorways open, doors drawn open or shut as
  the scan has them, closets and rooms the scan never saw into shut) and a box or a few for each
  piece of furniture (beds solid to the floor, so the robot can't get under or into one; tables
  solid unless the robot could really walk under them), the graded plants kept as scanned, and the
  junk outside its exterior walls cleared. `scene.edits` points at another file: `{"edits":
  [{"note": ..., "cleared": [[xmin, ymin, zmin, xmax, ymax, zmax], ...], "added": [...]}, ...]}`,
  in metres.

Only a scan's degree-0 (view-independent) colour is kept: higher spherical harmonics would have
to be turned with it. Splats from a 3DGS PLY are drawn as the InteriorGS ones are, evaluated in 3D.

### People

```python
Go2Sim.blueprint(scene={"package": "supersplat/3f89bbd3"}, people={"count": 2, "speed": 0.4})
```

People walk slow loops through the scene: 1.69 m tall, 0.47 m across, made of a few capsules and
a box. Each is a kinematic (mocap) body the robot bumps into and is pushed by, and they pass
through each other and the scene; a person waits while the robot is within
`people.yield_distance` ahead of them or right beside them, and after `people.patience_s` turns back
and walks their loop the other way. The camera draws them (among a splat scene's splats, so
walls hide them, or as meshes in a glTF scene), the lidar casts them, `contacts` reports touching
one as `person`, `people` and tf give their poses (`person_<i>`), and `urdf_info` carries them.

Their loops are `people.routes` (each a list of (x, y) waypoints), or else picked by
`people.seed`: four spots 3 m apart on the open floor reachable from the robot's start, joined
by the shortest ways that keep 0.4 m from anything up to 1.6 m. Each starts at the first point
of their loop 2.5 m from the robot.

## Streams

| stream | type | |
|---|---|---|
| `cmd_vel` (in) | `Twist` | held for `physics.command_timeout`, then the robot stands still |
| `lidar` | `PointCloud2` | one 0.1 s scan, in `mid360_link` where each ray fired, with per-point `offset_times` and `lines`, as the device reports it |
| `color_image`, `camera_info` | `Image`, `CameraInfo` | the nose camera, RGB |
| `imu` | `Imu` | body angular velocity, specific force and orientation |
| `odom` | `PoseStamped` | the true body pose in `world` |
| `tf` | `TFMessage` | `world` → `base_link` → `mid360_link`, `camera_optical`, and each leg link (`FL_hip` → `FL_thigh` → `FL_calf`, ...); `world` → `person_<i>` |
| `contacts` | `dim_go2_sim.Contacts` | (robot part, floor/clutter/person) pairs touching, whenever they change; ground truth for grading |
| `people` | `PoseArray` | each person's pose in `world` (on the floor, facing their way), with odom; ground truth |
| `urdf_info` | `std_msgs.String` | the robot and its scene as one URDF, every 5 s, for [dim-controller](https://github.com/jeff-hykin/dim-controller) |
| `mesh_request` (in), `mesh_info` | `std_msgs.String`, `shape_msgs.MeshFile` | a mesh the URDF names, asked for and sent; an empty request gets `urdf_info` right away |

### The URDF

As dimOS's `UrdfManager` shares one: each mesh is named
`package://dim_go2_sim/<path>?hash=<sha256 of the file>`, and that name on `mesh_request` gets
the file on `mesh_info` (named with its current hash) unless the request ends `&have=<that
hash>`. The root link is `world`; `base_link` hangs off it by a floating joint and each link is
posed by its tf frame. The scene is fixed at `world`: `scene_mesh` is its collision surfaces,
simplified to `urdf.scene_faces` triangles in STL tiles (`scenes/<scene>/mesh_000.stl`, ...),
and for a splat scene (an InteriorGS home, a scan) `scene_splats` is its Gaussian splats as one
SPZ file (`scenes/interiorgs_<id>/splats.spz`, world frame, Z up). The files are built once per scene into
`physics.cache_dir`. `urdf.enabled=false` turns all of it, and the leg tf, off.

## Config

`Go2SimConfig` has a section per part, each overridable from a blueprint or the dimOS CLI:

- `scene`: `package`, `scan`, `scan_scale`, `edits`, `mjcf`, `visual`, `visual_y_up`, `start` (x, y, yaw;
  default the scene's first spawn), `floor_z`, `floor_top_z`
- `physics`: `timestep`, `fitted`, `actuator_tau`, `policy`, `track_velocity`,
  `command_timeout`, `max_velocity`, `realtime`, `cache_dir`
- `lidar`: `enabled`, `rate_hz`, `xyz`, `pitch_deg`, `point_rate_hz`, `downsample`,
  `min_range`, `max_range`, `noise`, `dropout`, `self_occlusion`, `motion_bins_hz`, `seed`
- `camera`: `enabled`, `rate_hz`, `width`, `height`, `fovy_deg`, `xyz`, `pitch_deg`, `near`,
  `far`, `detail_distance`, `simplify`, `max_texture_size`, `ambient`, `sun`, `background`
- `urdf`: `enabled`, `period_s`, `scene_faces`, `tile_faces`, `splats_sh_degree`
- `people`: `count`, `speed`, `seed`, `routes`, `yield_distance`, `patience_s`
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

The tests on a real InteriorGS home and the ranch scan run once they are cooked, or with
`DIM_GO2_SIM_DOWNLOAD=1`.

## Licenses

Apache-2.0. Bundled data keeps its own terms: the Go2 model (Unitree, BSD-3,
`data/go2/LICENSE`), the walking policy (rl_sar, Apache-2.0, `data/policy/NOTICE`), and the
Mid-360 pattern (`data/mid360/NOTICE.md`).

InteriorGS scenes are not bundled: they are downloaded on first use, and keep their own
license, [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (non-commercial use
only). InteriorGS is by SpatialVerse; the USDZ and collision-mesh conversions
are SAGE-3D's ([paper](https://huggingface.co/papers/2510.21307),
[code](https://github.com/Galery23/SAGE-3D_Official)).

The ranch home scan is not bundled either: it is downloaded from SuperSplat on first use.
"[Home Scan (Creation process in description)](https://superspl.at/scene/3f89bbd3)" is by
Isaiah Sweeney ([luxury_scans](https://superspl.at/user/luxury_scans)), licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The sim levels, scales and
voxelises it, and keeps only its degree-0 colour.
