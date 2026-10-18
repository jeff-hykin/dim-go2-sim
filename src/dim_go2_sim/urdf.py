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

"""The Go2 and its scene as one URDF, shared as dimOS's UrdfManager shares one (for dim-controller).

The XML goes out on urdf_info, each mesh named package://dim_go2_sim/<path>?hash=<sha256 of the
file>. A mesh_request for one of those names is answered on mesh_info with the file, named with
its current hash, unless the request carries &have=<that hash>: the asker's copy is fresh.

The root is world; base_link hangs off it by a floating joint, posed by tf, as is every leg link.
The scene is fixed at world: its collision surfaces as STL tiles and, for an InteriorGS home, its
Gaussian splats as one SPZ file.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import struct
import time
from urllib.parse import parse_qs
import xml.etree.ElementTree as ET

from dimos.types.timestamped import Timestamped
import mujoco
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from dim_go2_sim.config import UrdfConfig

PACKAGE = "dim_go2_sim"
# Bump when the files change, so cached ones are rebuilt.
ASSETS_VERSION = 1
SCENE_RGBA = (0.72, 0.72, 0.75, 1.0)
_MESH_HEADER = struct.Struct("<dI")
_STL_TRIANGLE = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])

Mesh = tuple[NDArray[np.float32], NDArray[np.uint32]]  # vertices, triangles


class MeshFile(Timestamped):
    """One mesh file's bytes, named by the URI a URDF uses for it (dimOS's shape_msgs.MeshFile)."""

    msg_name = "shape_msgs.MeshFile"

    def __init__(self, uri: str, data: bytes, ts: float | None = None) -> None:
        self.uri = uri
        self.data = data
        self.ts = time.time() if ts is None else ts

    def lcm_encode(self) -> bytes:
        uri = self.uri.encode()
        return _MESH_HEADER.pack(self.ts, len(uri)) + uri + self.data

    @classmethod
    def lcm_decode(cls, data: bytes, **kwargs: object) -> MeshFile:
        ts, uri_length = _MESH_HEADER.unpack_from(data)
        start = _MESH_HEADER.size + uri_length
        return cls(data[_MESH_HEADER.size : start].decode(), bytes(data[start:]), ts)


@dataclass(frozen=True)
class SharedUrdf:
    xml: str
    files: dict[str, tuple[Path, str]]  # mesh name without its query -> the file and its sha256

    def answer(self, request: str) -> MeshFile | None:
        """The file a mesh_request asks for, None if the asker has it already; KeyError for a name
        the URDF doesn't use: a request is never a path into the filesystem."""
        name, _, query = request.partition("?")
        path, digest = self.files[name]
        if parse_qs(query).get("have", [""])[0] == digest:
            return None
        return MeshFile(f"{name}?hash={digest}", path.read_bytes())


@dataclass(frozen=True)
class SceneSource:
    name: str  # names its files: package://dim_go2_sim/scenes/<name>/...
    key: str  # changes whenever what it is built from does
    mesh: Callable[[], Mesh]  # world-frame collision surfaces, called only on a cache miss
    splats: Path | None = None  # Gaussian splats: a NuRec container or a 3DGS PLY


def share_urdf(
    robot: mujoco.MjModel,
    prefix: str,
    scene: SceneSource | None,
    config: UrdfConfig,
    cache_dir: Path | None = None,
) -> SharedUrdf:
    """The URDF of *robot* (the Go2 alone) and *scene*, each link named as its tf frame (after
    *prefix*), its files built once into the cache."""
    cache = cache_dir or Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    root = cache / PACKAGE / "urdf"
    tag = f"{ASSETS_VERSION} {version('dim-go2-sim')}"
    files: dict[str, Path] = {}
    robot_dir = _cached(root / f"go2-{_digest(tag)}", lambda folder: _write_robot(robot, folder))
    for path in sorted(robot_dir.glob("*.stl")):
        files[f"package://{PACKAGE}/go2/{path.name}"] = path
    xml = ET.Element("robot", name="go2")
    world = prefix + "world"
    ET.SubElement(xml, "link", name=world)
    _robot_links(xml, robot, prefix, json.loads((robot_dir / "parts.json").read_text()))
    if scene is not None:
        settings = (config.scene_faces, config.tile_faces, config.splats_sh_degree)
        scene_dir = _cached(
            root / f"{scene.name}-{_digest(f'{tag} {scene.key} {settings}')}",
            lambda folder: _write_scene(scene, config, folder),
        )
        package = f"package://{PACKAGE}/scenes/{scene.name}"
        meshes = {f"{package}/{p.name}": p for p in sorted(scene_dir.glob("mesh_*.stl"))}
        files.update(meshes)
        _fixed_link(xml, world, prefix + "scene_mesh", list(meshes))
        if (scene_dir / "splats.spz").is_file():
            files[f"{package}/splats.spz"] = scene_dir / "splats.spz"
            _fixed_link(xml, world, prefix + "scene_splats", [f"{package}/splats.spz"])
    hashed = {name: (path, _file_digest(path)) for name, path in files.items()}
    for mesh in xml.iter("mesh"):
        name = mesh.attrib["filename"]
        mesh.set("filename", f"{name}?hash={hashed[name][1]}")
    return SharedUrdf(ET.tostring(xml, encoding="unicode"), hashed)


def leg_bodies(model: mujoco.MjModel, trunk: int) -> list[tuple[int, int]]:
    """(body, parent) for every body below *trunk*, parents first: each one's tf edge, named as
    the body (base_link for the trunk)."""
    below = {trunk}
    edges = []
    for body in range(trunk + 1, model.nbody):
        if int(model.body_parentid[body]) in below:
            below.add(body)
            edges.append((body, int(model.body_parentid[body])))
    return edges


def body_in_parent(
    data: mujoco.MjData, body: int, parent: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """*body*'s position and rotation matrix in *parent*'s frame."""
    parent_rotation = data.xmat[parent].reshape(3, 3)
    position = parent_rotation.T @ (data.xpos[body] - data.xpos[parent])
    return position, parent_rotation.T @ data.xmat[body].reshape(3, 3)


def _cached(folder: Path, write: Callable[[Path], None]) -> Path:
    if not (folder / "done").is_file():
        partial = folder.with_name(f"{folder.name}.{os.getpid()}.tmp")
        partial.mkdir(parents=True, exist_ok=True)
        write(partial)
        (partial / "done").touch()
        partial.rename(folder)
    return folder


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1 << 22):
            digest.update(chunk)
    return digest.hexdigest()


def _rpy(quat_wxyz: NDArray[np.float64]) -> str:
    w, x, y, z = quat_wxyz
    return " ".join(f"{a:.6g}" for a in Rotation.from_quat([x, y, z, w]).as_euler("xyz"))


def _xyz(values: NDArray[np.float64]) -> str:
    return " ".join(f"{v:.6g}" for v in values)


def _robot_links(
    xml: ET.Element, model: mujoco.MjModel, prefix: str, parts: dict[str, dict[str, object]]
) -> None:
    trunk = model.body("base").id
    names = {trunk: prefix + "base_link"}
    for body, _ in leg_bodies(model, trunk):
        names[body] = prefix + model.body(body).name
    for body, link in names.items():
        element = ET.SubElement(xml, "link", name=link)
        for file, part in parts.items():
            if part["body"] == model.body(body).name:
                visual = ET.SubElement(element, "visual")
                geometry = ET.SubElement(visual, "geometry")
                ET.SubElement(geometry, "mesh", filename=f"package://{PACKAGE}/go2/{file}")
                material = ET.SubElement(visual, "material", name=file.removesuffix(".stl"))
                ET.SubElement(material, "color", rgba=_xyz(np.asarray(part["rgba"])))
    joint = ET.SubElement(xml, "joint", name="base_joint", type="floating")
    ET.SubElement(joint, "parent", link=prefix + "world")
    ET.SubElement(joint, "child", link=names[trunk])
    for body, parent in leg_bodies(model, trunk):
        index = int(model.body_jntadr[body])
        joint = ET.SubElement(xml, "joint", name=model.joint(index).name, type="revolute")
        ET.SubElement(joint, "parent", link=names[parent])
        ET.SubElement(joint, "child", link=names[body])
        pose = {"xyz": _xyz(model.body_pos[body]), "rpy": _rpy(model.body_quat[body])}
        ET.SubElement(joint, "origin", **pose)
        ET.SubElement(joint, "axis", xyz=_xyz(model.jnt_axis[index]))
        lower, upper = (f"{limit:.6g}" for limit in model.jnt_range[index])
        ET.SubElement(joint, "limit", lower=lower, upper=upper, effort="0", velocity="0")


def _fixed_link(xml: ET.Element, world: str, name: str, meshes: list[str]) -> None:
    link = ET.SubElement(xml, "link", name=name)
    for uri in meshes:
        visual = ET.SubElement(link, "visual")
        ET.SubElement(ET.SubElement(visual, "geometry"), "mesh", filename=uri)
        material = ET.SubElement(visual, "material", name="scene")
        ET.SubElement(material, "color", rgba=_xyz(np.asarray(SCENE_RGBA)))
    joint = ET.SubElement(xml, "joint", name=f"{name}_joint", type="fixed")
    ET.SubElement(joint, "parent", link=world)
    ET.SubElement(joint, "child", link=name)


def _write_robot(model: mujoco.MjModel, folder: Path) -> None:
    """One STL per body and material: its visual meshes, merged in the body's frame."""
    parts: dict[tuple[int, int], list[Mesh]] = {}
    for geom in range(model.ngeom):
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH or model.geom_group[geom] != 2:
            continue
        mesh = model.geom_dataid[geom]
        v0, nv = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        f0, nf = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, model.geom_quat[geom])
        vertices = model.mesh_vert[v0 : v0 + nv] @ rotation.reshape(3, 3).T + model.geom_pos[geom]
        key = (int(model.geom_bodyid[geom]), int(model.geom_matid[geom]))
        parts.setdefault(key, []).append((vertices, model.mesh_face[f0 : f0 + nf]))
    files = {}
    for (body, material), meshes in parts.items():
        name = model.body(body).name
        file = f"{name}_{model.material(material).name}.stl"
        # full detail: automatic decimation tears thin parts into spikes
        (folder / file).write_bytes(stl(*merged(meshes)))
        files[file] = {"body": name, "rgba": model.mat_rgba[material].round(4).tolist()}
    (folder / "parts.json").write_text(json.dumps(files))


def _write_scene(scene: SceneSource, config: UrdfConfig, folder: Path) -> None:
    vertices, faces = simplified(*scene.mesh(), config.scene_faces)
    for i, tile in enumerate(tiles(vertices, faces, config.tile_faces)):
        (folder / f"mesh_{i:03d}.stl").write_bytes(stl(*tile))
    if scene.splats is not None:
        from dim_go2_sim.splat import load_splats, spz

        (folder / "splats.spz").write_bytes(spz(load_splats(scene.splats), config.splats_sh_degree))


def merged(meshes: list[Mesh]) -> Mesh:
    offsets = np.cumsum([0] + [len(v) for v, _ in meshes[:-1]])
    faces = [f + offset for (_, f), offset in zip(meshes, offsets, strict=True)]
    return np.concatenate([v for v, _ in meshes]).astype(np.float32), np.concatenate(faces)


def simplified(vertices: NDArray[np.floating], faces: NDArray[np.integer], max_faces: int) -> Mesh:
    """Quadric decimation to about *max_faces* triangles, if there are more."""
    if len(faces) <= max_faces:
        return vertices.astype(np.float32), faces.astype(np.uint32)
    import fast_simplification  # type: ignore[import-untyped]

    points, kept = fast_simplification.simplify(
        vertices.astype(np.float64),
        faces.astype(np.int64),
        target_reduction=1 - max_faces / len(faces),
    )
    return points.astype(np.float32), kept.astype(np.uint32).reshape(-1, 3)


def tiles(vertices: NDArray[np.float32], faces: NDArray[np.uint32], max_faces: int) -> list[Mesh]:
    """The mesh cut into pieces of at most *max_faces* triangles, by halving the widest side."""
    centroids = vertices[faces].mean(axis=1)
    pending, pieces = [np.arange(len(faces))], []
    while pending:
        group = pending.pop()
        if len(group) <= max_faces:
            pieces.append(group)
            continue
        points = centroids[group, :2]
        axis = int(np.argmax(points.max(0) - points.min(0)))
        order = group[np.argsort(points[:, axis], kind="stable")]
        pending += [order[len(order) // 2 :], order[: len(order) // 2]]
    result = []
    for group in pieces:
        used, local = np.unique(faces[group], return_inverse=True)
        result.append((vertices[used], local.reshape(-1, 3).astype(np.uint32)))
    return result


def stl(vertices: NDArray[np.float32], faces: NDArray[np.uint32]) -> bytes:
    """Binary STL."""
    corners = vertices[faces].astype(np.float32)
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    triangles = np.zeros(len(faces), _STL_TRIANGLE)
    triangles["normal"] = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)
    triangles["vertices"] = corners
    return b"\0" * 80 + struct.pack("<I", len(faces)) + triangles.tobytes()
