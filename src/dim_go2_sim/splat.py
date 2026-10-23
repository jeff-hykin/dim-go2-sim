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

"""Camera frames drawn from a 3D Gaussian splat with wgpu (Metal, Vulkan, or lavapipe).

Each frame the splats in view are sorted far to near on the GPU, then drawn as instanced quads
over their projected ellipses (EWA, as 3DGS) and blended over the ones behind. Within a quad,
each pixel's ray is evaluated against the splat in 3D, as 3DGUT (which trained the InteriorGS
splats) does: a linearised projection smears splats near the camera across the image. Colour is
degree-3 spherical harmonics, evaluated per view on the GPU.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import gzip
import math
from pathlib import Path
import struct
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dim_go2_sim.config import CameraConfig
from dim_go2_sim.people import splats as person_splats
from dim_go2_sim.people import yaw_rotation

# A splat's quad reaches this many standard deviations, where 3DGS stops drawing it.
QUAD_SIGMAS = 3.0
# 3DGUT, which trained the InteriorGS splats, draws none thinner than this many pixels. (It also
# skips any nearer than 0.2 m; here only the camera's near plane does, as each pixel's ray is
# evaluated in 3D, nothing near smears, and a wall 0.2 m off would vanish.)
MIN_PIXEL_RADIUS = 0.5477
SH_COEFFICIENTS = 16  # degree 3
# Depth keys are 16-bit, so the GPU sorts them by counting: ~0.2 mm steps across a house.
DEPTH_LEVELS = 65535
SH_C0 = 0.28209479177387814  # degree 0's basis: colour = 0.5 + SH_C0 * coefficient
FRAMES_IN_FLIGHT = 2  # one read back while the next is drawn (wgpu polls wait on all)
PERSON_SIGMA = 0.012  # metres: a person's splats, 2 cm apart, close their surfaces

# What the render and sort shaders share: the camera, and each splat's footprint on the image.
COMMON = """
struct Camera {
    rotation: mat4x4<f32>,  // world to optical (x right, y down, z forward), as columns
    position: vec4<f32>,
    focal: vec2<f32>,
    size: vec2<f32>,
    near: f32,
    far: f32,
    count: u32,  // splats
};

// Per splat, 8 words: position (3 x f32), then f16 pairs: (w, x) (y, z) of the rotation,
// (log sx, log sy), (log sz, opacity).
fn splat_position(index: u32) -> vec3<f32> {
    let base = index * 8u;
    return vec3<f32>(
        bitcast<f32>(splats[base]), bitcast<f32>(splats[base + 1u]), bitcast<f32>(splats[base + 2u])
    );
}

fn optical(index: u32) -> vec3<f32> {
    let view = mat3x3<f32>(camera.rotation[0].xyz, camera.rotation[1].xyz, camera.rotation[2].xyz);
    return view * (splat_position(index) - camera.position.xyz);
}

struct Footprint {
    t: vec3<f32>,  // the centre in the optical frame
    r: mat3x3<f32>,  // the splat's axes in the optical frame, as columns
    scale: vec3<f32>,
    opacity: f32,
    center: vec2<f32>,  // pixels
    axis: vec2<f32>,  // of the projected ellipse's major axis
    major: f32,  // variances along it and across, pixels squared
    minor: f32,
    // Standard deviations its quad reaches: QUAD_SIGMAS, or less where a faint splat's alpha has
    // fallen under the 1/255 a fragment needs; 0 if it never reaches that.
    extent: f32,
};

fn footprint(index: u32) -> Footprint {
    var out: Footprint;
    let base = index * 8u;
    let wx = unpack2x16float(splats[base + 3u]);
    let yz = unpack2x16float(splats[base + 4u]);
    let sxy = unpack2x16float(splats[base + 5u]);
    let szo = unpack2x16float(splats[base + 6u]);
    let view = mat3x3<f32>(camera.rotation[0].xyz, camera.rotation[1].xyz, camera.rotation[2].xyz);
    let t = optical(index);

    let q = normalize(vec4<f32>(wx.x, wx.y, yz.x, yz.y));  // w x y z
    let w = q.x; let x = q.y; let y = q.z; let z = q.w;
    let r = view * mat3x3<f32>(
        vec3<f32>(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)),
        vec3<f32>(2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x)),
        vec3<f32>(2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)),
    );
    let f = camera.focal;
    // As 3DGUT's minimum projected ray radius: no splat is thinner than about half a pixel.
    let scale = max(exp(vec3<f32>(sxy, szo.x)), vec3<f32>(MIN_PIXEL_RADIUS * t.z / f.x));
    let m = r * mat3x3<f32>(
        vec3<f32>(scale.x, 0.0, 0.0), vec3<f32>(0.0, scale.y, 0.0), vec3<f32>(0.0, 0.0, scale.z)
    );
    // The quad bounds the splat's projected 2D covariance (EWA splatting).
    let j = mat3x2<f32>(  // the projection's Jacobian, columns
        vec2<f32>(f.x / t.z, 0.0), vec2<f32>(0.0, f.y / t.z),
        vec2<f32>(-f.x * t.x / (t.z * t.z), -f.y * t.y / (t.z * t.z)),
    );
    let cov = j * (m * transpose(m)) * transpose(j);
    let a = cov[0][0]; let b = cov[0][1]; let c = cov[1][1];
    let mid = 0.5 * (a + c);
    let radius = sqrt(max(0.25 * (a - c) * (a - c) + b * b, 0.0));
    out.major = mid + radius;
    out.minor = max(mid - radius, 1e-6);
    out.axis = vec2<f32>(1.0, 0.0);
    if abs(b) > 1e-12 { out.axis = normalize(vec2<f32>(b, out.major - a)); }
    else if c > a { out.axis = vec2<f32>(0.0, 1.0); }
    out.center = vec2<f32>(f.x * t.x / t.z, f.y * t.y / t.z) + 0.5 * camera.size;
    out.t = t;
    out.r = r;
    out.scale = scale;
    out.opacity = szo.y;
    out.extent = min(QUAD_SIGMAS, sqrt(2.0 * log(max(255.0 * min(szo.y, 0.99), 1.0))));
    return out;
}

fn on_screen(p: Footprint) -> bool {
    let reach = p.extent * sqrt(p.major);
    let inside = all(p.center >= vec2<f32>(-reach)) && all(p.center <= camera.size + reach);
    return p.extent > 0.0 && inside;
}
"""

SHADER = """
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> order: array<u32>;
@group(0) @binding(2) var<storage, read> splats: array<u32>;
// Per splat and colour channel, 16 f16 spherical-harmonic coefficients in 8 words.
@group(0) @binding(3) var<storage, read> sh_red: array<u32>;
@group(0) @binding(4) var<storage, read> sh_green: array<u32>;
@group(0) @binding(5) var<storage, read> sh_blue: array<u32>;

struct Varyings {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) origin: vec3<f32>,
    @location(1) ray: vec3<f32>,
    @location(2) @interpolate(flat) color: vec4<f32>,
};

fn sh_channel(buffer: u32, index: u32, basis: array<f32, 16>) -> f32 {
    var b = basis;  // a var, so it can be indexed at run time
    var total = 0.0;
    for (var word = 0u; word < 8u; word++) {
        var packed: u32;
        if buffer == 0u { packed = sh_red[index * 8u + word]; }
        else if buffer == 1u { packed = sh_green[index * 8u + word]; }
        else { packed = sh_blue[index * 8u + word]; }
        let pair = unpack2x16float(packed);
        total += pair.x * b[2u * word] + pair.y * b[2u * word + 1u];
    }
    return total;
}

fn sh_basis(d: vec3<f32>) -> array<f32, 16> {
    let x = d.x; let y = d.y; let z = d.z;
    let xx = x * x; let yy = y * y; let zz = z * z;
    return array<f32, 16>(
        0.28209479177387814,
        -0.4886025119029199 * y, 0.4886025119029199 * z, -0.4886025119029199 * x,
        1.0925484305920792 * x * y, -1.0925484305920792 * y * z,
        0.31539156525252005 * (2.0 * zz - xx - yy), -1.0925484305920792 * x * z,
        0.5462742152960396 * (xx - yy),
        -0.5900435899266435 * y * (3.0 * xx - yy), 2.890611442640554 * x * y * z,
        -0.4570457994644658 * y * (4.0 * zz - xx - yy),
        0.3731763325901154 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy),
        -0.4570457994644658 * x * (4.0 * zz - xx - yy),
        1.445305721320277 * z * (xx - yy), -0.5900435899266435 * x * (xx - 3.0 * yy),
    );
}

@vertex
fn vs_main(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> Varyings {
    var out: Varyings;
    out.position = vec4<f32>(0.0, 0.0, 2.0, 1.0);  // outside the clip volume: drawn nowhere
    let index = order[instance];
    let p = footprint(index);
    if !on_screen(p) { return out; }

    let corner = (vec2<f32>(f32(vertex & 1u), f32(vertex >> 1u)) * 2.0 - 1.0) * p.extent;
    let normal = vec2<f32>(-p.axis.y, p.axis.x);
    let pixel = p.center + corner.x * sqrt(p.major) * p.axis + corner.y * sqrt(p.minor) * normal;
    let ndc = pixel / camera.size * 2.0 - 1.0;
    out.position = vec4<f32>(ndc.x, -ndc.y, 0.5, 1.0);
    // Within the quad, each pixel's ray is evaluated against the splat in 3D, as 3DGUT renders
    // it: in the splat's unit frame, from the camera's origin along the pixel's ray.
    let unit = transpose(mat3x3<f32>(p.r[0] / p.scale.x, p.r[1] / p.scale.y, p.r[2] / p.scale.z));
    out.origin = -(unit * p.t);
    out.ray = unit * vec3<f32>((pixel - 0.5 * camera.size) / camera.focal, 1.0);  // linear
    let basis = sh_basis(normalize(splat_position(index) - camera.position.xyz));
    let rgb = vec3<f32>(
        sh_channel(0u, index, basis), sh_channel(1u, index, basis), sh_channel(2u, index, basis)
    );
    out.color = vec4<f32>(max(rgb + 0.5, vec3<f32>(0.0)), p.opacity);
    return out;
}

@fragment
fn fs_main(in: Varyings) -> @location(0) vec4<f32> {
    // The Gaussian's peak along the ray: its squared distance from the splat's centre.
    let across = cross(in.origin, in.ray);
    let power = dot(across, across) / dot(in.ray, in.ray);
    let alpha = min(0.99, in.color.a * exp(-0.5 * power));
    if alpha < 1.0 / 255.0 { discard; }
    return vec4<f32>(in.color.rgb * alpha, alpha);  // premultiplied, blended far to near
}
"""

# The splats in view sorted far to near on the GPU, by a 16-bit depth key across their depth range
# (a counting sort: cull, then a histogram of the keys, its prefix sum, and a scatter), so the CPU
# never touches a splat; the draw's instance count is written there too. Splats with one key (in a
# house, ~0.2 mm apart in depth) are drawn in no set order among themselves.
SORT_SHADER = """
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> splats: array<u32>;
@group(0) @binding(2) var<storage, read> reach: array<f32>;  // metres, of each splat's quad
@group(0) @binding(3) var<storage, read_write> keys: array<u32>;  // depth bits, then the key
// The depth range in view, as f32 bits (positive, so ordered as u32): the most of the bits
// inverted (the least depth), and the most of the bits.
@group(0) @binding(4) var<storage, read_write> range: array<atomic<u32>, 2>;
@group(0) @binding(5) var<storage, read_write> bins: array<atomic<u32>, KEYS>;
@group(0) @binding(6) var<storage, read_write> order: array<u32>;
@group(0) @binding(7) var<storage, read_write> draw: array<u32, 4>;  // vertices, instances, 0, 0

const CULLED = 0xffffffffu;
const KEYS = DEPTH_LEVELSu + 1u;
var<workgroup> least: atomic<u32>;
var<workgroup> most: atomic<u32>;
var<workgroup> sums: array<u32, 256>;

fn splat(id: vec3<u32>, groups: vec3<u32>) -> u32 {
    return id.y * groups.x * 256u + id.x;
}

// Those between the near and far planes whose quads reach the image.
@compute @workgroup_size(256)
fn cull(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_index) local: u32,
) {
    if local == 0u { atomicStore(&least, 0u); atomicStore(&most, 0u); }
    workgroupBarrier();
    let index = splat(id, groups);
    if index < camera.count {
        var key = CULLED;
        let depth = optical(index).z;
        if depth > camera.near && depth - reach[index] < camera.far && on_screen(footprint(index)) {
            key = bitcast<u32>(depth);
            atomicMax(&least, ~key);
            atomicMax(&most, key);
        }
        keys[index] = key;
    }
    workgroupBarrier();
    if local == 0u {
        atomicMax(&range[0], atomicLoad(&least));
        atomicMax(&range[1], atomicLoad(&most));
    }
}

@compute @workgroup_size(256)
fn histogram(
    @builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>
) {
    let index = splat(id, groups);
    if index >= camera.count || keys[index] == CULLED { return; }
    let low = bitcast<f32>(~atomicLoad(&range[0]));
    let high = bitcast<f32>(atomicLoad(&range[1]));
    let scaled = (high - bitcast<f32>(keys[index])) * (f32(DEPTH_LEVELSu) / max(high - low, 1e-6));
    let key = min(u32(scaled), DEPTH_LEVELSu);
    keys[index] = key;
    atomicAdd(&bins[key], 1u);
}

// One workgroup: each bin's count becomes where its run starts.
@compute @workgroup_size(256)
fn scan(@builtin(local_invocation_index) local: u32) {
    let per = KEYS / 256u;
    var total = 0u;
    for (var k = 0u; k < per; k++) { total += atomicLoad(&bins[local * per + k]); }
    sums[local] = total;
    workgroupBarrier();
    for (var step = 1u; step < 256u; step *= 2u) {
        var add = 0u;
        if local >= step { add = sums[local - step]; }
        workgroupBarrier();
        sums[local] += add;
        workgroupBarrier();
    }
    var start = sums[local] - total;
    for (var k = 0u; k < per; k++) {
        let count = atomicLoad(&bins[local * per + k]);
        atomicStore(&bins[local * per + k], start);
        start += count;
    }
    if local == 255u {
        draw[0] = 4u;
        draw[1] = sums[255];
        draw[2] = 0u;
        draw[3] = 0u;
    }
}

@compute @workgroup_size(256)
fn scatter(
    @builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>
) {
    let index = splat(id, groups);
    if index >= camera.count || keys[index] == CULLED { return; }
    order[atomicAdd(&bins[keys[index]], 1u)] = index;
}
"""

# The blended frame (rgba16float) to the published image's bytes, RGB8 row-major, on the GPU: each
# invocation packs 4 pixels (12 bytes) into 3 words, rounded as numpy's clip(255 c + 0.5).
PACK_SHADER = """
@group(0) @binding(0) var frame: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> rgb: array<u32>;

fn pixel(index: u32, size: vec2<u32>) -> vec4<u32> {
    if index >= size.x * size.y { return vec4<u32>(0u); }
    let value = textureLoad(frame, vec2<u32>(index % size.x, index / size.x), 0);
    return vec4<u32>(clamp(value * 255.0 + 0.5, vec4<f32>(0.0), vec4<f32>(255.0)));
}

@compute @workgroup_size(256)
fn pack(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let size = textureDimensions(frame);
    let quad = id.y * groups.x * 256u + id.x;
    if quad * 4u >= size.x * size.y { return; }
    let a = pixel(quad * 4u, size);
    let b = pixel(quad * 4u + 1u, size);
    let c = pixel(quad * 4u + 2u, size);
    let d = pixel(quad * 4u + 3u, size);
    rgb[quad * 3u] = a.r | (a.g << 8u) | (a.b << 16u) | (b.r << 24u);
    rgb[quad * 3u + 1u] = b.g | (b.b << 8u) | (c.r << 16u) | (c.g << 24u);
    rgb[quad * 3u + 2u] = c.b | (d.r << 8u) | (d.g << 16u) | (d.b << 24u);
}
"""


def _wgsl(code: str) -> str:
    return (
        (COMMON + code)
        .replace("QUAD_SIGMAS", f"{QUAD_SIGMAS:.1f}")
        .replace("MIN_PIXEL_RADIUS", f"{MIN_PIXEL_RADIUS}")
        .replace("DEPTH_LEVELS", str(DEPTH_LEVELS))
    )


def _workgroups(invocations: int) -> tuple[int, int]:
    """Workgroups of 256 for *invocations*, as x by y: no dimension may pass 65535."""
    groups = max(1, math.ceil(invocations / 256))
    x = min(groups, 65535)
    return x, math.ceil(groups / x)


@dataclass
class Splats:
    """Gaussians in the world frame: positions, w-x-y-z rotations, log scales, opacities in
    [0, 1], and (N, K, 3) spherical-harmonic colour coefficients, K = 1, 4, 9 or 16 (degree 0-3),
    as 3DGS's: colour = 0.5 + the SH sum."""

    positions: NDArray[np.float32]
    rotations: NDArray[np.float16]
    log_scales: NDArray[np.float16]
    opacities: NDArray[np.float16]
    sh: NDArray[np.float16]

    def __len__(self) -> int:
        return len(self.positions)

    @property
    def sh_degree(self) -> int:
        return math.isqrt(self.sh.shape[1]) - 1


def load_splats(path: Path | str) -> Splats:
    """Splats from a NuRec container (.nurec), a 3DGS PLY (.ply), or a SOG's meta.json."""
    path = Path(path)
    if path.suffix == ".nurec":
        return load_nurec(path)
    if path.suffix == ".ply":
        return load_ply(path)
    if path.name.endswith("meta.json"):
        return load_sog(path)
    raise ValueError(f"{path}: not splats (.nurec, .ply, or a SOG's meta.json)")


def is_splats(path: Path | str) -> bool:
    return Path(path).suffix in (".nurec", ".ply") or str(path).endswith("meta.json")


_PLY_TYPES = {
    "char": "i1",
    "uchar": "u1",
    "short": "<i2",
    "ushort": "<u2",
    "int": "<i4",
    "uint": "<u4",
    "float": "<f4",
    "double": "<f8",
    "int8": "i1",
    "uint8": "u1",
    "int16": "<i2",
    "uint16": "<u2",
    "int32": "<i4",
    "uint32": "<u4",
    "float32": "<f4",
    "float64": "<f8",
}


def load_ply(path: Path | str) -> Splats:
    """Splats from a 3DGS PLY (binary little-endian): x y z, f_dc_*, f_rest_* (channel-major),
    opacity (a logit), scale_* (logs), rot_* (w x y z)."""
    with open(path, "rb") as file:
        if file.readline().strip() != b"ply":
            raise ValueError(f"{path}: not a PLY")
        fields: list[tuple[str, str]] = []
        count, vertex = 0, False
        while (line := file.readline().decode("ascii").split()) != ["end_header"]:
            if line[:1] == ["format"] and line[1] != "binary_little_endian":
                raise ValueError(f"{path}: only binary little-endian PLYs are read")
            if line[:1] == ["element"]:
                vertex = line[1] == "vertex"
                count = int(line[2]) if vertex else count
            elif line[:1] == ["property"] and vertex:
                fields.append((line[2], _PLY_TYPES[line[1]]))
        data = np.fromfile(file, np.dtype(fields), count)

    def columns(*names: str) -> NDArray[np.float32]:
        return np.stack([data[n].astype(np.float32) for n in names], axis=1)

    names = set(data.dtype.names or ())
    rest = sum(1 for n in names if n.startswith("f_rest_")) // 3
    coefficients = min(rest + 1, SH_COEFFICIENTS)
    sh = np.zeros((count, coefficients, 3), np.float32)
    sh[:, 0] = columns("f_dc_0", "f_dc_1", "f_dc_2")
    for c in range(3):
        for k in range(1, coefficients):
            sh[:, k, c] = data[f"f_rest_{c * rest + k - 1}"]
    return Splats(
        positions=columns("x", "y", "z"),
        rotations=columns("rot_0", "rot_1", "rot_2", "rot_3").astype(np.float16),
        log_scales=columns("scale_0", "scale_1", "scale_2").astype(np.float16),
        opacities=(1 / (1 + np.exp(-data["opacity"].astype(np.float32)))).astype(np.float16),
        sh=sh.astype(np.float16),
    )


def save_ply(splats: Splats, path: Path | str) -> None:
    """*splats* as a 3DGS PLY, as load_ply reads it."""
    coefficients = splats.sh.shape[1]
    names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"]
    names += [f"f_rest_{i}" for i in range(3 * (coefficients - 1))]
    names += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    sh = splats.sh.astype(np.float32)
    opacity = np.clip(splats.opacities.astype(np.float32), 1e-6, 1 - 1e-6)
    table = np.concatenate(
        [
            splats.positions.astype(np.float32),
            sh[:, 0],
            sh[:, 1:].transpose(0, 2, 1).reshape(len(splats), -1),  # channel-major
            np.log(opacity / (1 - opacity))[:, None],
            splats.log_scales.astype(np.float32),
            splats.rotations.astype(np.float32),
        ],
        axis=1,
    ).astype("<f4")
    header = "ply\nformat binary_little_endian 1.0\nelement vertex {}\n{}end_header\n".format(
        len(splats), "".join(f"property float {n}\n" for n in names)
    )
    with open(path, "wb") as file:
        file.write(header.encode("ascii"))
        file.write(table.tobytes())


def load_sog(meta: Path | str) -> Splats:
    """Splats from PlayCanvas's SOG (version 2): a meta.json beside lossless WebP images."""
    import json

    from PIL import Image

    folder = Path(meta).parent
    info = json.loads(Path(meta).read_text())
    if info.get("version") != 2:
        raise ValueError(f"{meta}: only SOG version 2 is read")
    count = info["count"]

    def pixels(name: str) -> NDArray[np.uint8]:
        image = np.asarray(Image.open(folder / name).convert("RGBA"))
        return image.reshape(-1, 4)[:count]

    low, high = (pixels(f)[:, :3].astype(np.float64) for f in info["means"]["files"])
    mins, maxs = np.array(info["means"]["mins"]), np.array(info["means"]["maxs"])
    logged = mins + (maxs - mins) * (low + 256 * high) / 65535
    positions = np.sign(logged) * np.expm1(np.abs(logged))  # stored as sign(x) log(|x| + 1)
    # The three smaller quaternion components, then which (w x y z) is the largest, plus 252.
    quats = pixels(info["quats"]["files"][0])
    largest = np.clip(quats[:, 3].astype(np.int64) - 252, 0, 3)
    smaller = (quats[:, :3] / 255 * 2 - 1) / math.sqrt(2)
    rotations = np.zeros((count, 4))
    others = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])[largest]
    np.put_along_axis(rotations, others, smaller, axis=1)
    rest = np.sqrt(np.clip(1 - (smaller**2).sum(axis=1), 0, None))
    np.put_along_axis(rotations, largest[:, None], rest[:, None], axis=1)
    log_scales = np.asarray(info["scales"]["codebook"])[pixels(info["scales"]["files"][0])[:, :3]]
    colour = pixels(info["sh0"]["files"][0])
    dc = np.asarray(info["sh0"]["codebook"])[colour[:, :3]]
    sh = dc[:, None, :]
    if "shN" in info and info["shN"].get("bands", 0) > 0:
        coefficients = (info["shN"]["bands"] + 1) ** 2 - 1
        codebook = np.asarray(info["shN"]["codebook"])
        centroids = np.asarray(Image.open(folder / info["shN"]["files"][0]).convert("RGBA"))
        labels = pixels(info["shN"]["files"][1]).astype(np.int64)
        label = np.minimum(labels[:, 0] | (labels[:, 1] << 8), info["shN"]["count"] - 1)
        # Centroid c's coefficient k is pixel (c % 64 * coefficients + k, c // 64).
        k = np.arange(coefficients)
        palette = centroids[label[:, None] // 64, (label[:, None] % 64) * coefficients + k, :3]
        sh = np.concatenate([sh, codebook[palette]], axis=1)
    return Splats(
        positions=positions.astype(np.float32),
        rotations=rotations.astype(np.float16),
        log_scales=log_scales.astype(np.float16),
        opacities=(colour[:, 3] / 255).astype(np.float16),  # stored as the opacity itself
        sh=sh[:, :SH_COEFFICIENTS].astype(np.float16),
    )


def load_nurec(path: Path | str) -> Splats:
    """Splats from an NVIDIA NuRec container (3DGRUT's USDZ export): gzipped msgpack of arrays."""
    import msgpack

    with gzip.open(path) as file:
        state = msgpack.unpackb(file.read())["nre_data"]["state_dict"]
    prefix = ".gaussians_nodes.gaussians."

    def array(name: str) -> NDArray[np.float32]:
        shape, raw = state[f"{prefix}{name}.shape"], state[prefix + name]
        kind = np.float16 if len(raw) == 2 * math.prod(shape) else np.float32  # precision 16 or 32
        return np.frombuffer(raw, kind).reshape(shape).astype(np.float32)

    count = len(array("densities"))
    # The albedo is the degree-0 coefficient; the rest are coefficient-major, RGB innermost.
    albedo = array("features_albedo").reshape(count, 1, 3)
    sh = np.concatenate([albedo, array("features_specular").reshape(count, -1, 3)], axis=1)
    density = array("densities")[:, 0]
    return Splats(
        positions=array("positions"),
        rotations=array("rotations").astype(np.float16),
        log_scales=array("scales").astype(np.float16),
        opacities=(1 / (1 + np.exp(-density))).astype(np.float16),  # densities are logits
        sh=sh[:, :SH_COEFFICIENTS].astype(np.float16),
    )


def spz(splats: Splats, sh_degree: int = 3) -> bytes:
    """Niantic's SPZ (version 2, gzipped), as Spark loads it: positions in 12-bit fixed point,
    the rest quantised to bytes; in the world frame (Z up, metres)."""
    count = len(splats)
    sh_degree = min(sh_degree, splats.sh_degree)

    def byte(values: NDArray[np.floating]) -> NDArray[np.uint8]:
        return np.clip(np.rint(values), 0, 255).astype(np.uint8)

    fixed = np.rint(splats.positions.astype(np.float64) * (1 << 12)).astype("<i4")
    rotations = splats.rotations.astype(np.float32)
    xyzw = np.roll(rotations / np.linalg.norm(rotations, axis=1, keepdims=True), -1, axis=1)
    xyzw *= np.where(xyzw[:, 3:] < 0, -1.0, 1.0)  # w >= 0, so only x, y, z are stored
    parts = [
        struct.pack("<IIIBBBB", 0x5053474E, 2, count, sh_degree, 12, 0, 0),
        fixed.view(np.uint8).reshape(count, 3, 4)[:, :, :3].tobytes(),
        byte(splats.opacities.astype(np.float32) * 255).tobytes(),
        byte(splats.sh[:, 0].astype(np.float32) * (0.15 * 255) + 127.5).tobytes(),
        byte((splats.log_scales.astype(np.float32) + 10) * 16).tobytes(),
        byte(xyzw[:, :3] * 127.5 + 127.5).tobytes(),
    ]
    rest = (sh_degree + 1) ** 2 - 1
    if rest:
        # Degree 1 keeps 5 bits a value, the higher degrees 4, as SPZ's own encoder.
        levels = np.rint(splats.sh[:, 1 : 1 + rest].astype(np.float32) * 128) + 128
        bucket = np.where(np.arange(rest) < 3, 8, 16)[None, :, None]
        parts.append(byte((levels + bucket // 2) // bucket * bucket).tobytes())
    return gzip.compress(b"".join(parts), 6)


def splats_of(positions: NDArray[np.float32], rgb: NDArray[np.float32]) -> Splats:
    """Opaque round splats (PERSON_SIGMA) at *positions*, coloured *rgb*."""
    count = len(positions)
    return Splats(
        positions=positions.astype(np.float32),
        rotations=np.tile(np.array([1, 0, 0, 0], np.float16), (count, 1)),
        log_scales=np.full((count, 3), math.log(PERSON_SIGMA), np.float16),
        opacities=np.full(count, 0.99, np.float16),
        sh=((rgb - 0.5) / SH_C0).astype(np.float16)[:, None, :],
    )


def _concatenated(parts: list[Splats]) -> Splats:
    """*parts* as one, at the highest spherical-harmonic degree among them (the rest zero)."""
    k = max(p.sh.shape[1] for p in parts)
    sh = [np.pad(p.sh, ((0, 0), (0, k - p.sh.shape[1]), (0, 0))) for p in parts]
    fields = ("positions", "rotations", "log_scales", "opacities")
    return Splats(
        *(np.concatenate([getattr(p, f) for p in parts]) for f in fields), np.concatenate(sh)
    )


def _packed_splats(splats: Splats) -> NDArray[np.uint32]:
    """8 words per splat, laid out as the shader reads them."""
    words = np.zeros((len(splats), 8), np.uint32)
    words[:, :3] = splats.positions.view(np.uint32)
    halves = np.concatenate(
        [splats.rotations, splats.log_scales, splats.opacities[:, None]], axis=1
    ).astype(np.float16)
    words[:, 3:7] = halves.view(np.uint32)
    return words


class SplatRenderer:
    """Draws *splats*, and *people* people among them, from a camera pose: position, forward and up
    in the world frame. Each person is opaque round splats (people.splats), sorted with the
    scene's, so a wall hides whoever is behind it. All of a frame's work is on the GPU (sorting
    the splats, drawing them, and packing the image's bytes); the CPU only reads the bytes back."""

    def __init__(self, splats: Splats, config: CameraConfig, people: int = 0) -> None:
        import wgpu  # type: ignore[import-untyped]

        from dim_go2_sim.camera import focal_length

        self._scene_count = len(splats)
        self._people = [person_splats(i) for i in range(people)]
        if people:
            hidden = np.float32([0.0, 0.0, -100.0])  # out of sight until placed
            splats = _concatenated([splats, *(splats_of(p + hidden, c) for p, c in self._people)])
        self._config = config
        self._count = len(splats)
        # How far each splat's quad can reach, metres, for culling it against the far plane.
        reach = QUAD_SIGMAS * np.exp(splats.log_scales.astype(np.float32).max(axis=1))
        self._focal = focal_length(config)
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        packed = _packed_splats(splats)
        self._people_packed = packed[self._scene_count :].copy()
        sh = splats.sh
        if sh.shape[1] < SH_COEFFICIENTS:  # the shader reads degree 3; the rest are zero
            sh = np.concatenate(
                [sh, np.zeros((len(sh), SH_COEFFICIENTS - sh.shape[1], 3), np.float16)], axis=1
            )
        channels = [np.ascontiguousarray(sh[:, :, c]).view(np.uint32) for c in range(3)]
        largest = max(packed.nbytes, 4 * len(splats), *(c.nbytes for c in channels))
        limits = adapter.limits
        self._device = device = adapter.request_device_sync(
            required_limits={
                "max-storage-buffer-binding-size": min(
                    limits["max-storage-buffer-binding-size"], max(largest, 128 << 20)
                ),
                "max-buffer-size": min(limits["max-buffer-size"], max(largest, 256 << 20)),
            }
        )
        storage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST

        def buffer(data: NDArray[np.generic]) -> object:
            gpu = device.create_buffer(size=max(data.nbytes, 16), usage=storage)
            device.queue.write_buffer(gpu, 0, data)
            return gpu

        def scratch(size: int, usage: int = 0) -> object:
            return device.create_buffer(size=max(size, 16), usage=storage | usage)

        self._camera = device.create_buffer(
            size=112, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )
        self._order = scratch(4 * len(splats))
        self._range = scratch(8)
        self._bins = scratch(4 * (DEPTH_LEVELS + 1))
        self._draw = scratch(16, wgpu.BufferUsage.INDIRECT)
        self._splats = buffer(packed)
        module = device.create_shader_module(code=_wgsl(SHADER))
        self._pipeline = device.create_render_pipeline(
            layout="auto",
            vertex={"module": module, "entry_point": "vs_main"},
            primitive={"topology": wgpu.PrimitiveTopology.triangle_strip},
            fragment={
                "module": module,
                "entry_point": "fs_main",
                "targets": [
                    {
                        "format": wgpu.TextureFormat.rgba16float,
                        "blend": {
                            "color": {
                                "src_factor": wgpu.BlendFactor.one,
                                "dst_factor": wgpu.BlendFactor.one_minus_src_alpha,
                            },
                            "alpha": {
                                "src_factor": wgpu.BlendFactor.one,
                                "dst_factor": wgpu.BlendFactor.one_minus_src_alpha,
                            },
                        },
                    }
                ],
            },
        )
        resources = [self._camera, self._order, self._splats, *(buffer(c) for c in channels)]
        self._bindings = device.create_bind_group(
            layout=self._pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": i, "resource": {"buffer": b, "offset": 0, "size": b.size}}
                for i, b in enumerate(resources)
            ],
        )

        # The sort's four passes share one layout.
        sorting = [
            self._camera,
            self._splats,
            buffer(reach),
            scratch(4 * len(splats)),  # keys
            self._range,
            self._bins,
            self._order,
            self._draw,
        ]
        kinds = ["uniform", "read-only-storage", "read-only-storage"] + ["storage"] * 5
        layout = device.create_bind_group_layout(
            entries=[
                {"binding": i, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": kind}}
                for i, kind in enumerate(kinds)
            ]
        )
        sort_module = device.create_shader_module(code=_wgsl(SORT_SHADER))
        pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[layout])
        self._sort = [
            device.create_compute_pipeline(
                layout=pipeline_layout, compute={"module": sort_module, "entry_point": name}
            )
            for name in ("cull", "histogram", "scan", "scatter")
        ]
        self._sort_bindings = device.create_bind_group(
            layout=layout,
            entries=[
                {"binding": i, "resource": {"buffer": b, "offset": 0, "size": b.size}}
                for i, b in enumerate(sorting)
            ],
        )
        self._splat_workgroups = _workgroups(len(splats))

        self._target = device.create_texture(
            size=(config.width, config.height, 1),
            format=wgpu.TextureFormat.rgba16float,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING,
        )
        self._view = self._target.create_view()
        background = config.background.lstrip("#")
        self._background = [int(background[i : i + 2], 16) / 255 for i in (0, 2, 4)]

        quads = math.ceil(config.width * config.height / 4)  # 4 pixels, 3 words of RGB each
        self._pack = device.create_compute_pipeline(
            layout="auto",
            compute={
                "module": device.create_shader_module(code=PACK_SHADER),
                "entry_point": "pack",
            },
        )
        rgb = scratch(12 * quads, wgpu.BufferUsage.COPY_SRC)
        # One frame can be read back while the next is drawn (submit, then collect).
        self._readbacks = [
            device.create_buffer(
                size=rgb.size, usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST
            )
            for _ in range(FRAMES_IN_FLIGHT)
        ]
        self._in_flight: deque[tuple[Any, Any]] = deque()  # readback buffers, being mapped
        self._pack_bindings = device.create_bind_group(
            layout=self._pack.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": self._view},
                {"binding": 1, "resource": {"buffer": rgb, "offset": 0, "size": rgb.size}},
            ],
        )
        self._rgb = rgb
        self._pack_workgroups = _workgroups(quads)

    def place_people(self, poses: NDArray[np.float64]) -> None:
        """Moves each person's splats to their pose: x, y, yaw."""
        if not self._people:
            return
        start = 0
        for (local, _), (x, y, yaw) in zip(self._people, poses, strict=True):
            world = local @ yaw_rotation(yaw).T.astype(np.float32) + np.float32([x, y, 0.0])
            self._people_packed[start : start + len(local), :3] = world.view(np.uint32)
            start += len(local)
        self._device.queue.write_buffer(self._splats, self._scene_count * 32, self._people_packed)

    def capture(
        self,
        position: NDArray[np.float64],
        forward: NDArray[np.float64],
        up: NDArray[np.float64],
        people: NDArray[np.float64] | None = None,
    ) -> NDArray[np.uint8]:
        self.submit(position, forward, up, people)
        return self.collect()

    def submit(
        self,
        position: NDArray[np.float64],
        forward: NDArray[np.float64],
        up: NDArray[np.float64],
        people: NDArray[np.float64] | None = None,
    ) -> None:
        """Starts drawing the view from *position*; collect() returns it. Up to FRAMES_IN_FLIGHT
        can be drawing at once: the GPU runs them in order."""
        if len(self._in_flight) == len(self._readbacks):
            raise RuntimeError("collect a frame before submitting another")
        import wgpu

        if people is not None:
            self.place_people(people)
        cfg = self._config
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        rotation = np.stack([right, down, forward])  # world to optical
        uniform = np.zeros(28, np.float32)
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = rotation
        uniform[:16] = matrix.T.ravel()  # WGSL matrices are column-major
        uniform[16:19] = position
        uniform[20:26] = (self._focal, self._focal, cfg.width, cfg.height, cfg.near, cfg.far)
        uniform[26:27] = np.uint32(self._count).view(np.float32)
        queue = self._device.queue
        queue.write_buffer(self._camera, 0, uniform)
        encoder = self._device.create_command_encoder()
        encoder.clear_buffer(self._range)
        encoder.clear_buffer(self._bins)
        sorting = encoder.begin_compute_pass()
        sorting.set_bind_group(0, self._sort_bindings)
        cull, histogram, scan, scatter = self._sort
        for pipeline, workgroups in [
            (cull, self._splat_workgroups),
            (histogram, self._splat_workgroups),
            (scan, (1, 1)),
            (scatter, self._splat_workgroups),
        ]:
            sorting.set_pipeline(pipeline)
            sorting.dispatch_workgroups(*workgroups)
        sorting.end()
        rendering = encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": self._view,
                    "clear_value": (*self._background, 1.0),
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                }
            ]
        )
        rendering.set_pipeline(self._pipeline)
        rendering.set_bind_group(0, self._bindings)
        rendering.draw_indirect(self._draw, 0)
        rendering.end()
        packing = encoder.begin_compute_pass()
        packing.set_pipeline(self._pack)
        packing.set_bind_group(0, self._pack_bindings)
        packing.dispatch_workgroups(*self._pack_workgroups)
        packing.end()
        readback = next(b for b in self._readbacks if all(b is not f for f, _ in self._in_flight))
        encoder.copy_buffer_to_buffer(self._rgb, 0, readback, 0, self._rgb.size)
        queue.submit([encoder.finish()])
        # Mapped now, not on collect: wgpu's poll waits on everything submitted when it starts,
        # so a map asked for after the next frame's submit would wait for that frame too.
        self._in_flight.append((readback, readback.map_async(wgpu.MapMode.READ)))

    def collect(self) -> NDArray[np.uint8]:
        """The earliest submitted frame not yet collected, RGB, once the GPU has drawn it."""
        cfg = self._config
        readback, mapped = self._in_flight.popleft()
        mapped.sync_wait()
        try:
            raw = readback.read_mapped(copy=False)
            size = cfg.width * cfg.height * 3
            return np.frombuffer(raw, np.uint8, size).reshape(cfg.height, cfg.width, 3).copy()
        finally:
            readback.unmap()
