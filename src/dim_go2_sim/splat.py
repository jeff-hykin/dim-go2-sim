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

Each frame the splats in view are sorted far to near on the CPU, then drawn as instanced quads
over their projected ellipses (EWA, as 3DGS) and blended over the ones behind. Within a quad,
each pixel's ray is evaluated against the splat in 3D, as 3DGUT (which trained the InteriorGS
splats) does: a linearised projection smears splats near the camera across the image. Colour is
degree-3 spherical harmonics, evaluated per view on the GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import math
from pathlib import Path
import struct

import numpy as np
from numpy.typing import NDArray

from dim_go2_sim.config import CameraConfig

# A splat's quad reaches this many standard deviations, where 3DGS stops drawing it.
QUAD_SIGMAS = 3.0
# 3DGUT, which trained the InteriorGS splats, draws none nearer than this, metres, and none
# thinner than this many pixels.
NEAR_CLIP = 0.2
MIN_PIXEL_RADIUS = 0.5477
SH_COEFFICIENTS = 16  # degree 3
# Depth keys are 16-bit, so numpy sorts them by radix: ~0.2 mm steps across a house.
DEPTH_LEVELS = 65535

SHADER = """
struct Camera {
    rotation: mat4x4<f32>,  // world to optical (x right, y down, z forward), as columns
    position: vec4<f32>,
    focal: vec2<f32>,
    size: vec2<f32>,
};
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> order: array<u32>;
// Per splat, 8 words: position (3 x f32), then f16 pairs: (w, x) (y, z) of the rotation,
// (log sx, log sy), (log sz, opacity).
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
    let base = index * 8u;
    let world = vec3<f32>(
        bitcast<f32>(splats[base]), bitcast<f32>(splats[base + 1u]), bitcast<f32>(splats[base + 2u])
    );
    let wx = unpack2x16float(splats[base + 3u]);
    let yz = unpack2x16float(splats[base + 4u]);
    let sxy = unpack2x16float(splats[base + 5u]);
    let szo = unpack2x16float(splats[base + 6u]);
    let view = mat3x3<f32>(camera.rotation[0].xyz, camera.rotation[1].xyz, camera.rotation[2].xyz);
    let t = view * (world - camera.position.xyz);

    let q = normalize(vec4<f32>(wx.x, wx.y, yz.x, yz.y));  // w x y z
    let w = q.x; let x = q.y; let y = q.z; let z = q.w;
    let r = view * mat3x3<f32>(  // the splat's axes in the optical frame, as columns
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
    let major = mid + radius;
    let minor = max(mid - radius, 1e-6);
    var axis = vec2<f32>(1.0, 0.0);
    if abs(b) > 1e-12 { axis = normalize(vec2<f32>(b, major - a)); }
    else if c > a { axis = vec2<f32>(0.0, 1.0); }
    let center = vec2<f32>(f.x * t.x / t.z, f.y * t.y / t.z) + 0.5 * camera.size;
    let reach = QUAD_SIGMAS * sqrt(major);
    if any(center < vec2<f32>(-reach)) || any(center > camera.size + reach) { return out; }

    let corner = (vec2<f32>(f32(vertex & 1u), f32(vertex >> 1u)) * 2.0 - 1.0) * QUAD_SIGMAS;
    let normal = vec2<f32>(-axis.y, axis.x);
    let pixel = center + corner.x * sqrt(major) * axis + corner.y * sqrt(minor) * normal;
    let ndc = pixel / camera.size * 2.0 - 1.0;
    out.position = vec4<f32>(ndc.x, -ndc.y, 0.5, 1.0);
    // Within the quad, each pixel's ray is evaluated against the splat in 3D, as 3DGUT renders
    // it: in the splat's unit frame, from the camera's origin along the pixel's ray.
    let unit = transpose(mat3x3<f32>(r[0] / scale.x, r[1] / scale.y, r[2] / scale.z));
    out.origin = -(unit * t);
    out.ray = unit * vec3<f32>((pixel - 0.5 * camera.size) / f, 1.0);  // linear across the quad
    let basis = sh_basis(normalize(world - camera.position.xyz));
    let rgb = vec3<f32>(
        sh_channel(0u, index, basis), sh_channel(1u, index, basis), sh_channel(2u, index, basis)
    );
    out.color = vec4<f32>(max(rgb + 0.5, vec3<f32>(0.0)), szo.y);
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
""".replace("QUAD_SIGMAS", f"{QUAD_SIGMAS:.1f}").replace("MIN_PIXEL_RADIUS", f"{MIN_PIXEL_RADIUS}")


@dataclass
class Splats:
    """Gaussians in the world frame: positions, w-x-y-z rotations, log scales, opacities in
    [0, 1], and (N, 16, 3) spherical-harmonic colour coefficients."""

    positions: NDArray[np.float32]
    rotations: NDArray[np.float16]
    log_scales: NDArray[np.float16]
    opacities: NDArray[np.float16]
    sh: NDArray[np.float16]

    def __len__(self) -> int:
        return len(self.positions)


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
    """Draws *splats* from a camera pose: position, forward and up in the world frame."""

    def __init__(self, splats: Splats, config: CameraConfig) -> None:
        import wgpu  # type: ignore[import-untyped]

        from dim_go2_sim.camera import focal_length

        self._config = config
        self._positions = splats.positions
        # How far each splat's quad can reach, metres, for culling it against the view.
        self._reach = QUAD_SIGMAS * np.exp(splats.log_scales.astype(np.float32).max(axis=1))
        self._focal = focal_length(config)
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        packed = _packed_splats(splats)
        channels = [np.ascontiguousarray(splats.sh[:, :, c]).view(np.uint32) for c in range(3)]
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

        def buffer(data: NDArray[np.uint32]) -> object:
            gpu = device.create_buffer(size=max(data.nbytes, 16), usage=storage)
            device.queue.write_buffer(gpu, 0, data)
            return gpu

        self._camera = device.create_buffer(
            size=96, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )
        self._order = device.create_buffer(size=max(4 * len(splats), 16), usage=storage)
        module = device.create_shader_module(code=SHADER)
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
        resources = [self._camera, self._order, buffer(packed), *(buffer(c) for c in channels)]
        self._bindings = device.create_bind_group(
            layout=self._pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": i, "resource": {"buffer": b, "offset": 0, "size": b.size}}
                for i, b in enumerate(resources)
            ],
        )
        self._target = device.create_texture(
            size=(config.width, config.height, 1),
            format=wgpu.TextureFormat.rgba16float,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
        )
        self._view = self._target.create_view()
        background = config.background.lstrip("#")
        self._background = [int(background[i : i + 2], 16) / 255 for i in (0, 2, 4)]

    def visible_order(
        self, position: NDArray[np.float64], forward: NDArray[np.float64]
    ) -> NDArray[np.uint32]:
        """The splats between the near and far planes, far to near. Those off to the side are
        culled on the GPU: the depth is all the CPU can afford per splat."""
        cfg = self._config
        depth = self._positions @ forward.astype(np.float32) - np.float32(position @ forward)
        visible = (depth > max(cfg.near, NEAR_CLIP)) & (depth - self._reach < cfg.far)
        index = np.flatnonzero(visible)
        if not len(index):
            return index.astype(np.uint32)
        depth = depth[index]
        low, high = float(depth.min()), float(depth.max())
        key = ((high - depth) * (DEPTH_LEVELS / max(high - low, 1e-6))).astype(np.uint16)
        return index[np.argsort(key, kind="stable")].astype(np.uint32)

    def capture(
        self, position: NDArray[np.float64], forward: NDArray[np.float64], up: NDArray[np.float64]
    ) -> NDArray[np.uint8]:
        import wgpu

        cfg = self._config
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        rotation = np.stack([right, down, forward])  # world to optical
        order = self.visible_order(position, forward)
        uniform = np.zeros(24, np.float32)
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = rotation
        uniform[:16] = matrix.T.ravel()  # WGSL matrices are column-major
        uniform[16:19] = position
        uniform[20:24] = (self._focal, self._focal, cfg.width, cfg.height)
        queue = self._device.queue
        queue.write_buffer(self._camera, 0, uniform)
        if len(order):
            queue.write_buffer(self._order, 0, order)
        encoder = self._device.create_command_encoder()
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
        if len(order):
            rendering.draw(4, len(order))
        rendering.end()
        queue.submit([encoder.finish()])
        raw = queue.read_texture(
            {"texture": self._target, "origin": (0, 0, 0), "mip_level": 0},
            {"offset": 0, "bytes_per_row": cfg.width * 8, "rows_per_image": cfg.height},
            (cfg.width, cfg.height, 1),
        )
        rgba = np.frombuffer(raw, np.float16).reshape(cfg.height, cfg.width, 4)
        rgb = np.clip(rgba[:, :, :3].astype(np.float32) * 255 + 0.5, 0, 255)
        return rgb.astype(np.uint8)
