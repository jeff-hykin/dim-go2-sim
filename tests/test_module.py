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

"""Go2Sim through dimOS: a blueprint with a probe that drives it and counts what comes out."""

from collections import Counter
import json
from pathlib import Path
import struct
import threading
import time
from typing import Any

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.coordination.module_coordinator import ModuleCoordinator
from dimos.core.core import rpc
from dimos.core.module import Module
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.Imu import Imu
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.msgs.tf2_msgs.TFMessage import TFMessage
import numpy as np
import pytest

from dim_go2_sim import Contacts, Go2Sim


class Probe(Module):
    cmd_vel: Out[Twist]
    lidar: In[PointCloud2]
    color_image: In[Image]
    camera_info: In[CameraInfo]
    imu: In[Imu]
    odom: In[PoseStamped]
    tf: In[TFMessage]
    contacts: In[Contacts]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._counts: Counter[str] = Counter()
        self._last: dict[str, Any] = {}
        self._lock = threading.Lock()

    @rpc
    def start(self) -> None:
        super().start()
        for name in ("lidar", "color_image", "camera_info", "imu", "odom", "tf", "contacts"):
            getattr(self, name).subscribe(lambda msg, name=name: self._on(name, msg))

    def _on(self, name: str, msg: Any) -> None:
        with self._lock:
            self._counts[name] += 1
            self._last[name] = msg

    @rpc
    def counts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)

    @rpc
    def position(self) -> list[float]:
        with self._lock:
            pose = self._last["odom"]
        return [pose.position.x, pose.position.y]

    @rpc
    def image_spread(self) -> float:
        """How much the last image varies: a blank background is zero."""
        with self._lock:
            image = self._last["color_image"]
        return float(np.asarray(image.data, dtype=np.float64).std())

    @rpc
    def drive(self, vx: float, seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.cmd_vel.publish(Twist(linear=Vector3(vx, 0.0, 0.0), angular=Vector3()))
            time.sleep(0.1)


def write_box_glb(path: Path) -> None:
    """A 1 m red cube at (3, 0, 0.5) in a Y-up glTF: something for the camera to see."""
    corners = np.array(
        [[x, y, z] for x in (-0.5, 0.5) for y in (0.0, 1.0) for z in (-0.5, 0.5)], np.float32
    ) + np.array([3.0, 0.0, 0.0], np.float32)  # Y-up: y is height, -z is the world's y
    faces = np.array(
        [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
         [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]],
        np.uint16,
    )  # fmt: skip
    blob = corners.tobytes() + faces.tobytes()
    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "materials": [{"pbrMetallicRoughness": {"baseColorFactor": [1, 0, 0, 1]}}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": 0}]}],
        "buffers": [{"byteLength": len(blob)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": corners.nbytes},
            {"buffer": 0, "byteOffset": corners.nbytes, "byteLength": faces.nbytes},
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": 8,
                "type": "VEC3",
                "min": corners.min(0).tolist(),
                "max": corners.max(0).tolist(),
            },
            {"bufferView": 1, "componentType": 5123, "count": faces.size, "type": "SCALAR"},
        ],
    }
    text = json.dumps(gltf).encode()
    text += b" " * (-len(text) % 4)
    blob += b"\0" * (-len(blob) % 4)
    chunks = struct.pack("<I4s", len(text), b"JSON") + text
    chunks += struct.pack("<I4s", len(blob), b"BIN\0") + blob
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, 12 + len(chunks)) + chunks)


@pytest.fixture
def coordinator(tmp_path: Path):
    visual = tmp_path / "box.glb"
    write_box_glb(visual)
    blueprint = autoconnect(
        Go2Sim.blueprint(
            scene={"visual": str(visual)},
            physics={"cache_dir": str(tmp_path / "cache")},
            camera={"width": 160, "height": 120},
        ),
        Probe.blueprint(),
    )
    coordinator = ModuleCoordinator.build(blueprint)
    try:
        yield coordinator
    finally:
        coordinator.stop()


def test_drives_and_publishes_every_stream(coordinator: ModuleCoordinator) -> None:
    probe = coordinator.get_instance(Probe)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline and len(probe.counts()) < 7:
        time.sleep(0.5)
    assert set(probe.counts()) == {
        "lidar", "color_image", "camera_info", "imu", "odom", "tf", "contacts"
    }  # fmt: skip
    before = probe.counts()
    start = probe.position()
    probe.drive(0.4, 3.0)
    after = probe.counts()
    assert after["lidar"] - before["lidar"] >= 20  # 10 Hz
    assert after["odom"] - before["odom"] >= 100  # 50 Hz
    assert after["color_image"] - before["color_image"] >= 10
    assert probe.position()[0] - start[0] > 0.6
    assert probe.image_spread() > 5  # the red cube, not just background
