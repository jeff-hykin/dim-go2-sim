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

"""Which scene geometry the robot touches, by robot part and kind of geometry. Ground truth."""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Literal

from dimos.types.timestamped import Timestamped
from dimos_lcm.std_msgs import String as LCMString
import mujoco
import numpy as np

Part = Literal["trunk", "leg", "foot"]
Kind = Literal["floor", "clutter"]


@dataclass(frozen=True, order=True)
class Contact:
    part: Part
    kind: Kind


class Contacts(Timestamped):
    """The set of contacts at one instant, published whenever it changes."""

    msg_name = "dim_go2_sim.Contacts"

    def __init__(self, contacts: list[Contact] | None = None, ts: float | None = None) -> None:
        self.contacts = contacts or []
        self.ts = time.time() if ts is None else ts

    def lcm_encode(self) -> bytes:
        payload = {"ts": self.ts, "contacts": [[c.part, c.kind] for c in self.contacts]}
        return LCMString(data=json.dumps(payload)).lcm_encode()  # type: ignore[no-any-return]

    @classmethod
    def lcm_decode(cls, data: bytes) -> Contacts:
        payload = json.loads(LCMString.lcm_decode(data).data)
        return cls([Contact(part, kind) for part, kind in payload["contacts"]], payload["ts"])


class ContactSensor:
    """Classifies each static scene geom once as floor (its top below floor_top_z) or clutter."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        robot_bodies: set[int],
        trunk: int,
        feet: tuple[int, ...],
        floor_top_z: float,
    ) -> None:
        self._data = data
        self._trunk = trunk
        self._feet = set(feet)
        self._robot = {g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in robot_bodies}
        self._body = model.geom_bodyid
        self._kind: dict[int, Kind] = {}
        for g in range(model.ngeom):
            if g in self._robot or model.body_weldid[model.geom_bodyid[g]] != 0:
                continue
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
                self._kind[g] = "floor"
                continue
            up = data.geom_xmat[g].reshape(3, 3)[2]
            center, half = model.geom_aabb[g, :3], model.geom_aabb[g, 3:]
            top = data.geom_xpos[g, 2] + up @ center + np.abs(up) @ half
            self._kind[g] = "floor" if top < floor_top_z else "clutter"

    def contacts(self) -> list[Contact]:
        """Every (robot part, scene kind) pair in contact right now, sorted."""
        found: set[Contact] = set()
        for i in range(self._data.ncon):
            c = self._data.contact[i]
            for scene, robot in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                kind = self._kind.get(int(scene))
                if kind is not None and int(robot) in self._robot:
                    found.add(Contact(self._part(int(robot)), kind))
        return sorted(found)

    def _part(self, geom: int) -> Part:
        if geom in self._feet:
            return "foot"
        if self._body[geom] == self._trunk:
            return "trunk"
        return "leg"
