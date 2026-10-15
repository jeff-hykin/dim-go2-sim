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

"""The open rl_sar robot_lab Go2 walking policy, and the loop that makes it track a command."""

from __future__ import annotations

from importlib.resources import as_file, files
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import onnxruntime as ort

LEGS = ("FR", "FL", "RR", "RL")
JOINT_NAMES = tuple(f"{leg}_{part}_joint" for leg in LEGS for part in ("hip", "thigh", "calf"))
DEFAULT_POSE = np.array([0.0, 0.8, -1.5] * 4)
KP, KD = 20.0, 0.5
CONTROL_DT = 0.02
# Most the command may change per tick, as the Go2's own driver limits it.
COMMAND_SLEW = np.array([0.05, 0.04, 0.10])


class OnnxPolicy:
    """A 45-observation MLP with the legged-gym layout: joint targets for a (vx, vy, wz)."""

    action_scale = np.array([0.125, 0.25, 0.25] * 4)
    angular_velocity_scale = 0.25
    joint_velocity_scale = 0.05

    def __init__(self, path: Path | None = None) -> None:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        with as_file(files("dim_go2_sim") / "data/policy/policy.onnx") as bundled:
            self._session = ort.InferenceSession(
                str(path or bundled), options, providers=["CPUExecutionProvider"]
            )
        self._input = self._session.get_inputs()[0].name
        self.reset()

    def reset(self) -> None:
        self._last_action = np.zeros(len(JOINT_NAMES))

    def act(
        self,
        angular_velocity: NDArray[np.float64],
        gravity: NDArray[np.float64],
        joint_position: NDArray[np.float64],
        joint_velocity: NDArray[np.float64],
        command: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        obs = np.concatenate(
            [
                angular_velocity * self.angular_velocity_scale,
                gravity,
                command,
                joint_position - DEFAULT_POSE,
                joint_velocity * self.joint_velocity_scale,
                self._last_action,
            ]
        )
        x = np.clip(obs, -100, 100).astype(np.float32)[None, :]
        (out,) = self._session.run(None, {self._input: x})
        self._last_action = np.clip(out[0].astype(np.float64), -100, 100)
        targets: NDArray[np.float64] = DEFAULT_POSE + self._last_action * self.action_scale
        return targets


class VelocityTracker:
    """Integral trim on the low-passed body velocity (vx, vy in the heading frame, and wz)."""

    tau = 0.3
    ki = np.array([1.5, 1.5, 2.0])
    limit = np.array([0.4, 0.3, 0.6])

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._measured = np.zeros(3)
        self._trim = np.zeros(3)

    def update(
        self, command: NDArray[np.float64], measured: NDArray[np.float64], dt: float
    ) -> NDArray[np.float64]:
        self._measured += dt / (self.tau + dt) * (measured - self._measured)
        error = command - self._measured
        self._trim = np.clip(self._trim + self.ki * error * dt, -self.limit, self.limit)
        trimmed: NDArray[np.float64] = command + self._trim
        return trimmed
