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

"""The Livox Mid-360's continuous four-channel firing pattern and its range response.

Fourier geometry and return model from DimOS PR #4441 (data/mid360/NOTICE.md). Noise and
dropout approximate the real sensor; they are not a calibrated reflectivity model.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from importlib.resources import files

import numpy as np
from numpy.typing import NDArray


class _FiringPattern:
    def __init__(
        self, fast: float, slow: float, order_fast: int, order_slow: int, coefs: NDArray[np.float64]
    ) -> None:
        pairs = [
            (m, q)
            for m in range(order_fast + 1)
            for q in range(-order_slow, order_slow + 1)
            if not (m == 0 and q < 0)
        ]
        columns = sum(1 if pair == (0, 0) else 2 for pair in pairs)
        if coefs.shape != (4, columns, 3):
            raise ValueError("Mid-360 coefficients must match four channels and harmonic orders")
        self.fast, self.slow = fast, slow
        self.order_fast, self.order_slow = order_fast, order_slow
        self.m = np.array([m for m, _ in pairs])
        self.q = np.array([q + order_slow for _, q in pairs])
        self.folded = np.empty((len(pairs), 12), dtype=np.complex128)
        column = 0
        for index, pair in enumerate(pairs):
            self.folded[index] = coefs[:, column].ravel()
            column += 1
            if pair != (0, 0):
                self.folded[index] -= 1j * coefs[:, column].ravel()
                column += 1

    def directions(self, indices: NDArray[np.int64], rate: int) -> NDArray[np.float64]:
        groups, inverse = np.unique(indices // 4, return_inverse=True)
        times = groups * (4.0 / rate)
        fast = np.exp(2j * np.pi * self.fast * times)
        slow = np.exp(2j * np.pi * self.slow * times)
        ones = np.ones(len(groups))
        # Two exponentials per group; harmonics by multiplication.
        fast_pow = np.cumprod(np.column_stack([ones, *[fast] * self.order_fast]), axis=1)
        slow_pos = np.cumprod(np.column_stack([ones, *[slow] * self.order_slow]), axis=1)
        slow_pow = np.concatenate([np.conj(slow_pos[:, :0:-1]), slow_pos], axis=1)
        harmonic = fast_pow[:, self.m] * slow_pow[:, self.q]
        vectors = (harmonic @ self.folded).real.reshape(-1, 4, 3)
        result: NDArray[np.float64] = vectors[inverse, indices % 4]
        result /= np.linalg.norm(result, axis=1, keepdims=True)
        return result


@cache
def _pattern() -> _FiringPattern:
    with (files("dim_go2_sim") / "data/mid360/fourier.npz").open("rb") as f, np.load(f) as archive:
        return _FiringPattern(
            float(archive["f1"]),
            float(archive["f2"]),
            int(archive["m1"]),
            int(archive["m2"]),
            np.asarray(archive["coefs"], dtype=np.float64),
        )


@dataclass(frozen=True)
class Scan:
    """One scan's rays in the sensor frame, each with its firing time from the scan's start."""

    directions: NDArray[np.float64]
    offsets: NDArray[np.float64]
    lines: NDArray[np.uint8]


@dataclass(frozen=True)
class Mid360:
    point_rate_hz: int = 200_000
    downsample: int = 1  # keep every n-th four-laser group
    min_range: float = 0.16
    max_range: float = 40.0
    noise: bool = True
    dropout: bool = True
    seed: int = 0

    def scan(self, start: float, duration: float) -> Scan:
        count = round(duration * self.point_rate_hz) // 4 * 4
        offsets = (np.arange(0, count, 4 * self.downsample)[:, None] + np.arange(4)).ravel()
        indices = round(start * self.point_rate_hz) + offsets
        return Scan(
            directions=_pattern().directions(indices, self.point_rate_hz),
            offsets=offsets.astype(np.float64) / self.point_rate_hz,
            lines=(indices % 4).astype(np.uint8),
        )

    def measure(
        self, ranges: NDArray[np.float64], cos_incidence: NDArray[np.float64], start: float
    ) -> NDArray[np.float64]:
        """Measured ranges for true ones, -1 where nothing returns."""
        # Seeded per scan, so a dropped scan cannot shift later noise.
        rng = np.random.default_rng([self.seed, round(start * self.point_rate_hz)])
        cosine = np.clip(np.abs(cos_incidence), 0.0, 1.0)
        keep = (ranges >= self.min_range) & (ranges <= self.max_range)
        if self.dropout:
            probability = np.interp(
                np.rad2deg(np.arccos(cosine)),
                (0, 78, 79.5, 82.5, 85, 88, 90),
                (1, 1, 0.76, 0.5, 0.22, 0.08, 0),
            )
            keep &= rng.random(len(ranges)) < probability
        result = ranges.copy()
        if self.noise:
            sigma = np.hypot(0.0034, 0.00073 * ranges) / np.maximum(cosine, 0.05) ** 0.78
            result += rng.standard_normal(len(ranges)) * sigma
        keep &= (result >= self.min_range) & (result <= self.max_range)
        result[~keep] = -1
        return result
