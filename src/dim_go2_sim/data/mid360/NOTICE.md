# Mid360 Continuous Firing Model

Copyright 2026 Dimensional Inc. Apache-2.0, as in the repository LICENSE.

`fourier.npz` retains the four-channel Fourier coefficients from Andrew's
DimOS PR #4441, commit `63a9684d4505f3f54d0515c83e1b5fdb29ba4bdc`,
`data/go2_sim/mid360_pattern.npz`. Source archive SHA-256:
`f4a154d8be7ba9855c8292fde5e051dda591a35d58a1d74b63484b8ef679575c`.

Only the two rotor rates were changed from 181.083626 / 9.9 Hz to
181.08313465387945 / 9.899443879866606 Hz. They were fitted to the first
four seconds of a passive G1 Mid360 recording on 2026-10-06, with the shape
coefficients frozen, then checked on held-out time and a second recording.
Point rate is 200,000/s; harmonic orders are 2 and 8. Runtime phase is zero
at simulation time zero. This is an approximation measured on one device,
not a universal factory calibration or a reflectivity model.

Method, capture hashes and residual errors:
[archived hardware comparison](https://github.com/dimensionalOS/dimos/blob/8ef292d4fc426c715b8d15a7b2f89bfc215a05f1/experiments/mid360/README.md).
The previous Livox reference table is preserved in Git at `7465a7b2f`;
it is no longer a runtime alternative.
