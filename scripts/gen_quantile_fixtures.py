"""Golden fixtures for the quantile head: run the model's own
QuantileToDistribution over the committed regressor forwards and record
the read-outs the serving path needs — monotone quantiles, mean, and
bands at the evaluation-harness alpha levels. Pure post-processing of
fixtures/regressor_*.npz; no model forward involved.

    uv run python scripts/gen_quantile_fixtures.py
"""

import glob
import os

import numpy as np
import torch
from tabicl._model.quantile_dist import QuantileToDistribution

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The band levels the evaluation harness uses (tfmeval conditional.py).
ALPHAS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

# log_prob probes: z = icdf at these levels covers both exponential
# tails, the spline interior, and exact knot landings (0.3, 0.5).
PROBE_ALPHAS = [0.0002, 0.01, 0.3, 0.5, 0.97, 0.999, 0.99995]


def main() -> None:
    q2d = QuantileToDistribution(num_quantiles=999)
    for path in sorted(glob.glob(f"{ROOT}/fixtures/regressor_*.npz")):
        raw = torch.from_numpy(np.load(path)["out"])  # (1, test, 999)
        dist = q2d(raw)
        z = dist.icdf(torch.tensor(PROBE_ALPHAS, dtype=raw.dtype))
        out_path = path.replace("regressor_", "quantile_")
        np.savez_compressed(
            out_path,
            raw=raw.numpy().astype(np.float32),
            quantiles=dist.quantiles.numpy().astype(np.float32),
            mean=dist.quantiles.mean(dim=-1).numpy().astype(np.float32),
            bands=dist.icdf(torch.tensor(ALPHAS, dtype=raw.dtype)).numpy().astype(np.float32),
            z=z.numpy().astype(np.float32),
            log_prob=dist.log_prob(z).numpy().astype(np.float32),
        )
        print(f"{out_path}: bands {tuple(dist.icdf(torch.tensor(ALPHAS)).shape)}")


if __name__ == "__main__":
    main()
