"""Golden fixtures: the torch train-mode forward at several
(T, H, train_size), committed so the Rust port is graded without any
Python present. Stage 1 of the fidelity gate — the shapes deliberately
vary train_size, the axis the ONNX evaluation showed a traced graph
gets silently wrong.

The checkpoints download themselves on first run (see _checkpoints.py).

    uv run python scripts/gen_fixtures.py
"""

import os

import numpy as np
import torch
from tabicl._model.tabicl import TabICL

from _checkpoints import fetch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHAPES = [(10, 4, 7), (16, 5, 12), (24, 3, 16)]  # (rows T, features H, train)


def main() -> None:
    for name in ("classifier", "regressor"):
        ckpt = torch.load(fetch(name), map_location="cpu", weights_only=True)
        model = TabICL(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        model.train()  # the wrapper calls the core in train mode
        for t, h, tr in SHAPES:
            g = torch.Generator().manual_seed(t * 1000 + h * 100 + tr)
            x = torch.randn(1, t, h, generator=g)
            if name == "classifier":
                y = torch.randint(0, 3, (1, tr), generator=g).float()
            else:
                y = torch.randn(1, tr, generator=g)
            with torch.no_grad():
                out = model(x, y)
            path = f"{ROOT}/fixtures/{name}_{t}x{h}x{tr}.npz"
            np.savez_compressed(
                path,
                x=x.numpy().astype(np.float32),
                y=y.numpy().astype(np.float32),
                out=out.numpy().astype(np.float32),
            )
            print(f"{path}: out {tuple(out.shape)}")


if __name__ == "__main__":
    main()
