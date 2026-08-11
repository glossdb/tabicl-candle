"""Golden fixtures: the torch train-mode forward at several
(T, H, train_size), committed so the Rust port is graded without any
Python present. Stage 1 of the fidelity gate — the shapes deliberately
vary train_size, the axis the ONNX evaluation showed a traced graph
gets silently wrong.

    uv run python scripts/gen_fixtures.py
"""

import glob
import os
import sys

import numpy as np
import torch
from tabicl._model.tabicl import TabICL

HUB = os.path.expanduser("~/.cache/huggingface/hub/models--jingang--TabICL")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHAPES = [(10, 4, 7), (16, 5, 12), (24, 3, 16)]  # (rows T, features H, train)


def find(pattern: str) -> str:
    hits = glob.glob(f"{HUB}/snapshots/*/{pattern}")
    if not hits:
        sys.exit(f"no checkpoint matching {pattern} under {HUB}")
    return sorted(hits)[-1]


def main() -> None:
    for name, pattern in [("classifier", "tabicl-classifier-v2-*.ckpt"),
                          ("regressor", "tabicl-regressor-v2-*.ckpt")]:
        ckpt = torch.load(find(pattern), map_location="cpu", weights_only=True)
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
