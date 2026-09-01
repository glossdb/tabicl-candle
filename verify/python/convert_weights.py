"""Convert the TabICL torch checkpoints to safetensors under weights/,
pinning sha256 digests into fixtures/DIGESTS (committed — the Rust
loader verifies against them). Conversion is mechanical: every tensor
is plain float32, no shared storage (verified 2026-08-11).

The checkpoints download themselves on first run (see _checkpoints.py).

    uv run python verify/python/convert_weights.py
"""

import hashlib
import json
import os

import torch
from safetensors.torch import save_file

from _checkpoints import fetch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    os.makedirs(f"{ROOT}/weights", exist_ok=True)
    digests = {}
    for name in ("classifier", "regressor"):
        src = fetch(name)
        ckpt = torch.load(src, map_location="cpu", weights_only=True)
        out = f"{ROOT}/weights/tabicl-{name}.safetensors"
        save_file({k: v.contiguous() for k, v in ckpt["state_dict"].items()}, out)
        with open(f"{ROOT}/weights/tabicl-{name}.config.json", "w") as fh:
            json.dump(ckpt["config"], fh, indent=2, default=str)
        digest = hashlib.sha256(open(out, "rb").read()).hexdigest()
        digests[name] = {"source": os.path.basename(src), "sha256": digest}
        print(f"{name}: {os.path.basename(src)} -> {out}")
        print(f"  sha256 {digest}")
    with open(f"{ROOT}/fixtures/DIGESTS", "w") as fh:
        json.dump(digests, fh, indent=2)


if __name__ == "__main__":
    main()
