"""Open a PR on jingang/TabICL adding safetensors conversions of the v2
checkpoints (plus the config json each checkpoint carries), so
consumers outside torch — Rust/candle above all — can fetch tensors
without loading a pickle.

Remote names mirror the source checkpoint basenames read from
fixtures/DIGESTS, and the local weights are re-verified against the
pinned digests before anything is sent. Dry run by default:

    uv run python verify/python/upload_hub_safetensors.py             # dry run
    uv run python verify/python/upload_hub_safetensors.py --create-pr # needs `hf auth login`
"""

import hashlib
import json
import os
import sys

REPO_ID = "jingang/TabICL"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PR_TITLE = "safetensors + config for the v2 checkpoints"
PR_DESCRIPTION = """\
Adds safetensors conversions of the two v2 checkpoints, plus a small
config.json per checkpoint, so consumers outside torch (Rust/candle,
safetensors readers generally) can fetch the weights without loading a
pickle:

- tabicl-classifier-v2-20260212.safetensors / .config.json
- tabicl-regressor-v2-20260212.safetensors / .config.json

The conversion is mechanical: every tensor in the checkpoint's
state_dict saved as-is (all plain float32, no shared storage), and the
checkpoint's own `config` dict written as JSON alongside. Nothing is
renamed or transformed, so state_dict consumers and safetensors
consumers see identical tensors.

Verified against the torch forward: a candle port of the v2
architecture loads exactly these files and matches the torch
train-mode forward at ~2e-5 max abs diff (fp32) on both checkpoints,
CPU and Metal.
"""


def main() -> None:
    with open(f"{ROOT}/fixtures/DIGESTS") as fh:
        digests = json.load(fh)

    ops = []  # (local path, remote name)
    for name in ("classifier", "regressor"):
        entry = digests[name]
        local_st = f"{ROOT}/weights/tabicl-{name}.safetensors"
        actual = hashlib.sha256(open(local_st, "rb").read()).hexdigest()
        if actual != entry["sha256"]:
            sys.exit(
                f"{name}: local weights do not match the pinned digest — "
                "re-run verify/python/convert_weights.py"
            )
        base = entry["source"].removesuffix(".ckpt")
        ops.append((local_st, f"{base}.safetensors", entry["sha256"]))
        ops.append((f"{ROOT}/weights/tabicl-{name}.config.json", f"{base}.config.json", None))

    print(f"PR to {REPO_ID}: {PR_TITLE}\n")
    for local, remote, sha in ops:
        size = os.path.getsize(local) / 1e6
        line = f"  {remote}  <-  {os.path.relpath(local, ROOT)}  ({size:.1f} MB)"
        print(line if sha is None else f"{line}\n      sha256 {sha}")

    if "--create-pr" not in sys.argv[1:]:
        print("\ndry run — pass --create-pr to open the hub PR (needs `hf auth login`)")
        return

    from huggingface_hub import CommitOperationAdd, HfApi

    info = HfApi().create_commit(
        repo_id=REPO_ID,
        operations=[
            CommitOperationAdd(path_in_repo=remote, path_or_fileobj=local)
            for local, remote, _ in ops
        ],
        commit_message=PR_TITLE,
        commit_description=PR_DESCRIPTION,
        create_pr=True,
    )
    print(f"\nopened: {info.pr_url}")


if __name__ == "__main__":
    main()
