"""Checkpoint resolution for the oracle scripts — a helper, not a script.

The v2 checkpoints live on the Hugging Face hub (repo jingang/TabICL) and
are fetched on first use, then served from the hub cache. Which filename
counts as v2 is read off the installed tabicl wrappers, so this follows
the pinned tabicl version instead of a name copied here.

The usual hub environment applies: HF_HOME / HF_HUB_CACHE relocate the
cache, HF_TOKEN authenticates, HF_HUB_OFFLINE=1 restricts to what is
already cached.
"""

import inspect
import sys

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError
from tabicl import TabICLClassifier, TabICLRegressor

REPO_ID = "jingang/TabICL"
WRAPPERS = {"classifier": TabICLClassifier, "regressor": TabICLRegressor}


def filename(name: str) -> str:
    """The checkpoint file the installed tabicl defaults to for `name`."""
    params = inspect.signature(WRAPPERS[name].__init__).parameters
    return params["checkpoint_version"].default


def fetch(name: str) -> str:
    """Local path to the `name` checkpoint, downloading it once if needed."""
    fn = filename(name)
    try:
        return hf_hub_download(repo_id=REPO_ID, filename=fn, local_files_only=True)
    except LocalEntryNotFoundError:
        pass
    print(f"{fn} not cached — downloading ~110 MB from {REPO_ID}", flush=True)
    try:
        return hf_hub_download(repo_id=REPO_ID, filename=fn)
    except Exception as exc:  # offline, no network, hub down, no token
        sys.exit(f"could not download {fn} from {REPO_ID}: {exc}")
