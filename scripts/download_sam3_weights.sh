#!/usr/bin/env bash
# Download the native SAM3 checkpoint + BPE vocab used by lpnrecog.
#
# The HuggingFace repo (facebook/sam3) is gated: accept the license and run
# `huggingface-cli login` first, or download `sam3.pt` manually.
set -euo pipefail

DEST="${1:-vendor/sam3/weights/master}"
mkdir -p "$DEST"

python - <<PY
from huggingface_hub import hf_hub_download
import shutil

for name in ("sam3.pt", "config.json"):
    path = hf_hub_download(repo_id="facebook/sam3", filename=name)
    shutil.copy(path, "$DEST/" + name)
    print("saved", "$DEST/" + name)
PY
