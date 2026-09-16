#!/usr/bin/env bash
# Download the native SAM 3 checkpoint (sam3.pt) used by lpnrecog.
#
# Usage:
#   bash scripts/download_sam3_weights.sh                 # ModelScope (default) -> vendor/sam3/weights/master
#   bash scripts/download_sam3_weights.sh --source hf     # HuggingFace (gated, login first)
#   bash scripts/download_sam3_weights.sh --source ms /data/weights
set -euo pipefail

SOURCE="ms"
DEST="vendor/sam3/weights/master"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source)
            SOURCE="$2"
            shift 2
            ;;
        -h|--help)
            grep '^#' "$0" | tail -n +2 | sed 's/^# \{0,1\}//' | head -8
            exit 0
            ;;
        *)
            DEST="$1"
            shift
            ;;
    esac
done

mkdir -p "$DEST"

case "$SOURCE" in
    ms)
        # ModelScope mirror, no login required, works well from mainland China.
        python - "$DEST" <<'PY'
import sys
from modelscope import snapshot_download

snapshot_download(
    "facebook/sam3",
    local_dir=sys.argv[1],
    allow_file_pattern=["sam3.pt", "config.json"],
)
PY
        ;;
    hf)
        # HuggingFace repo is gated: accept the license and run `hf auth login` first.
        python - "$DEST" <<'PY'
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

dest = Path(sys.argv[1])
for name in ("sam3.pt", "config.json"):
    src = hf_hub_download(repo_id="facebook/sam3", filename=name)
    shutil.copy(src, dest / name)
    print("saved", dest / name)
PY
        ;;
    *)
        echo "error: unknown --source '$SOURCE' (expected 'ms' or 'hf')" >&2
        exit 2
        ;;
esac

ls -lh "$DEST/sam3.pt"
