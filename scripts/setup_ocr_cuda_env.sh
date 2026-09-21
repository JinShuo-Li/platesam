#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OCR_ENV_DIR="${1:-${PROJECT_ROOT}/.venv-ocr-cuda}"
OCR_PYTHON_BIN="${OCR_PYTHON_BIN:-python3.12}"

if [[ "${OCR_ENV_DIR}" != /* ]]; then
    OCR_ENV_DIR="${PROJECT_ROOT}/${OCR_ENV_DIR}"
fi

echo "Creating PaddleOCR CUDA environment at ${OCR_ENV_DIR}"
"${OCR_PYTHON_BIN}" -m venv "${OCR_ENV_DIR}"

"${OCR_ENV_DIR}/bin/python" -m pip install \
    "paddlepaddle-gpu==3.3.1" \
    --index-url https://www.paddlepaddle.org.cn/packages/stable/cu130/

"${OCR_ENV_DIR}/bin/python" -m pip install \
    "numpy==1.26.4" \
    "paddleocr==3.7.0" \
    "paddlex[ocr-core]==3.7.0"

"${OCR_ENV_DIR}/bin/python" -m pip check
"${OCR_ENV_DIR}/bin/python" -c \
    "import paddle; assert paddle.device.is_compiled_with_cuda(); print('PaddleOCR CUDA environment is ready')"
