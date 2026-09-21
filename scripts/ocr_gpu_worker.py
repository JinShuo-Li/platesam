"""Persistent PaddleOCR GPU worker.

Requests are JSON lines read from stdin. Responses use a dedicated file
descriptor so Paddle/PaddleX logging on stdout cannot corrupt the protocol.
"""

from __future__ import annotations

import base64
import json
import os
import sys
from typing import Any, TextIO


def _send(stream: TextIO, payload: dict[str, Any]) -> None:
    stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
    stream.flush()


def _decode_image(image_b64: str):
    import cv2
    import numpy as np

    raw = base64.b64decode(image_b64, validate=True)
    encoded = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("failed to decode input image")
    return image


def _recognize(ocr, image) -> list[list[Any]]:
    pairs: list[list[Any]] = []
    for result in ocr.predict(image):
        texts = result.get("rec_texts") or []
        scores = result.get("rec_scores") or []
        for text, score in zip(texts, scores):
            pairs.append([str(text), float(score)])
    pairs.sort(key=lambda pair: pair[1], reverse=True)
    return pairs


def main() -> int:
    response_fd = os.environ.get("LPNRECOG_OCR_RESPONSE_FD")
    if response_fd is None:
        print("LPNRECOG_OCR_RESPONSE_FD is not set", file=sys.stderr)
        return 2

    response_stream = os.fdopen(
        int(response_fd), "w", encoding="utf-8", buffering=1
    )

    try:
        import paddle
        from paddleocr import PaddleOCR

        device = os.environ.get("LPNRECOG_OCR_DEVICE", "gpu:0")
        lang = os.environ.get("LPNRECOG_OCR_LANG", "ch")
        use_textline_orientation = (
            os.environ.get("LPNRECOG_OCR_TEXTLINE_ORIENTATION", "0") == "1"
        )

        if not paddle.device.is_compiled_with_cuda():
            raise RuntimeError("Paddle was not compiled with CUDA")
        if paddle.device.cuda.device_count() < 1:
            raise RuntimeError("Paddle cannot find a CUDA device")

        paddle.set_device(device)
        ocr = PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=use_textline_orientation,
            enable_mkldnn=False,
            device=device,
        )
        _send(
            response_stream,
            {
                "type": "ready",
                "device": paddle.device.get_device(),
                "paddle_version": paddle.__version__,
            },
        )
    except Exception as exc:
        _send(
            response_stream,
            {
                "type": "startup_error",
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        return 1

    for line in sys.stdin:
        request_id = None
        try:
            request = json.loads(line)
            request_id = request.get("id")

            if request.get("command") == "shutdown":
                _send(response_stream, {"type": "shutdown", "id": request_id})
                return 0

            image = _decode_image(request["image_b64"])
            _send(
                response_stream,
                {
                    "type": "result",
                    "id": request_id,
                    "results": _recognize(ocr, image),
                },
            )
        except Exception as exc:
            _send(
                response_stream,
                {
                    "type": "error",
                    "id": request_id,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
