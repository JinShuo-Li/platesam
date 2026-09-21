import sys
import textwrap

import numpy as np

from lpnrecog.ocr import GPUPlateOCR, is_valid_plate, normalize_plate_text

import pytest


def test_normalize_plate_text():
    assert normalize_plate_text("沪 A·12345") == "沪A12345"
    assert normalize_plate_text("京Ａ１２３４５") == "京A12345"
    assert normalize_plate_text(" 粤B I2345 ") == "粤B12345"  # I -> 1
    assert normalize_plate_text("") == ""


def test_is_valid_plate():
    assert is_valid_plate("沪A12345")
    assert is_valid_plate("京AD12345")  # new-energy 8 chars
    assert not is_valid_plate("A12345")
    assert not is_valid_plate("沪A123")
    assert not is_valid_plate("")


def test_gpu_plate_ocr_worker_protocol(tmp_path):
    worker = tmp_path / "fake_ocr_worker.py"
    worker.write_text(
        textwrap.dedent(
            """
            import json
            import os
            import sys

            output = os.fdopen(
                int(os.environ["LPNRECOG_OCR_RESPONSE_FD"]),
                "w",
                encoding="utf-8",
                buffering=1,
            )

            def send(payload):
                output.write(json.dumps(payload, ensure_ascii=False) + "\\n")
                output.flush()

            send({"type": "ready", "device": "fake:0"})
            for line in sys.stdin:
                request = json.loads(line)
                if request.get("command") == "shutdown":
                    send({"type": "shutdown", "id": request.get("id")})
                    break
                assert request.get("image_b64")
                send({
                    "type": "result",
                    "id": request.get("id"),
                    "results": [["沪 A·12345", 0.9], ["noise", 0.1]],
                })
            """
        ),
        encoding="utf-8",
    )

    ocr = GPUPlateOCR(
        python_executable=sys.executable,
        worker_script=worker,
        score_threshold=0.5,
        startup_timeout=5.0,
        request_timeout=5.0,
    )
    try:
        assert ocr.recognize(np.zeros((32, 96, 3), dtype=np.uint8)) == [
            ("沪A12345", 0.9)
        ]
        process = ocr._process
    finally:
        ocr.close()

    assert process is not None
    assert process.returncode == 0


@pytest.mark.slow
def test_paddleocr_on_synthetic_plate():
    from PIL import Image, ImageDraw, ImageFont

    from lpnrecog.ocr import PlateOCR

    font_path = "/home/lijs/.local/share/fonts/NotoSansSC-Bold.otf"
    try:
        font = ImageFont.truetype(font_path, 90)
    except OSError:
        pytest.skip("CJK font not available")

    img = Image.new("RGB", (440, 140), (150, 60, 0))
    ImageDraw.Draw(img).text((30, 20), "沪A12345", font=font, fill=(255, 255, 255))
    bgr = np.asarray(img)[:, :, ::-1]

    ocr = PlateOCR()
    text, score = ocr.recognize_plate(bgr)
    assert text == "沪A12345"
    assert score > 0.5
