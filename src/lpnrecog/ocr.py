"""PaddleOCR wrapper + Chinese plate text post-processing."""

from __future__ import annotations

import base64
import json
import os
import re
import selectors
import subprocess
import threading
import unicodedata
from pathlib import Path
from typing import List, Optional, Sequence, TextIO, Tuple

import cv2
import numpy as np

# Standard plates: 1 province char + 1 letter + 5 alnum (7 chars).
# New-energy plates: 1 province char + 1 letter + 6 alnum (8 chars).
_PLATE_PATTERNS = (
    re.compile(r"^[\u4e00-\u9fff][A-Z][A-Z0-9]{5}$"),
    re.compile(r"^[\u4e00-\u9fff][A-Z][A-Z0-9]{6}$"),
)

# Common confusions in plate fonts (only applied inside the alnum part).
_CONFUSIONS = str.maketrans({"I": "1", "O": "0", "Q": "0", "·": "", "-": ""})


def normalize_plate_text(text: str) -> str:
    """Normalize raw OCR output: NFKC, upper-case, strip non-plate chars."""
    text = unicodedata.normalize("NFKC", text or "")
    text = "".join(ch for ch in text if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")
    text = text.upper().replace(" ", "")
    if not text:
        return ""
    head, tail = text[:2], text[2:]
    tail = tail.translate(_CONFUSIONS)
    return head + tail


def is_valid_plate(text: str) -> bool:
    return any(p.match(text) for p in _PLATE_PATTERNS)


class PlateOCR:
    """Lazy PaddleOCR recognizer for rectified plate crops (BGR uint8)."""

    def __init__(
        self,
        lang: str = "ch",
        use_textline_orientation: bool = False,
        enable_mkldnn: bool = False,
        score_threshold: float = 0.0,
    ) -> None:
        self.lang = lang
        self.use_textline_orientation = use_textline_orientation
        # Paddle 3.3 + oneDNN has a PIR conversion bug on some CPUs; keep it off.
        self.enable_mkldnn = enable_mkldnn
        self.score_threshold = score_threshold
        self._ocr = None

    def load(self) -> "PlateOCR":
        if self._ocr is not None:
            return self
        from paddleocr import PaddleOCR

        self._ocr = PaddleOCR(
            lang=self.lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=self.use_textline_orientation,
            enable_mkldnn=self.enable_mkldnn,
        )
        return self

    def recognize(self, image_bgr: np.ndarray) -> List[Tuple[str, float]]:
        """Return ``[(text, score), ...]`` sorted by score (descending)."""
        if image_bgr is None or image_bgr.size == 0:
            return []
        self.load()
        results = self._ocr.predict(np.asarray(image_bgr))
        pairs: List[Tuple[str, float]] = []
        for res in results:
            texts = res.get("rec_texts") or []
            scores = res.get("rec_scores") or []
            for text, score in zip(texts, scores):
                score = float(score)
                if score < self.score_threshold:
                    continue
                pairs.append((normalize_plate_text(text), score))
        pairs = [(t, s) for t, s in pairs if t]
        pairs.sort(key=lambda p: p[1], reverse=True)
        return pairs

    def recognize_plate(
        self, crop_bgr: np.ndarray, rotate_180: bool = True
    ) -> Tuple[str, float]:
        """Recognize a single plate crop; optionally try the 180-degree rotation.

        Rectification cannot always tell which side of the plate is "up", so we
        try both orientations and keep the higher-scoring, valid-looking result.
        """
        candidates: Sequence[np.ndarray]
        if rotate_180:
            candidates = (crop_bgr, np.rot90(crop_bgr, 2))
        else:
            candidates = (crop_bgr,)

        best_text, best_score, best_rank = "", 0.0, -1.0
        for candidate in candidates:
            pairs = self.recognize(np.ascontiguousarray(candidate))
            if not pairs:
                continue
            text, score = pairs[0]
            rank = score + (0.05 if is_valid_plate(text) else 0.0)
            if rank > best_rank:
                best_text, best_score, best_rank = text, score, rank
        return best_text, best_score


class GPUPlateOCR(PlateOCR):
    """PaddleOCR client backed by a persistent, isolated GPU worker process."""

    def __init__(
        self,
        lang: str = "ch",
        use_textline_orientation: bool = False,
        score_threshold: float = 0.0,
        device: str = "gpu:0",
        python_executable: Optional[Path] = None,
        worker_script: Optional[Path] = None,
        startup_timeout: float = 120.0,
        request_timeout: float = 60.0,
    ) -> None:
        super().__init__(
            lang=lang,
            use_textline_orientation=use_textline_orientation,
            enable_mkldnn=False,
            score_threshold=score_threshold,
        )
        project_root = Path(__file__).resolve().parents[2]
        self.device = device
        self.python_executable = Path(
            python_executable or project_root / ".venv-ocr-cuda" / "bin" / "python"
        )
        self.worker_script = Path(
            worker_script or project_root / "scripts" / "ocr_gpu_worker.py"
        )
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout
        self._process: Optional[subprocess.Popen[str]] = None
        self._responses: Optional[TextIO] = None
        self._next_request_id = 0
        self._lock = threading.RLock()

    def _read_response(self, timeout: float) -> dict:
        if self._responses is None or self._process is None:
            raise RuntimeError("OCR GPU worker is not running")

        with selectors.DefaultSelector() as selector:
            selector.register(self._responses, selectors.EVENT_READ)
            if not selector.select(timeout):
                raise TimeoutError(
                    f"OCR GPU worker did not respond within {timeout:.1f}s"
                )

        line = self._responses.readline()
        if not line:
            return_code = self._process.poll()
            raise RuntimeError(
                f"OCR GPU worker exited unexpectedly (code={return_code})"
            )
        response = json.loads(line)
        if not isinstance(response, dict):
            raise RuntimeError(f"invalid OCR GPU worker response: {response!r}")
        return response

    def _send(self, payload: dict) -> None:
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("OCR GPU worker is not running")
        try:
            self._process.stdin.write(
                json.dumps(payload, ensure_ascii=False) + "\n"
            )
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError("failed to send request to OCR GPU worker") from exc

    def load(self) -> "GPUPlateOCR":
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return self
            if not self.python_executable.is_file():
                raise FileNotFoundError(
                    f"OCR GPU Python not found: {self.python_executable}"
                )
            if not self.worker_script.is_file():
                raise FileNotFoundError(
                    f"OCR GPU worker not found: {self.worker_script}"
                )

            response_read_fd, response_write_fd = os.pipe()
            env = os.environ.copy()
            env["LPNRECOG_OCR_RESPONSE_FD"] = str(response_write_fd)
            env["LPNRECOG_OCR_DEVICE"] = self.device
            env["LPNRECOG_OCR_LANG"] = self.lang
            env["LPNRECOG_OCR_TEXTLINE_ORIENTATION"] = (
                "1" if self.use_textline_orientation else "0"
            )

            wsl_driver_dir = Path("/usr/lib/wsl/lib")
            if wsl_driver_dir.is_dir():
                current = env.get("LD_LIBRARY_PATH")
                env["LD_LIBRARY_PATH"] = (
                    f"{wsl_driver_dir}:{current}" if current else str(wsl_driver_dir)
                )

            try:
                process = subprocess.Popen(
                    [str(self.python_executable), "-u", str(self.worker_script)],
                    stdin=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    bufsize=1,
                    pass_fds=(response_write_fd,),
                    env=env,
                )
            except Exception:
                os.close(response_read_fd)
                os.close(response_write_fd)
                raise

            os.close(response_write_fd)
            self._process = process
            self._responses = os.fdopen(
                response_read_fd, "r", encoding="utf-8", buffering=1
            )

            try:
                response = self._read_response(self.startup_timeout)
                if response.get("type") != "ready":
                    error = response.get("error", response)
                    raise RuntimeError(f"OCR GPU worker failed to start: {error}")
            except Exception:
                self.close()
                raise
            return self

    def recognize(self, image_bgr: np.ndarray) -> List[Tuple[str, float]]:
        if image_bgr is None or image_bgr.size == 0:
            return []

        with self._lock:
            self.load()
            ok, encoded = cv2.imencode(".png", np.ascontiguousarray(image_bgr))
            if not ok:
                raise ValueError("failed to encode OCR input image")

            self._next_request_id += 1
            request_id = self._next_request_id
            self._send(
                {
                    "id": request_id,
                    "image_b64": base64.b64encode(encoded.tobytes()).decode("ascii"),
                }
            )
            response = self._read_response(self.request_timeout)

            if response.get("id") != request_id:
                raise RuntimeError(
                    "OCR GPU worker returned a mismatched request id: "
                    f"expected {request_id}, got {response.get('id')}"
                )
            if response.get("type") == "error":
                raise RuntimeError(
                    f"OCR GPU worker request failed: {response.get('error')}"
                )
            if response.get("type") != "result":
                raise RuntimeError(f"unexpected OCR GPU worker response: {response}")

            pairs: List[Tuple[str, float]] = []
            for text, score_value in response.get("results", []):
                score = float(score_value)
                if score < self.score_threshold:
                    continue
                normalized = normalize_plate_text(str(text))
                if normalized:
                    pairs.append((normalized, score))
            pairs.sort(key=lambda pair: pair[1], reverse=True)
            return pairs

    def close(self) -> None:
        with self._lock:
            process = self._process
            if process is None:
                return

            try:
                if process.poll() is None:
                    try:
                        self._next_request_id += 1
                        self._send(
                            {"id": self._next_request_id, "command": "shutdown"}
                        )
                        self._read_response(min(self.request_timeout, 5.0))
                    except (OSError, RuntimeError, TimeoutError, json.JSONDecodeError):
                        pass
                if process.stdin is not None:
                    try:
                        process.stdin.close()
                    except OSError:
                        pass
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5.0)
            finally:
                if self._responses is not None:
                    self._responses.close()
                self._responses = None
                self._process = None

    def __enter__(self) -> "GPUPlateOCR":
        return self.load()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
