"""PaddleOCR wrapper + Chinese plate text post-processing."""

from __future__ import annotations

import re
import unicodedata
from typing import List, Optional, Sequence, Tuple

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
