"""SAM3-based license plate segmentation (XPU/CUDA/CPU)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image

from .device import bf16_autocast, empty_cache, get_device
from .types import PlateDetection

DEFAULT_PROMPTS: tuple[str, ...] = (
    "license plate",
    "number plate",
    "vehicle registration plate",
    "car plate",
)

DEFAULT_VEHICLE_PROMPTS: tuple[str, ...] = ("car",)

#: Text prompts whose text-encoder output is computed once at load time and
#: reused for every image. Everything else goes through the normal text path.
CACHED_PROMPTS: tuple[str, ...] = DEFAULT_PROMPTS + DEFAULT_VEHICLE_PROMPTS

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CKPT = _REPO_ROOT / "vendor" / "sam3" / "weights" / "master" / "sam3.pt"
_DEFAULT_BPE = (
    _REPO_ROOT
    / "vendor"
    / "sam3"
    / "sam3"
    / "assets"
    / "bpe_simple_vocab_16e6.txt.gz"
)


def _default_checkpoint() -> Path:
    env = os.environ.get("LPNRECOG_SAM3_CKPT")
    return Path(env) if env else _DEFAULT_CKPT


class PlateSegmenter:
    """Thin wrapper around SAM3 image model + text prompts.

    The model is loaded lazily so that importing this module (e.g. for ``--help``)
    does not pay the 3.4 GB weight-loading cost.
    """

    def __init__(
        self,
        checkpoint: Optional[Union[str, Path]] = None,
        bpe_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        prompts: Sequence[str] = DEFAULT_PROMPTS,
        confidence_threshold: float = 0.4,
        merge_iou: float = 0.6,
        min_mask_area: int = 32,
        aspect_ratio_range: tuple[float, float] = (0.25, 10.0),
        min_mask_fill: float = 0.3,
        use_autocast: bool = True,
        fallback_zoom: bool = True,
        vehicle_prompts: Sequence[str] = DEFAULT_VEHICLE_PROMPTS,
        zoom_margin: float = 0.25,
        zoom_max_side: int = 1008,
        zoom_max_regions: int = 4,
    ) -> None:
        self.checkpoint = Path(checkpoint) if checkpoint else _default_checkpoint()
        self.bpe_path = Path(bpe_path) if bpe_path else _DEFAULT_BPE
        self.device = get_device(device)
        self.prompts = tuple(prompts)
        self.confidence_threshold = confidence_threshold
        self.merge_iou = merge_iou
        self.min_mask_area = min_mask_area
        self.aspect_ratio_range = aspect_ratio_range
        self.min_mask_fill = min_mask_fill
        self.use_autocast = use_autocast
        self.fallback_zoom = fallback_zoom
        self.vehicle_prompts = tuple(vehicle_prompts)
        self.zoom_margin = zoom_margin
        self.zoom_max_side = zoom_max_side
        self.zoom_max_regions = zoom_max_regions

        self.use_text_cache = True

        self._model = None
        self._processor = None
        self._text_cache: dict[str, dict[str, torch.Tensor]] = {}

    # ------------------------------------------------------------------ setup
    def load(self) -> "PlateSegmenter":
        if self._model is not None:
            return self
        if not self.checkpoint.exists():
            raise FileNotFoundError(
                f"SAM3 checkpoint not found: {self.checkpoint}. "
                "Set LPNRECOG_SAM3_CKPT or pass checkpoint=..."
            )
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        bpe = str(self.bpe_path) if self.bpe_path.exists() else None
        self._model = build_sam3_image_model(
            bpe_path=bpe,
            device=self.device,
            checkpoint_path=str(self.checkpoint),
            load_from_HF=False,
        )
        self._processor = Sam3Processor(
            self._model, confidence_threshold=self.confidence_threshold
        )
        for prompt in CACHED_PROMPTS:
            self._cache_text_prompt(prompt)
        empty_cache(self.device)
        return self

    def _cache_text_prompt(self, prompt: str) -> None:
        """Run the text encoder once and keep all outputs for reuse."""
        with torch.inference_mode(), bf16_autocast(
            self.device, enabled=self.use_autocast
        ):
            text_outputs = self._model.backbone.forward_text(
                [prompt], device=self.device
            )
        self._text_cache[prompt] = text_outputs

    @property
    def processor(self):
        self.load()
        return self._processor

    # -------------------------------------------------------------- inference
    def segment(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        prompts: Optional[Sequence[str]] = None,
    ) -> List[PlateDetection]:
        """Detect license plates; returns merged, filtered detections."""
        self.load()
        prompts = tuple(prompts) if prompts else self.prompts

        with torch.inference_mode(), bf16_autocast(
            self.device, enabled=self.use_autocast
        ):
            pil_image = _to_pil(image)
            detections = self._segment_pass(pil_image, prompts)
            detections = [d for d in detections if self._keep(d)]
            if not detections and self.fallback_zoom:
                detections = self._fallback_zoom(pil_image, prompts)

            detections = self._merge(detections)
            detections.sort(key=lambda d: d.score, reverse=True)
            return detections

    def _segment_pass(
        self, pil_image: Image.Image, prompts: Sequence[str]
    ) -> List[PlateDetection]:
        """Run all prompts on one image; call under autocast."""
        state = self.processor.set_image(pil_image)
        detections: List[PlateDetection] = []
        for prompt in prompts:
            self.processor.reset_all_prompts(state)
            state = self._set_text_prompt(state, prompt)
            detections.extend(self._extract(state, prompt))
        return detections

    def _set_text_prompt(self, state, prompt: str):
        """Like ``Sam3Processor.set_text_prompt`` but reuse cached text features."""
        cached = self._text_cache.get(prompt)
        if not self.use_text_cache or cached is None:
            return self.processor.set_text_prompt(prompt=prompt, state=state)
        state["backbone_out"].update(cached)
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.processor.model._get_dummy_prompt()
        return self.processor._forward_grounding(state)

    def _fallback_zoom(
        self, pil_image: Image.Image, prompts: Sequence[str]
    ) -> List[PlateDetection]:
        """Coarse-to-fine fallback: locate vehicles, crop-zoom, re-run plate prompts.

        Small/foreshortened plates that the full-frame pass misses become much
        easier once the surrounding vehicle is cropped and upscaled.
        """
        width, height = pil_image.size

        vehicle_boxes = self._vehicle_boxes(pil_image)
        if not vehicle_boxes:
            # Last resort: zoom into the bottom half where plates usually live.
            vehicle_boxes = [(0, height // 2, width, height)]

        results: List[PlateDetection] = []
        for vx0, vy0, vx1, vy1 in vehicle_boxes[: self.zoom_max_regions]:
            mx = (vx1 - vx0) * self.zoom_margin
            my = (vy1 - vy0) * self.zoom_margin
            x0 = max(0, int(vx0 - mx))
            y0 = max(0, int(vy0 - my))
            x1 = min(width, int(vx1 + mx))
            y1 = min(height, int(vy1 + my))
            if x1 - x0 < 32 or y1 - y0 < 16:
                continue
            crop = pil_image.crop((x0, y0, x1, y1))
            scale = self.zoom_max_side / max(crop.size)
            if scale > 1:
                crop = crop.resize(
                    (round(crop.width * scale), round(crop.height * scale)),
                    Image.BICUBIC,
                )
            crop_dets = self._segment_pass(crop, prompts)
            for det in crop_dets:
                if not self._keep(det):
                    continue
                mask = det.mask
                if scale != 1:
                    mask = (
                        np.asarray(
                            Image.fromarray(mask).resize(
                                (x1 - x0, y1 - y0), Image.NEAREST
                            )
                        )
                        > 0
                    )
                mask_full = np.zeros((height, width), dtype=bool)
                h = min(mask.shape[0], height - y0)
                w = min(mask.shape[1], width - x0)
                mask_full[y0 : y0 + h, x0 : x0 + w] = mask[:h, :w]
                bx0, by0, bx1, by1 = det.box
                results.append(
                    PlateDetection(
                        box=(bx0 + x0, by0 + y0, bx1 + x0, by1 + y0),
                        score=det.score,
                        mask=mask_full,
                        prompt=f"{det.prompt}@zoom",
                    )
                )
        return self._merge(results)

    def _vehicle_boxes(
        self, pil_image: Image.Image
    ) -> List[tuple[float, float, float, float]]:
        state = self.processor.set_image(pil_image)
        boxes: List[tuple[float, float, float, float]] = []
        for prompt in self.vehicle_prompts:
            self.processor.reset_all_prompts(state)
            state = self._set_text_prompt(state, prompt)
            scores = state["scores"].float().cpu().tolist()
            for score, box in zip(scores, state["boxes"].float().cpu().tolist()):
                if score < self.confidence_threshold:
                    continue
                boxes.append(tuple(float(v) for v in box))
        # Greedy box-NMS so overlapping car detections are not zoomed twice.
        boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
        kept: List[tuple[float, float, float, float]] = []
        for box in boxes:
            if any(_box_iou(box, k) > 0.5 for k in kept):
                continue
            kept.append(box)
        return kept


    def _extract(self, state, prompt: str) -> List[PlateDetection]:
        scores = state["scores"].float().cpu()
        boxes = state["boxes"].float().cpu()
        masks = state["masks"].cpu().numpy()
        out = []
        for score, box, mask in zip(scores.tolist(), boxes.tolist(), masks):
            mask = mask if mask.ndim == 2 else mask[0]
            out.append(
                PlateDetection(
                    box=tuple(float(v) for v in box),
                    score=float(score),
                    mask=mask.astype(bool),
                    prompt=prompt,
                )
            )
        return out

    def _keep(self, det: PlateDetection) -> bool:
        x0, y0, x1, y1 = det.box
        w, h = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
        ratio = w / h
        lo, hi = self.aspect_ratio_range
        if det.area < self.min_mask_area:
            return False
        if not lo <= ratio <= hi:
            return False
        # The mask should fill a decent share of its bounding box (plate-shaped).
        return det.area / (w * h) >= self.min_mask_fill

    @staticmethod
    def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
        inter = np.logical_and(a, b).sum()
        if inter == 0:
            return 0.0
        union = np.logical_or(a, b).sum()
        return float(inter) / float(union)

    def _merge(self, detections: List[PlateDetection]) -> List[PlateDetection]:
        detections = sorted(detections, key=lambda d: d.score, reverse=True)
        kept: List[PlateDetection] = []
        for det in detections:
            if any(self._mask_iou(det.mask, k.mask) > self.merge_iou for k in kept):
                continue
            kept.append(det)
        return kept


def _box_iou(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-6)


def _to_pil(image: Union[str, Path, Image.Image, np.ndarray]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    array = np.asarray(image)
    if array.ndim == 2:
        return Image.fromarray(array).convert("RGB")
    if array.shape[2] == 4:
        return Image.fromarray(array).convert("RGB")
    return Image.fromarray(array[:, :, ::-1]).convert("RGB")  # BGR -> RGB
