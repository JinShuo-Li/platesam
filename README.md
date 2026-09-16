# platesam

Course project: license plate recognition with **SAM 3** segmentation, perspective
rectification to the plate's real shape, and **PaddleOCR** text recognition.

```
image ──> SAM3 text-prompt segmentation ("license plate") ──> mask contour corners
      ──> perspective warp to 440x140 ──> PaddleOCR ──> results JSON + visualization
```

Runs on **Intel XPU** (validated on torch 2.14.0+xpu / WSL2), with CUDA/CPU fallback.

## Layout

```
├── src/lpnrecog/          # project code
│   ├── segmenter.py       # SAM3 wrapper (XPU/CUDA/CPU, multi-prompt merge, vehicle-crop fallback)
│   ├── rectify.py         # mask -> quad -> warp to 440x140 (pure cv2, unit-tested)
│   ├── ocr.py             # PaddleOCR wrapper + plate text normalization/validation
│   ├── pipeline.py        # end-to-end pipeline
│   ├── viz.py             # mask / box / CJK-aware result visualization
│   ├── cli.py             # command line entry point
│   └── device.py          # device selection (xpu > cuda > cpu) and bf16 autocast
├── vendor/sam3/           # upstream SAM3 code, .git removed, XPU patches applied
│   └── weights/master/    # sam3.pt checkpoint (gitignored, not committed)
├── assets/samples/        # demo images (Wikimedia Commons)
├── tests/                 # unit tests
└── scripts/               # helper scripts (weights download)
```

## Setup

Prebuilt conda env: `sam3` (Python 3.12 + **torch 2.14.0+xpu** + torchvision 0.29.0+xpu).

```bash
conda activate sam3

# Key dependencies when rebuilding the env from scratch (Intel XPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu
pip install -e vendor/sam3 --no-deps        # native SAM3 (editable)
pip install -e . --no-deps                  # this project
pip install paddlepaddle paddleocr einops pycocotools psutil pytest
```

Weights: `vendor/sam3/weights/master/sam3.pt` (3.4 GB, native checkpoint).
If missing, use `bash scripts/download_sam3_weights.sh` (the HF repo `facebook/sam3`
is gated, log in first) or point `LPNRECOG_SAM3_CKPT` to another path.

## Usage

```bash
conda activate sam3

# single image
python -m lpnrecog -i assets/samples/guangdong_plate.jpg -o outputs/demo

# whole directory, multiple prompts, custom threshold/size
python -m lpnrecog -i assets/samples -o outputs/demo \
    --prompt "license plate" --conf 0.3 --plate-size 440x140

# outputs: outputs/demo/results.json, outputs/demo/viz/*.jpg, outputs/demo/crops/*.jpg
```

Python API:

```python
from lpnrecog import PlateRecognitionPipeline

pipe = PlateRecognitionPipeline()  # auto device: xpu
for res in pipe.run("assets/samples/guangdong_plate.jpg"):
    print(res.text, res.text_score, res.box)
```

Tests:

```bash
pytest -m "not slow"     # pure logic (rectify / text post-processing)
pytest -m slow           # requires the PaddleOCR models
```

## XPU port (changes vs upstream SAM3)

| File | Change |
| --- | --- |
| `sam3/model_builder.py` | `importlib.resources` replaces removed `pkg_resources`; added `_best_device()` (xpu>cuda>cpu); `_setup_device_and_mode` accepts any device |
| `sam3/model/decoder.py` | no hardcoded `device="cuda"` coord precompute at init; coords are built lazily on the input device |
| `sam3/model/position_encoding.py` | precomputed cache built on CPU and moved to the input device on forward |
| `sam3/perflib/fused.py` | `addmm_act` follows the input dtype instead of hardcoding bf16 (identical under bf16 autocast) |
| `sam3/model/sam3_image_processor.py` | `Sam3Processor` defaults its device to the model's device |

Inference on XPU uses `torch.autocast("xpu", dtype=torch.bfloat16)` (equivalent to the
official notebook's CUDA bf16); fp32 global attention at 1008² input otherwise OOMs.

## Known limitations

- Extreme side views / cropped plates (e.g. `assets/samples/byd_surui.jpg`) segment
  correctly but lose too much information; OCR output may be an invalid string. The JSON
  marks this via the `valid` field.
- Only the single-row blue plate 440x140 is supported; double-row yellow and green
  new-energy plates have different dimensions.
- PaddleOCR runs on CPU (`enable_mkldnn=False` works around a paddle 3.3 + oneDNN PIR bug).

## Licensing and attribution

- `vendor/sam3/` is Meta's SAM 3 code and is redistributed under the **SAM License**
  (see `vendor/sam3/LICENSE`; a copy is kept as required by the agreement). Model weights
  are not committed. Research using SAM 3 must acknowledge its use.
- Everything outside `vendor/sam3/` (i.e. `src/`, `tests/`, `scripts/`) is MIT-licensed,
  see `LICENSE`.
- `assets/samples/*.jpg` come from Wikimedia Commons (BYD Surui / Geely King Kong Cross in
  China / 粵 license plate of Guangdong) under their respective CC licenses, demo use only.
