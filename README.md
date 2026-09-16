# platesam

**License plate recognition** — SAM 3 text-prompt segmentation → perspective rectification
to the plate's real shape → PaddleOCR.

Built as a course project, validated end-to-end on an **Intel XPU** laptop (WSL2, no CUDA).

```text
             ┌──────────────────────────────┐
 image ────► │ 1. SAM 3 segmentation        │  text prompt: "license plate"
             │    (open-vocabulary mask)    │  merged over several prompt candidates
             └──────────────┬───────────────┘
                            │ binary mask
             ┌──────────────▼───────────────┐
             │ 2. Rectification             │  mask → largest contour → 4 corners
             │    warp to 440×140 (mm)      │  (minAreaRect fallback, 90° auto-rotate)
             └──────────────┬───────────────┘
                            │ frontal plate crop
             ┌──────────────▼───────────────┐
             │ 3. PaddleOCR recognition     │  text normalization + validity check
             └──────────────┬───────────────┘
                            ▼
                 results.json + viz/*.jpg + crops/*.jpg
```

## Demo

<table>
<tr>
<td width="50%"><img src="docs/result_guangdong.jpg" alt="Guangdong plate"><br><code>粤CF6926</code> · text score 0.98 · det score 0.95</td>
<td width="50%"><img src="docs/result_geely.jpg" alt="Geely plate"><br><code>贵HEE253</code> · text score 0.97 · det score 0.94</td>
</tr>
</table>

Measured on the sample images (Intel XPU, bf16 autocast, ~4 GB peak memory):

| Image | Prompt | Detection | OCR | Time |
| --- | --- | --- | --- | --- |
| `assets/samples/guangdong_plate.jpg` (plate close-up) | `license plate` | 0.95 | `粤CF6926` (0.98, valid) | 2.2 s |
| `assets/samples/geely_kingkong.jpg` (car in street) | `license plate` | 0.94 | `贵HEE253` (0.97, valid) | 2.4 s |
| `assets/samples/byd_surui.jpg` (extreme side view) | `license plate` | 0.89 | `JGS503` (0.99, **invalid** — info lost) | 20 s¹ |

¹ The plate is nearly edge-on and cropped by the image border, so the zoom fallback kicks in.
The result is flagged `"valid": false` in the JSON.

## Features

- **XPU first**: runs natively on Intel XPU (`torch 2.14.0+xpu`), no CUDA required; falls back
  to CUDA/CPU automatically. Includes a documented set of upstream patches (see below).
- **Open-vocabulary segmentation**: no plate detector to train — SAM 3 finds plates from
  text prompts alone.
- **Multi-prompt merge**: several English prompt candidates are run and merged by mask IoU
  (`license plate`, `number plate`, `vehicle registration plate`, `car plate`).
- **Fixed-prompt text cache**: the `license plate` text embedding is computed once at load and
  reused for every image — see [Fixed-prompt text cache](#fixed-prompt-text-cache).
- **Coarse-to-fine fallback**: if the full-frame pass finds nothing, vehicles are located with
  the `car` prompt, cropped with margin, upscaled, and re-run; last resort is the bottom half.
- **Geometry-aware rectification**: contour → quad (`approxPolyDP`, `minAreaRect` fallback) →
  perspective warp to the canonical 440×140 mm single-row plate aspect, with 90° auto-rotation
  for vertical detections and a small inset to drop plate frames.
- **OCR post-processing**: full-width → half-width, common confusions (`I`→`1`, `O`→`0`),
  and Chinese plate-format validation (standard 7-char and new-energy 8-char patterns).
- **Usable outputs**: per-image JSON, annotated visualization with CJK labels, and rectified crops.

## Repository layout

```text
├── src/lpnrecog/            # project code (MIT)
│   ├── segmenter.py         #   SAM3 wrapper: devices, prompts, merge, zoom fallback
│   ├── rectify.py           #   mask → quad → 440×140 warp (pure OpenCV, unit-tested)
│   ├── ocr.py               #   PaddleOCR wrapper + text normalization/validation
│   ├── pipeline.py          #   end-to-end pipeline
│   ├── viz.py               #   mask overlay, box/quad, CJK-aware labels
│   ├── cli.py               #   command line interface
│   ├── device.py            #   device selection + bf16 autocast helpers
│   └── types.py             #   dataclasses (detections / results)
├── vendor/sam3/             # upstream SAM 3, nested .git removed, XPU patches applied
│   └── weights/master/      # sam3.pt checkpoint (gitignored, download separately)
├── assets/samples/          # demo images (Wikimedia Commons, see Licensing)
├── docs/                    # README figures
├── scripts/                 # weight download + text-cache benchmark
└── tests/                   # pytest unit tests
```

## Requirements

| Item | Tested / required |
| --- | --- |
| OS | Linux (developed on WSL2, kernel 6.18) |
| Hardware | Intel XPU (iGPU/dGPU with Level Zero); CUDA GPU or CPU also work |
| Python | 3.10–3.12 (tested on 3.12) |
| PyTorch | 2.14.0+xpu + torchvision 0.29.0+xpu (CUDA/CPU builds also fine) |
| Paddle | paddlepaddle 3.3.1 + paddleocr 3.7.0 (CPU) |
| Disk | ~10 GB (3.4 GB checkpoint + models + env) |
| Memory | ≥8 GB RAM; XPU inference peaks at ~4 GB |

## Installation

The reference environment is a conda env named `sam3`. To rebuild it from scratch:

```bash
conda create -n sam3 python=3.12 -y
conda activate sam3

# 1) PyTorch for Intel XPU (or use the official CUDA/CPU index for other hardware)
pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu

# 2) Vendored SAM 3 and this project (editable, no dependency resolution side effects)
pip install -e vendor/sam3 --no-deps
pip install -e . --no-deps

# 3) Runtime dependencies
pip install "numpy<2" opencv-python-headless pillow \
            paddlepaddle paddleocr \
            einops pycocotools psutil pytest
```

> **Note:** `paddleocr` pulls `opencv-contrib-python`; keep `numpy<2` (SAM 3 requires it).
> If `import sam3` fails with `ModuleNotFoundError: pkg_resources`, you are running the
> unpatched upstream code — this repo already replaced it with `importlib.resources`.

## Model weights

`lpnrecog` needs the **native checkpoint** `sam3.pt` (3.45 GB), not the HF `model.safetensors`
(that one is for the `transformers` integration and is not used here).

Expected location:

```text
vendor/sam3/weights/master/
├── sam3.pt      3.45 GB   required  (native checkpoint)
└── config.json  25.8 KB   optional  (model metadata)
```

### Option A — ModelScope (recommended, no login)

```bash
bash scripts/download_sam3_weights.sh               # default: ModelScope → vendor/sam3/weights/master
# or explicitly:
bash scripts/download_sam3_weights.sh --source ms /data/weights
```

### Option B — HuggingFace (gated)

Accept the license at <https://huggingface.co/facebook/sam3>, then:

```bash
hf auth login
bash scripts/download_sam3_weights.sh --source hf
```

### Option C — manual / existing copy

Any directory works:

```bash
export LPNRECOG_SAM3_CKPT=/path/to/sam3.pt
python -m lpnrecog --checkpoint /path/to/sam3.pt -i ...
```

### Verify

```bash
ls -lh vendor/sam3/weights/master/sam3.pt
# sha256 (ModelScope / HF mirror, this revision):
#   9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e
```

## Quick start

```bash
conda activate sam3

# whole sample directory
python -m lpnrecog -i assets/samples -o outputs/demo --conf 0.3 --prompt "license plate"

# expected:
#   geely_kingkong.jpg:  1 plate(s) -> 贵HEE253(0.97)
#   guangdong_plate.jpg: 1 plate(s) -> 粤CF6926(0.98)
#   byd_surui.jpg:       1 plate(s) -> JGS503(0.99,invalid)
#   results written to outputs/demo/results.json
```

Outputs:

```text
outputs/demo/
├── results.json            # machine-readable results
├── viz/<name>_viz.jpg      # annotated image (mask, quad, plate text)
└── crops/<name>_plate0.jpg # rectified 440×140 plate crops
```

Python API:

```python
from lpnrecog import PlateRecognitionPipeline
from lpnrecog.ocr import is_valid_plate

pipe = PlateRecognitionPipeline()          # device auto: xpu > cuda > cpu
for res in pipe.run("assets/samples/geely_kingkong.jpg"):
    print(res.text, res.text_score, res.box, is_valid_plate(res.text))
    # 贵HEE253 0.9691 (169.0, 524.2, 278.7, 585.5) True
```

## CLI reference

```text
python -m lpnrecog -i INPUT [-o OUTPUT] [options]
```

| Flag | Default | Description |
| --- | --- | --- |
| `-i, --input` | required | image file or directory (searched recursively) |
| `-o, --output` | `outputs` | output directory |
| `--prompt` | `license plate`, `number plate`, `vehicle registration plate`, `car plate` | SAM3 text prompt; repeatable |
| `--conf` | `0.4` | detection score threshold |
| `--device` | auto | `xpu` / `cuda` / `cpu` |
| `--checkpoint` | `vendor/sam3/weights/master/sam3.pt` | SAM3 checkpoint path |
| `--plate-size` | `440x140` | rectified crop size `WxH` |
| `--no-viz` | off | skip annotated images |
| `--no-crops` | off | skip rectified crops |
| `--json` | `<output>/results.json` | results JSON path |

### results.json format

```json
[
  {
    "image": "assets/samples/geely_kingkong.jpg",
    "elapsed_s": 2.39,
    "plates": [
      {
        "box": [169.0, 524.2, 278.7, 585.5],
        "det_score": 0.9414,
        "prompt": "license plate",
        "text": "贵HEE253",
        "text_score": 0.9691,
        "valid": true
      }
    ]
  }
]
```

## How it works

### 1. Segmentation (`segmenter.py`)

- SAM 3 is loaded lazily from the local checkpoint (`build_sam3_image_model`, no HF download).
- `set_image` runs once per image; every prompt then runs a cheap grounding pass and the
  detections are merged by mask IoU (score-sorted greedy NMS).
- Candidates are filtered by mask area, bounding-box aspect ratio, and mask fill ratio
  (plate-shaped detections only).
- **Fallback zoom**: when the full-frame pass returns nothing — small or foreshortened plates —
  the `car` prompt locates vehicles, each is cropped with 25% margin, upscaled to 1008 px, and
  re-run through the plate prompts; masks are mapped back to full-image coordinates.

#### Fixed-prompt text cache

The prompt `"license plate"` never changes between images, so its SAM 3 text-encoder output
(`language_features`, `language_mask`, `language_embeds`) is computed **once** in
`PlateSegmenter.load()` via the existing `backbone.forward_text(...)` and reused by every
grounding pass. The generic path is untouched: any other prompt (e.g. `number plate`, `car`)
still runs the text encoder through `Sam3Processor.set_text_prompt`.
`segmenter.use_text_cache = False` restores the original behavior.

Measured on the XPU laptop (1008² input, bf16, single `"license plate"` prompt):

| | value |
| --- | --- |
| one text-encoder pass (cost avoided per inference) | ~63 ms |
| steady-state inference, uncached | ~1.78–1.83 s |
| steady-state inference, cached | ~1.73–1.77 s |
| speedup | 1.03–1.05× |
| detection outputs | bit-identical (box/score max \\|Δ\\| = 0, mask IoU = 1.0) |
| peak XPU memory | unchanged (~3.97 GB) |

The gain is bounded by the image encoder that dominates full-frame inference; cached tensors
(a few KB) also do not increase the measured peak memory. Reproduce with
`python scripts/benchmark_text_cache.py --runs 5`.

### 2. Rectification (`rectify.py`)

- Largest contour of the mask → `approxPolyDP`, falling back to `minAreaRect`.
- Corners are ordered `tl, tr, br, bl`, shrunk 2% toward the centroid (drops plate frames),
  and warped to **440×140** — the real-world single-row blue plate (440 mm × 140 mm, ~3.14:1)
  — so OCR sees a frontal, axis-aligned plate.
- Vertical detections are rotated back to landscape automatically.

### 3. OCR (`ocr.py`)

- PaddleOCR (`lang="ch"`, doc-orientation/unwarping disabled for speed,
  `enable_mkldnn=False` to work around a paddle 3.3 + oneDNN PIR bug).
- The 180°-rotated crop is also tried (rectification cannot always determine "up"); the
  higher-scoring, format-valid result wins.
- Text is normalized (NFKC, upper-case, confusion map) and validated against Chinese plate
  patterns (standard 7-char, new-energy 8-char).

## XPU port (changes vs upstream SAM 3)

| File | Change |
| --- | --- |
| `sam3/model_builder.py` | `importlib.resources` replaces the removed `pkg_resources`; new `_best_device()` (xpu > cuda > cpu); `_setup_device_and_mode` accepts any device |
| `sam3/model/decoder.py` | removed hardcoded `device="cuda"` coordinate precompute at init; coords are built lazily on the input device |
| `sam3/model/position_encoding.py` | precomputed position-encoding cache is built on CPU and moved to the input device on forward |
| `sam3/perflib/fused.py` | `addmm_act` follows the input dtype instead of hardcoding bf16 (identical under bf16 autocast) |
| `sam3/model/sam3_image_processor.py` | `Sam3Processor` defaults its device to the model's device |

Inference on XPU uses `torch.autocast("xpu", dtype=torch.bfloat16)` (equivalent to the official
notebook's CUDA bf16); plain fp32 global attention at 1008² input would need ~18 GB and OOMs.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `ModuleNotFoundError: pkg_resources` | Use this repo's patched `vendor/sam3` (`pip install -e vendor/sam3 --no-deps`) |
| `RuntimeError: Torch not compiled with CUDA enabled` | You are on unpatched upstream code; this repo removes the hardcoded CUDA device |
| `torch.OutOfMemoryError: XPU out of memory` | Ensure inference runs under `bf16_autocast` (the pipeline does); close other GPU apps |
| `NotImplementedError: ConvertPirAttribute2RuntimeAttribute ...` (Paddle) | `PlateOCR` already sets `enable_mkldnn=False`; do not override it |
| `Can't initialize Level Zero Sysman` warning | Harmless WSL2 driver warning from `torch.xpu` |
| `CUDA is not available ... Disabling autocast` warnings from video modules | Harmless: video-only decorators; the image pipeline does not use them |
| Plate not detected on a full car photo | Lower `--conf`, or rely on the automatic vehicle-crop zoom fallback |

## Testing

```bash
pytest -m "not slow"    # rectification geometry + text post-processing (no models needed)
pytest -m slow          # PaddleOCR smoke test on a synthetic plate (downloads OCR models)

# fixed-prompt cache: output equivalence + latency + peak XPU memory
python scripts/benchmark_text_cache.py --runs 5
```

## Known limitations

- Extreme side views / plates cut by the image border lose too much information; OCR may return
  an invalid string (flagged `"valid": false`).
- Only the single-row 440×140 plate geometry is modeled; double-row yellow plates (440×220)
  and green new-energy plates would need another branch.
- PaddleOCR runs on CPU here; the XPU is used for SAM 3 only.
- No batch inference yet: images are processed one by one (model and OCR are loaded once).

## License and attribution

- `vendor/sam3/` is Meta's SAM 3 code, redistributed under the **SAM License**
  (copy kept at [`vendor/sam3/LICENSE`](vendor/sam3/LICENSE) as required). Restrictions apply
  (e.g. no military/warfare use, no reverse engineering). **Model weights are not committed.**
  If you publish research using SAM 3, acknowledge it.
- Everything outside `vendor/sam3/` (`src/`, `tests/`, `scripts/`) is **MIT** licensed,
  see [`LICENSE`](LICENSE).
- Demo images in `assets/samples/` are from Wikimedia Commons — BYD Surui, Geely King Kong Cross
  in China, and 粵 license plate of Guangdong — under their respective CC licenses, used for
  demonstration only.
- PaddleOCR is provided by the PaddlePaddle project under Apache-2.0.
