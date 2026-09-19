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
- **Fixed-prompt text cache**: all fixed pipeline prompts (the four plate candidates plus
  `car`) are encoded once at load and reused for every image — see
  [Fixed-prompt text cache](#fixed-prompt-text-cache).
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
├── scripts/                 # weight download, cache benchmark, XPU profiler
└── tests/                   # pytest unit tests
```

## Environment Setup

The following is the verified dependency environment for this project, including core dependencies and the full dependency list.

### Core Environment

| Component | Version |
| --- | --- |
| Python | 3.10+ recommended |
| PyTorch | 2.14.0+xpu |
| TorchVision | 0.29.0+xpu |
| PaddlePaddle | 3.3.1 |
| PaddleOCR | 3.7.0 |
| PaddleX | 3.7.2 |
| NumPy | 1.26.4 |

> Note: `torch` and `torchvision` are **XPU builds** (Intel GPU acceleration), which require Intel oneAPI runtimes (e.g., `intel-sycl-rt`, `intel-openmp`, `oneccl`).

### Full Dependency List

```
aiohappyeyeballs       2.7.1
aiohttp                3.14.3
aiosignal              1.4.0
aistudio_sdk           0.3.9
annotated-types        0.8.0
anyio                  4.15.1
attrs                  26.1.0
bce-python-sdk         0.9.79
certifi                2026.7.22
cffi                   2.1.1
chardet                7.6.0
charset-normalizer     3.5.1
click                  8.5.0
colorlog               6.12.0
crc32c                 2.9.post0
cryptography           50.0.1
dpcpp-cpp-rt            2026.1.0
einops                 0.8.2
filelock               3.32.3
frozenlist              1.8.0
fsspec                 2026.7.0
ftfy                   6.1.1
future                 1.0.0
h11                    0.16.0
hf-xet                 1.6.0
httpcore               1.0.9
httpx                  0.28.1
huggingface_hub        1.32.0
idna                   3.20
imagesize              2.0.1
impi-rt                2021.18.1
iniconfig              2.3.0
intel-cmplr-lib-rt     2026.1.0
intel-cmplr-lib-ur     2026.1.0
intel-cmplr-lic-rt     2026.1.0
intel-opencl-rt        2026.1.0
intel-openmp           2026.1.0
intel-pti              1.0.1
intel-sycl-rt          2026.1.0
iopath                 0.1.10
Jinja2                 3.1.6
lpnrecog               0.1.0       /home/kkl/platesam
MarkupSafe              3.0.3
mkl                    2026.1.0
modelscope             1.40.1
modelscope-hub         0.4.3
mpmath                 1.3.0
multidict              6.8.0
networkx               3.6.1
numpy                  1.26.4
oneccl                 2022.1.1
oneccl-devel           2022.1.1
onemkl-license         2026.1.0
onemkl-sycl-blas       2026.1.0
onemkl-sycl-dft        2026.1.0
onemkl-sycl-lapack     2026.1.0
onemkl-sycl-rng        2026.1.0
onemkl-sycl-sparse     2026.1.0
opencv-contrib-python  4.10.0.84
opencv-python-headless 4.11.0.86
opt-einsum             3.3.0
packaging              26.3
paddleocr              3.7.0
paddlepaddle           3.3.1
paddlex                3.7.2
pandas                 3.0.6
pillow                 12.3.0
pip                    26.2.1
pluggy                 1.6.0
portalocker            4.3.2
prettytable            3.18.0
propcache              0.5.4
protobuf               7.36.2
psutil                 7.2.2
py-cpuinfo             9.0.0
pyclipper              1.4.0
pycocotools            2.0.11
pycparser              3.0
pycryptodome            3.23.0
pydantic               2.13.5
pydantic_core          2.46.5
pyelftools             0.32
Pygments               2.21.0
pypdfium2              5.13.0
pytest                 9.1.1
python-bidi            0.6.11
python-dateutil        2.9.0.post0
PyYAML                 6.0.2
pyzes                  0.1.2
regex                  2026.9.10
requests               2.34.2
ruamel.yaml            0.19.1
safetensors            0.8.0
sam3                   0.1.0       /home/kkl/platesam/vendor/sam3
setuptools             83.0.0
shapely                2.1.2
six                    1.17.0
sympy                  1.14.0
tbb                    2023.1.0
tcmlib                 1.5.0
timm                   1.0.29
torch                  2.14.0+xpu
torchvision            0.29.0+xpu
tqdm                   4.70.1
triton-xpu             3.8.0
typing_extensions      4.16.0
typing-inspection      0.4.4
ujson                  6.0.0
umf                    1.1.0
urllib3                2.8.0
wcwidth                0.8.4
wheel                  0.47.0
yarl                   1.25.1
```

### Local Packages

Two packages in the list are installed from local paths. After cloning the repository, install them from their corresponding directories:

| Package | Version | Path |
| --- | --- | --- |
| lpnrecog | 0.1.0 | `/home/kkl/platesam` |
| sam3 | 0.1.0 | `/home/kkl/platesam/vendor/sam3` |

Installation example:

```bash
pip install -e .
pip install -e vendor/sam3
```

### Installation

#### Option 1: Export and Install from Requirements (Recommended)

Export the dependency list from a verified environment:

```bash
pip freeze > requirements.txt
```

Install in a new environment:

```bash
pip install -r requirements.txt
```

#### Option 2: Manually Install Core Dependencies

```bash
pip install torch==2.14.0+xpu torchvision==0.29.0+xpu --index-url https://download.pytorch.org/whl/xpu
pip install paddlepaddle==3.3.1 paddleocr==3.7.0 paddlex==3.7.2
pip install numpy==1.26.4 opencv-contrib-python==4.10.0.84 opencv-python-headless==4.11.0.86
pip install -e .
pip install -e vendor/sam3
```

### Verification

After installation, run the following commands to confirm the key dependency versions:

```bash
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
python -c "import paddle; print(paddle.__version__)"
python -c "import paddleocr, paddlex; print(paddleocr.__version__, paddlex.__version__)"
python -c "import numpy, cv2; print(numpy.__version__, cv2.__version__)"
```

If the output matches the versions listed above, the environment setup is complete.

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

Every prompt the pipeline can issue is a fixed string, so their SAM 3 text-encoder outputs
(`language_features`, `language_mask`, `language_embeds`) are computed **once** in
`PlateSegmenter.load()` via the existing `backbone.forward_text(...)` and reused by every
grounding pass — including the `car` prompt used by the zoom fallback:

`license plate` · `number plate` · `vehicle registration plate` · `car plate` · `car`

The full `segment()` path runs under `torch.inference_mode()` (matching the original
`Sam3Processor.set_text_prompt()`), so cached grounding stays inference-only. The generic path
is untouched: any custom prompt still runs the text encoder through
`Sam3Processor.set_text_prompt`, and `segmenter.use_text_cache = False` restores the original
behavior. Caching all five prompts costs ~0.3 s once at load (5 × ~60 ms) and removes ~60 ms
per prompt from every inference (~0.25 s per default 4-prompt image).

Correctness: cached vs uncached outputs match exactly on all three sample images (detection
count, boxes, scores, mask IoU = 1.0, OCR text/scores), and a full result dump — including
mask SHA-256 hashes — of the previous revision matches bit-for-bit.

#### Stage-level XPU profile

Measured with `scripts/profile_xpu_pipeline.py --runs 5` on an Intel XPU laptop
(`torch 2.14.0+xpu`, bf16 autocast, 1008² input, `guangdong_plate.jpg`). Accelerator stages
are timed with `torch.xpu.synchronize()`; values are medians over 5 runs after warmup.

| stage | single prompt | default 4 prompts | share (default) |
| --- | ---: | ---: | ---: |
| image load (`imread`) | 3.9 ms | 3.7 ms | 0.1% |
| preprocess (to PIL) | 5.4 ms | 5.4 ms | 0.2% |
| **set_image (transform + ViT)** | **1241 ms** | **1212 ms** | **46.0%** |
| grounding: `license plate` | 234 ms | 236 ms | 9.0% |
| grounding: `number plate` | — | 229 ms | 8.7% |
| grounding: `vehicle registration plate` | — | 224 ms | 8.5% |
| grounding: `car plate` | — | 228 ms | 8.7% |
| mask upsample (`interpolate`) | 0.2 ms | 0.9 ms | 0.0% |
| XPU→CPU extract (scores/boxes/masks) | 2.9 ms | 11.7 ms | 0.4% |
| filter (shape) | 1.2 ms | 3.8 ms | 0.1% |
| NMS merge (mask IoU) | 0.0 ms | 3.1 ms | 0.1% |
| **segment total** | **1475 ms** | **2159 ms** | **82.0%** |
| rectify | 2.8 ms | 2.3 ms | 0.1% |
| OCR (PaddleOCR, CPU) | 420 ms | 415 ms | 15.8% |
| **pipeline total** | **1936 ms** | **2633 ms** | 100% |

The remaining bottleneck is the 1008² vision backbone (~1.2 s) plus one ~230 ms grounding
pass per prompt. Everything after the masks leave the XPU — normalization, mask upsampling,
D2H transfer, shape filtering, mask-IoU NMS, rectification — is below 1% each, so further work
has to target the image encoder or prompt batching; this round changes neither.

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

# stage-level XPU profile (single + default prompts) with cached/uncached verification
python scripts/profile_xpu_pipeline.py --runs 5
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
