# Whiteboard Digitiser — v2

A three-stage ML pipeline that converts whiteboard photographs into structured, formatted Markdown with LaTeX equations — combining YOLOv8 region detection, math-specific OCR, and a post-processing filter chain that suppresses false positives.

---

## Architecture

```text
Whiteboard Photo
      │
      ▼
┌──────────────────────────────────────┐
│  Stage 1 — Layout Detection          │
│  src/infer_v2.py                     │
│                                      │
│  YOLOv8-nano (fine-tuned, 2 classes) │
│    • equation                        │
│    • whiteboard                      │
│                                      │
│  Post-processing:                    │
│    1. Ceiling FP filter              │
│       (drop detections in top 25%)   │
│    2. Bbox merge                     │
│       (combine fragmented equations) │
│    3. Containment filter             │
│       (keep equations inside wb)     │
│                                      │
│  Output: equation crops              │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  Stage 2 — Math OCR                  │
│  src/transcribe_ocr.py               │
│                                      │
│  MathOCREngine fallback chain:       │
│    pix2tex  (LaTeX output)   ──┐     │
│    TrOCR    (handwriting)    ──┤     │
│    PaddleOCR (general text)  ──┘     │
│                                      │
│  Per-crop preprocessing:             │
│    CLAHE → Otsu binarize → upscale   │
│                                      │
│  Output: LaTeX / plain text          │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  Output                              │
│                                      │
│  • Annotated PNG with bbox overlay   │
│  • Markdown file with $$LaTeX$$      │
└──────────────────────────────────────┘
```

### Key source files

| File | Purpose |
| ---- | ------- |
| [src/infer_v2.py](src/infer_v2.py) | V2 pipeline entry point — detection → post-processing → OCR → Markdown |
| [src/detect_layout.py](src/detect_layout.py) | YOLOv8 inference, bbox helpers (`filter_by_center_y`, `merge_nearby_equations`, `draw_detections_v2`) |
| [src/transcribe_ocr.py](src/transcribe_ocr.py) | `MathOCREngine` with pix2tex → TrOCR → PaddleOCR fallback chain |
| [tests/generate_composite.py](tests/generate_composite.py) | Composite dataset generator — pastes MathWriting equations onto Roboflow whiteboard scenes |
| [tests/train_v2.py](tests/train_v2.py) | Fine-tune YOLOv8 locally (Apple Silicon) |
| [tests/validate_v2.py](tests/validate_v2.py) | Validation suite — checks ceiling FPs, containment, and LaTeX parseability |
| [notebooks/train_v2_colab.ipynb](notebooks/train_v2_colab.ipynb) | End-to-end training + pipeline demo on Google Colab |
| [tests/fetch_real_dataset.py](tests/fetch_real_dataset.py) | Download MathWriting or Roboflow whiteboard datasets |

---

## Quick Start

Run inference using the pre-trained model (`model/best_v2.pt` is included in the repo):

### 1. Clone and install

```bash
git clone <repository_url>
cd mla-proj
./setup.sh
source .venv/bin/activate
```

`setup.sh` creates a `.venv/` virtual environment and installs all dependencies from `requirements.txt`.

### 2. Install math OCR models

pix2tex and TrOCR are not bundled in `requirements.txt` because they are large (~300 MB combined). Install them once:

```bash
pip install "pix2tex[gui]" transformers
```

### 3. Run inference

```bash
python src/infer_v2.py --image data/images/your_photo.jpg
```

Output files:

| File | Description |
| ---- | ----------- |
| `output/<name>_v2.md` | Markdown with equations in `$$...$$` LaTeX blocks |
| `output/<name>_v2_det.png` | Annotated image with whiteboard (red) and equation (blue) bboxes |

Optional flags:

```bash
python src/infer_v2.py --image path/to/photo.jpg --conf 0.20   # lower detection threshold
python src/infer_v2.py --image path/to/photo.jpg --no-vis      # skip saving annotated image
python src/infer_v2.py --image path/to/photo.jpg --model path/to/custom.pt
```

---

## Retraining the Model

Follow these steps to regenerate the composite dataset and fine-tune from scratch.

### Step 1 — Download datasets

**MathWriting** (equation glyphs, required):

```bash
python tests/fetch_real_dataset.py --source mathwriting
# Downloads to data/raw/mathwriting/
```

**Roboflow whiteboard scenes** (requires a free [Roboflow API key](https://app.roboflow.com/settings/api)):

```bash
python tests/fetch_real_dataset.py --source roboflow --api-key YOUR_KEY
# Downloads ~1,124 real whiteboard photos to data/raw/roboflow_whiteboard/
```

### Step 2 — Generate composite dataset

Pastes MathWriting equation glyphs onto Roboflow whiteboard scene images, producing a 2-class dataset (`equation` + `whiteboard`) with correct bounding boxes:

```bash
python tests/generate_composite.py \
  --roboflow-dir data/raw/roboflow_whiteboard \
  --mathwriting-dir data/raw/mathwriting \
  --synthetic-negatives 40 \
  --output-dir data/raw/composite
```

This creates ~1,900 images in `data/raw/composite/{train,valid}/{images,labels}/` and writes `data/dataset_v2.yaml`.

### Step 3 — Train

#### Option A: Google Colab (recommended — free T4 GPU, ~25 min)

```bash
# Run locally to prepare the upload
cd data/raw && zip -r composite.zip composite/
```

Then:

1. Upload `composite.zip` to `My Drive/mla-proj/` in Google Drive
2. Open [`notebooks/train_v2_colab.ipynb`](notebooks/train_v2_colab.ipynb) in Colab
3. Set **Runtime → Change runtime type → GPU (T4)**
4. Run all cells — training, evaluation, and the full pipeline demo are included
5. Download `best.pt` from the final cell and place it at `model/best_v2.pt`

#### Option B: Local training (Apple Silicon MPS)

```bash
python tests/train_v2.py --epochs 40 --batch 8 --device mps
# Best checkpoint is automatically copied to model/best_v2.pt
```

Hyperparameters:

| Parameter | Apple Silicon | NVIDIA GPU |
| --------- | ------------- | ---------- |
| `--device` | `mps` | `cuda` |
| `--batch` | `8` | `16` |
| `--epochs` | `40` | `40` |
| `--imgsz` | `640` | `640` |

### Step 4 — Run inference

```bash
python src/infer_v2.py --image path/to/whiteboard.jpg
```

### Step 5 — Validate

```bash
python tests/validate_v2.py
```

Checks per image:

- Zero detections with centre_y in the top 25% of the image (ceiling false positive test)
- Every equation bbox centre falls inside a detected whiteboard bbox
- LaTeX output parseable by `sympy`

Output: `output/validation/*.png` + `output/validation/validation_report.json`

---

## Project Structure

```text
mla-proj/
├── src/
│   ├── infer_v2.py              # V2 pipeline entry point (detection + OCR + output)
│   ├── detect_layout.py         # YOLOv8 inference + bbox post-processing helpers
│   └── transcribe_ocr.py        # MathOCREngine (pix2tex / TrOCR / PaddleOCR)
├── tests/
│   ├── fetch_real_dataset.py    # Dataset downloader (MathWriting, Roboflow)
│   ├── generate_composite.py    # Composite dataset generator
│   ├── train_v2.py              # Local fine-tuning script
│   ├── validate_v2.py           # Validation suite
│   ├── test_math_ocr.py         # Math OCR backend comparison
│   └── test_pipeline.py         # Batch inference tester
├── notebooks/
│   ├── train_v2_colab.ipynb     # Colab training + full pipeline demo
│   ├── train_layout.ipynb       # Legacy v1 training notebook
│   └── yolov8n.pt               # COCO-pretrained base weights
├── model/
│   └── best_v2.pt               # Fine-tuned 2-class checkpoint (included)
├── data/
│   ├── dataset_v2.yaml          # 2-class dataset config (equation, whiteboard)
│   ├── images/                  # Test whiteboard photos
│   └── raw/                     # Downloaded datasets (git-ignored)
│       ├── composite/           # Generated composite training data
│       ├── roboflow_whiteboard/ # Roboflow scenes
│       └── mathwriting/         # MathWriting equation glyphs
├── runs/                        # Training outputs (git-ignored)
├── output/                      # Inference outputs (git-ignored)
├── website/                     # GitHub Pages interactive demo
├── report/                      # Academic report (LaTeX source + PDF)
├── requirements.txt
├── setup.sh
└── CLAUDE.md                    # Claude agent formatting guidelines
```

---

## Python Walkthrough — Core Ideas Step by Step

This section walks through each key idea in the pipeline with self-contained Python snippets you can run independently.

### Step 1 — CTC Alphabet and Token Encoding

The model operates on a token-level vocabulary (not raw characters). Index 0 is always the CTC blank token `ε`.

```python
from src.ctc.charset import CHARSET, BLANK_IDX, NUM_CLASSES

print(f"Vocabulary size: {NUM_CLASSES}")   # 174
print(f"Blank index:     {BLANK_IDX}")     # 0
print(f"First 10 tokens: {CHARSET[:10]}")  # ['ε', '0', '1', ..., '9']

# Encode a LaTeX string token-by-token
from src.ctc.charset import LatexTokenizer

tok = LatexTokenizer()
indices = tok.encode(r"x^{2}")
print(f"Encoded: {indices}")              # e.g. [59, 94, 53, 95]
decoded = tok.decode(indices)
print(f"Decoded: {decoded}")              # 'x^{2}'
```

---

### Step 2 — CRNN Model (CNN → BiLSTM → log-softmax)

The CRNN maps a fixed-height image strip `(B, 1, 32, W)` to a sequence of per-timestep class distributions `(B, T, |Σ'|)`.

```python
import torch
from src.ctc.model import CRNN

model = CRNN(num_classes=174, hidden_size=256)
model.eval()

# Simulate a batch of 2 images, width 128
x = torch.rand(2, 1, 32, 128)

with torch.no_grad():
    log_probs = model(x)          # (2, T, 174)  — log-softmax output
    probs     = model.get_probs(x)  # (2, T, 174)  — softmax output

print(f"T (timesteps) = {log_probs.shape[1]}")   # ≈ 31 for W=128
print(f"Each value in [−∞, 0]: {log_probs.max().item():.3f}")
```

The CNN collapses the height dimension to 1 via pooling; each remaining column becomes one timestep for the LSTM.

---

### Step 3 — CTC Loss (training)

`nn.CTCLoss` marginalises over all alignments `π ∈ B⁻¹(l)` simultaneously:

$$P(l \mid Y) = \sum_{\pi \in B^{-1}(l)} \prod_{t=1}^{T} y_{\pi_t}^t$$

```python
import torch
import torch.nn as nn
from src.ctc.model import CRNN
from src.ctc.charset import LatexTokenizer

model = CRNN(num_classes=174)
ctc_loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
tok = LatexTokenizer()

# One training example: image width 128, target "x^{2}"
x = torch.rand(1, 1, 32, 128)
log_probs = model(x)                        # (1, T, 174)
log_probs_t = log_probs.permute(1, 0, 2)   # CTCLoss expects (T, B, C)

target  = torch.tensor(tok.encode(r"x^{2}"), dtype=torch.long)
T       = log_probs_t.shape[0]

loss = ctc_loss(
    log_probs_t,
    target.unsqueeze(0),          # (B=1, label_len)
    torch.tensor([T]),            # input lengths
    torch.tensor([len(target)]),  # target lengths
)
print(f"CTC loss: {loss.item():.4f}")
```

---

### Step 4 — CTC Greedy Decoding (inference)

At inference time, take the argmax at each timestep, then apply the collapsing function B:

1. Remove consecutive duplicates.
2. Remove all blank tokens.

```python
import numpy as np
from src.ctc.decode import ctc_collapse_sequence, ctc_greedy_decode

# Simulated raw alignment from argmax: [0, 0, 5, 5, 0, 12, 12, 0]
#   0 = blank ε,  5 = 'x',  12 = '2'
raw_path = [0, 0, 5, 5, 0, 12, 12, 0]

idx_to_char = {0: "ε", 5: "x", 12: "2"}
collapsed = ctc_collapse_sequence(raw_path, blank_idx=0, idx_to_char=idx_to_char)
print(f"Decoded: '{collapsed}'")   # 'x2'

# From a full probability matrix (T=8, C=174)
probs = np.zeros((8, 174))
for t, idx in enumerate(raw_path):
    probs[t, idx] = 0.9            # dominant class at each step

decoded, greedy_path = ctc_greedy_decode(probs, blank_idx=0, idx_to_char=idx_to_char)
print(f"Greedy decoded: '{decoded}'")     # 'x2'
print(f"Greedy path:    {greedy_path}")   # [0 0 5 5 0 12 12 0]
```

---

### Step 5 — Beam Search Decoding

Beam search keeps the top-k hypotheses at each step instead of committing to the single argmax, recovering sequences that greedy misses.

```python
import numpy as np
from src.ctc.decode import ctc_beam_search

# Probability matrix: T=6 timesteps, 5 classes (0=blank, 1='a', 2='b', 3='c', 4='d')
probs = np.array([
    [0.6, 0.2, 0.1, 0.05, 0.05],  # t=0 — strong blank
    [0.1, 0.7, 0.1, 0.05, 0.05],  # t=1 — 'a'
    [0.6, 0.1, 0.2, 0.05, 0.05],  # t=2 — blank
    [0.1, 0.1, 0.7, 0.05, 0.05],  # t=3 — 'b'
    [0.6, 0.1, 0.1, 0.15, 0.05],  # t=4 — blank
    [0.1, 0.1, 0.1, 0.1,  0.6 ],  # t=5 — 'd'
])

idx_to_char = {0: "ε", 1: "a", 2: "b", 3: "c", 4: "d"}

results = ctc_beam_search(probs, blank_idx=0, beam_width=3, idx_to_char=idx_to_char)
for label, log_prob in results:
    print(f"  '{label}'  log_prob={log_prob:.3f}")
# top result: 'abd'
```

---

### Step 6 — Image Preprocessing for CRNN Input

Raw equation crops are noisy. The preprocessing pipeline standardises them before feeding the CNN:

```python
import cv2
import numpy as np
from src.ctc.infer import preprocess_for_ctc

# Load any equation crop (e.g. a YOLO output crop)
crop = cv2.imread("data/images/sample_equation.jpg")
crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

# Pipeline: white padding → CLAHE enhancement → grayscale → resize to H=32
gray_32 = preprocess_for_ctc(crop_rgb, target_height=32)
print(f"Output shape: {gray_32.shape}")   # (32, W) — variable width
```

---

### Step 7 — End-to-End Inference with the Trained Checkpoint

Load the checkpoint and run recognition on an equation crop:

```python
import cv2
from src.ctc.infer import CTCRecogniser

rec = CTCRecogniser("model/ctc_best.pt")   # loads CRNN weights

crop = cv2.imread("data/images/sample_equation.jpg")
crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

latex, probs, greedy_indices = rec.recognise(crop_rgb)

print(f"LaTeX:  {latex}")
print(f"probs shape:          {probs.shape}")           # (T, 174)
print(f"greedy_indices shape: {greedy_indices.shape}")  # (T,)
```

`probs` is the full `(T, num_classes)` probability matrix — pass it to `ctc_beam_search` or visualise it as a heatmap to inspect what the model attends to at each timestep.

---

### Step 8 — YOLOv8 Region Detection + Post-processing

Before OCR, the pipeline detects `equation` and `whiteboard` bounding boxes and filters false positives:

```python
from ultralytics import YOLO
from src.detect_layout import (
    image_file_to_array,
    run_inference,
    filter_by_center_y,
    merge_nearby_equations,
    equations_inside_whiteboards,
)

yolo = YOLO("model/best_v2.pt")
img  = image_file_to_array("data/images/your_photo.jpg")   # returns RGB ndarray
H, W = img.shape[:2]

# Run YOLO — returns list[CroppedRegion] sorted top-to-bottom
regions     = run_inference(yolo, img, conf=0.25)
equations   = [r for r in regions if r.class_id == 0]   # class 0 = equation
whiteboards = [r for r in regions if r.class_id == 1]   # class 1 = whiteboard

# Filter 1: drop equations whose centre is in the top 25% (ceiling artefacts)
equations = filter_by_center_y(equations, image_height=H, min_ratio=0.25)

# Filter 2: merge fragmented equation boxes that are close together
equations = merge_nearby_equations(equations, img, gap_ratio=0.03)

# Filter 3: keep only equations whose centre falls inside a whiteboard box
equations = equations_inside_whiteboards(equations, whiteboards)

print(f"Equations detected: {len(equations)}")
print(f"Whiteboard regions: {len(whiteboards)}")
```

---

### Step 9 — Full Pipeline (Detection → OCR → Markdown)

```python
from src.detect_layout import (
    image_file_to_array,
    run_inference,
    filter_by_center_y,
    merge_nearby_equations,
    equations_inside_whiteboards,
)
from src.ctc.infer import CTCRecogniser, results_to_markdown
from ultralytics import YOLO

yolo = YOLO("model/best_v2.pt")
rec  = CTCRecogniser("model/ctc_best.pt")

img  = image_file_to_array("data/images/whiteboard.jpg")
H, W = img.shape[:2]

regions     = run_inference(yolo, img)
equations   = [r for r in regions if r.class_id == 0]
whiteboards = [r for r in regions if r.class_id == 1]
equations   = filter_by_center_y(equations, H, min_ratio=0.25)
equations   = merge_nearby_equations(equations, img, gap_ratio=0.03)
equations   = equations_inside_whiteboards(equations, whiteboards)

latex_results = []
for eq in equations:
    latex, _, _ = rec.recognise(eq.crop)   # eq.crop is already the RGB equation patch
    latex_results.append((latex, ""))      # (ctc_output, fallback)

md = results_to_markdown(equations, latex_results)
print(md)
```

Output is a Markdown string with each detected equation rendered inside `$$...$$` LaTeX fences.

---

## Dependencies

| Package | Purpose |
| ------- | ------- |
| `ultralytics>=8.0` | YOLOv8 training and inference |
| `torch>=2.0` | PyTorch backend |
| `pix2tex>=0.1.1` | Primary math OCR — outputs LaTeX directly |
| `transformers>=4.30` | TrOCR handwriting model |
| `paddleocr>=2.7` | Fallback OCR engine |
| `paddlepaddle>=2.5` | PaddlePaddle backend |
| `opencv-python-headless>=4.8` | Image preprocessing |
| `roboflow>=1.1` | Roboflow dataset downloader |

Install all base dependencies:

```bash
pip install -r requirements.txt
```

Install math OCR models separately (large downloads):

```bash
pip install "pix2tex[gui]" transformers
```
