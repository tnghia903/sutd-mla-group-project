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
