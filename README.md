# Whiteboard Digitizer

A three-stage ML pipeline that converts whiteboard photographs into structured, formatted Markdown — combining YOLOv8 region detection, PP-OCRv3 text recognition, and Claude for semantic formatting.

---

## Architecture Overview

```text
Whiteboard Photo
      │
      ▼
┌─────────────────────────────────┐
│  Stage 1 — Layout Detection     │
│  src/detect_layout.py           │
│                                 │
│  YOLOv8-nano (fine-tuned)       │
│  Detects 5 region classes:      │
│    • Handwriting                │
│    • Diagram                    │
│    • Arrow                      │
│    • Equation                   │
│    • Sticky Note                │
│                                 │
│  Output: bounding boxes +       │
│          cropped region images  │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Stage 2 — Text Recognition     │
│  src/transcribe_ocr.py          │
│                                 │
│  PP-OCRv3 (PaddleOCR)          │
│  Runs on Handwriting +          │
│  Equation crops only            │
│                                 │
│  Preprocessing per crop:        │
│    CLAHE → denoise → threshold  │
│    → deskew → resize            │
│                                 │
│  Fallback chain:                │
│    PaddleOCR → EasyOCR →        │
│    Tesseract                    │
│                                 │
│  Output: raw transcription text │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Stage 3 — Semantic Formatting  │
│  Claude Opus 4.6 (Agent)        │
│                                 │
│  • Corrects OCR errors via      │
│    contextual inference         │
│  • Renders equations as LaTeX   │
│  • Structures output into       │
│    GitHub-flavoured Markdown    │
│                                 │
│  Output: formatted .md file     │
└─────────────────────────────────┘
```

### Key source files

| File | Purpose |
| ---- | ------- |
| [src/detect_layout.py](src/detect_layout.py) | YOLOv8 inference — loads model, runs detection, crops regions |
| [notebooks/train_layout.ipynb](notebooks/train_layout.ipynb) | Fine-tune YOLOv8 on whiteboard dataset |
| [src/transcribe_ocr.py](src/transcribe_ocr.py) | PP-OCRv3 OCR engine with whiteboard preprocessing |
| [data_prep.py](data_prep.py) | Synthetic dataset generator (30 labelled whiteboard images) |
| [tests/fetch_real_dataset.py](tests/fetch_real_dataset.py) | Download real whiteboard dataset from Roboflow or GitHub |
| [tests/merge_datasets.py](tests/merge_datasets.py) | Merge and remap classes across multiple datasets |
| [tests/test_pipeline.py](tests/test_pipeline.py) | Batch inference test across all images in `data/images/` |

---

## Getting Started

### Prerequisites

- Python 3.9+
- A free [Roboflow API key](https://app.roboflow.com/settings/api) (optional — only needed for the real whiteboard dataset)

### 1. Clone and install

```bash
git clone <repository_url>
cd mla-proj
./setup.sh
```

`setup.sh` creates a `.venv/` virtual environment and installs all dependencies from `requirements.txt`.

### 2. Activate the environment

Run this at the start of every session:

```bash
source .venv/bin/activate
```

---

## Training the Model

The default `yolov8n.pt` weights are COCO-trained and do not recognise whiteboard-specific regions. Fine-tuning on a real whiteboard dataset is required for meaningful results.

### Step 1 — Download the dataset

**With a Roboflow API key** (recommended — 1,124 real whiteboard photos with YOLO labels):

```bash
python tests/fetch_real_dataset.py --api-key YOUR_KEY
```

**Without a key** — downloads the open-source handwritten-diagram-datasets from GitHub (~500 diagram images, no bounding-box labels):

```bash
python tests/fetch_real_dataset.py --source github
```

Both commands write `data/dataset.yaml` pointing at the correct train/valid split.

### Step 2 — Fine-tune YOLOv8

Open and run the training notebook:

```bash
jupyter notebook notebooks/train_layout.ipynb
```

All hyperparameters are set in the **Configuration** cell at the top. Recommended values for MacBook Pro M1/M2/M3:

| Parameter | M1/M2/M3 | NVIDIA GPU |
| --------- | --------- | ---------- |
| `DEVICE` | `"mps"` | `"0"` |
| `BATCH` | `8` | `16` |
| `WORKERS` | `0` | `4` |
| `EPOCHS` | `50` | `50` |
| `PATIENCE` | `20` | `20` |

The best checkpoint is saved to `runs/detect/whiteboard_yolov8n/weights/best.pt`. The notebook's final cell prints the exact lines to update in `src/detect_layout.py`.

### Step 3 — Point the pipeline at the fine-tuned model

Edit [src/detect_layout.py](src/detect_layout.py) line 43:

```python
DEFAULT_MODEL_PATH = "runs/detect/whiteboard_yolov8n/weights/best.pt"
```

Also update `CLASS_NAMES` in the same file to match the `names:` block in `data/dataset.yaml`.

---

## Running Inference

### On a single image

```bash
python src/detect_layout.py data/images/<filename>
# optional: pass a custom model checkpoint as second arg
python src/detect_layout.py data/images/<filename> runs/detect/whiteboard_yolov8n/weights/best.pt
```

Output: annotated image at `output/detections.png` + JSON to stdout.

### Batch test across all images

```bash
python tests/test_pipeline.py                            # all images in data/images/
python tests/test_pipeline.py --limit 20                 # cap at 20 images
python tests/test_pipeline.py --conf 0.10                # lower confidence threshold
python tests/test_pipeline.py --model path/to/best.pt    # custom checkpoint
python tests/test_pipeline.py --no-vis                   # skip saving annotated PNGs
```

Results are written to `output/results_summary.json` and annotated images to `output/detections/`.

### Quick smoke test with synthetic data (no downloads required)

```bash
python data_prep.py                   # generate 30 synthetic labelled images
python src/detect_layout.py           # run on the first synthetic image
```

---

## Project Structure

```text
mla-proj/
├── notebooks/
│   └── train_layout.ipynb   # YOLOv8 fine-tuning notebook
├── src/
│   ├── detect_layout.py     # Stage 1: YOLOv8 detection & cropping
│   └── transcribe_ocr.py    # Stage 2: PP-OCRv3 text recognition
├── tests/
│   ├── fetch_real_dataset.py  # Dataset downloader (Roboflow / GitHub)
│   ├── merge_datasets.py      # Multi-dataset merge with class remapping
│   └── test_pipeline.py       # Batch inference tester
├── data/                      # Created at runtime
│   ├── raw/                   # Downloaded datasets (Roboflow, GitHub)
│   ├── images/                # Flat image directory for testing
│   ├── labels/                # YOLO-format label files
│   └── dataset.yaml           # Active dataset config
├── runs/                      # Training outputs (created by the notebook)
│   └── detect/
│       └── whiteboard_yolov8n/
│           └── weights/
│               ├── best.pt    # Best validation checkpoint
│               └── last.pt    # Final epoch checkpoint
├── output/                    # Inference outputs (created at runtime)
├── data_prep.py               # Synthetic dataset generator
├── requirements.txt
├── setup.sh
└── CLAUDE.md                  # Claude agent configuration
```

---

## Dependencies

| Package | Purpose |
| ------- | ------- |
| `ultralytics>=8.0` | YOLOv8 training and inference |
| `paddleocr>=2.7` | PP-OCRv3 text recognition |
| `paddlepaddle>=2.5` | PaddlePaddle backend for PaddleOCR |
| `torch>=2.0` | PyTorch (YOLOv8 backend) |
| `opencv-python-headless>=4.8` | Image preprocessing |
| `roboflow>=1.1` | Roboflow dataset downloader |

Install all at once:

```bash
pip install -r requirements.txt
```
