"""
layout_detector.py — YOLOv8-nano Whiteboard Region Detection & Cropping
========================================================================
Module Owner: [Team Member B — Student ID: XXXXXXX]

Purpose:
    1. Loads a pre-trained YOLOv8-nano model (or a fine-tuned checkpoint).
    2. Accepts an image (numpy array / file path) representing a whiteboard
       photograph or a scanned whiteboard capture.
    3. Runs single-shot object detection to localise whiteboard regions
       (Handwriting, Diagram, Arrow, Equation, Sticky Note).
    4. Mathematically crops each detected region from the original tensor and
       returns a structured list of CroppedRegion dataclass instances.

Key Equations:
    Given YOLO normalised output (x_c, y_c, w_n, h_n) and image dims (W, H):
        x_min = int((x_c - w_n/2) * W)
        y_min = int((y_c - h_n/2) * H)
        x_max = int((x_c + w_n/2) * W)
        y_max = int((y_c + h_n/2) * H)

    Crop:  region = image_tensor[y_min:y_max, x_min:x_max]

Dependencies:
    pip install ultralytics opencv-python-headless numpy
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 1. CONFIGURATION & DATA STRUCTURES
# ---------------------------------------------------------------------------

# Default model: YOLOv8-nano pre-trained on COCO
# For whiteboard-specific fine-tuning, replace with your checkpoint path.
DEFAULT_MODEL_PATH: str = "runs/detect/whiteboard_mathwriting_v1/weights/best.pt"

# Confidence threshold for keeping detections
CONF_THRESHOLD: float = 0.25

# Map integer class IDs to human-readable labels (mirrors data_prep.py)
CLASS_NAMES: dict[int, str] = {
    0: "Handwriting",
    1: "Diagram",
    2: "Arrow",
    3: "Equation",
    4: "Sticky Note",
}

# --- V2 Configuration (2-class composite model) ---
V2_MODEL_PATH: str = "model/best_v2.pt"
V2_CLASS_NAMES: dict[int, str] = {
    0: "equation",
    1: "whiteboard",
}
V2_PALETTE: dict[int, tuple[int, int, int]] = {
    0: (255, 0, 0),     # equation  — blue  (BGR for cv2)
    1: (0, 0, 255),     # whiteboard — red   (BGR for cv2)
}


@dataclasses.dataclass
class CroppedRegion:
    """Encapsulates one detected whiteboard region."""
    class_id:   int
    class_name: str
    confidence: float
    bbox_xyxy:  tuple[int, int, int, int]   # (x1, y1, x2, y2) absolute px
    crop:       np.ndarray                   # H×W×C uint8 tensor


# ---------------------------------------------------------------------------
# 2. IMAGE INGESTION (WHITEBOARD PHOTOS)
# ---------------------------------------------------------------------------

def image_file_to_array(image_path: str | Path) -> np.ndarray:
    """Load an image file (camera photo, scan) into a numpy RGB array."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def preprocess_whiteboard(
    image: np.ndarray,
    target_size: int = 640,
) -> np.ndarray:
    """
    Light preprocessing for whiteboard camera images:
        1. Resize longest edge to target_size (preserving aspect ratio).
        2. Slight contrast enhancement to improve marker visibility.

    Parameters:
        image:       RGB numpy array from a camera photo.
        target_size: Longest-edge target for YOLO input.

    Returns:
        Preprocessed RGB numpy array.
    """
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Light CLAHE contrast enhancement on the luminance channel
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(l_channel)
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return enhanced


# ---------------------------------------------------------------------------
# 3. YOLO INFERENCE
# ---------------------------------------------------------------------------

def load_yolo_model(model_path: str = DEFAULT_MODEL_PATH):
    """
    Load a YOLOv8 model via the Ultralytics API.

    The first call downloads weights automatically if `model_path` is
    one of the standard checkpoints (yolov8n.pt, yolov8s.pt, …).
    """
    from ultralytics import YOLO

    model = YOLO(model_path)
    print(f"[INFO] YOLOv8 model loaded: {model_path}")
    return model


def run_inference(
    model,
    image: np.ndarray,
    conf: float = CONF_THRESHOLD,
    class_filter: Optional[list[int]] = None,
) -> list[CroppedRegion]:
    """
    Run YOLOv8 inference on a single whiteboard image and return cropped regions.

    Pipeline:
        image (H×W×3 uint8)
            → YOLO forward pass
            → filter by confidence & class
            → mathematical crop of each bounding box

    Parameters:
        model:        Loaded YOLO model object.
        image:        RGB numpy array.
        conf:         Minimum confidence score.
        class_filter: Optional list of class IDs to keep (None = all).

    Returns:
        List[CroppedRegion] sorted top-to-bottom by y_min for
        approximate spatial-ordering preservation.
    """
    results = model.predict(source=image, conf=conf, verbose=False)

    if not results or len(results) == 0:
        print("[WARN] No detections returned by the model.")
        return []

    detections = results[0]  # single-image batch
    boxes  = detections.boxes
    H, W   = image.shape[:2]

    regions: list[CroppedRegion] = []

    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        score  = float(boxes.conf[i].item())

        # Optional class filter
        if class_filter is not None and cls_id not in class_filter:
            continue

        # Absolute pixel coordinates (xyxy format)
        x1, y1, x2, y2 = boxes.xyxy[i].tolist()

        # Clamp to image boundaries
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(W, int(x2))
        y2 = min(H, int(y2))

        # --- Mathematical crop ---
        crop = image[y1:y2, x1:x2].copy()

        regions.append(CroppedRegion(
            class_id   = cls_id,
            class_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}"),
            confidence = score,
            bbox_xyxy  = (x1, y1, x2, y2),
            crop       = crop,
        ))

    # Sort top-to-bottom, left-to-right for spatial ordering
    regions.sort(key=lambda r: (r.bbox_xyxy[1], r.bbox_xyxy[0]))

    print(f"[INFO] Detected {len(regions)} whiteboard regions.")
    return regions


# ---------------------------------------------------------------------------
# 4. VISUALISATION UTILITY
# ---------------------------------------------------------------------------

def draw_detections(
    image: np.ndarray,
    regions: list[CroppedRegion],
    output_path: str | Path = "output/detections.png",
) -> Path:
    """
    Draw bounding boxes and labels on a copy of the image and save to disk.
    Useful for qualitative evaluation.
    """
    vis = image.copy()

    # Colour palette (BGR for cv2)
    palette = {
        0: (0, 200, 0),     # Handwriting  — green
        1: (200, 0, 0),     # Diagram      — blue
        2: (0, 0, 200),     # Arrow        — red
        3: (200, 200, 0),   # Equation     — cyan
        4: (200, 0, 200),   # Sticky Note  — magenta
    }

    for r in regions:
        x1, y1, x2, y2 = r.bbox_xyxy
        colour = palette.get(r.class_id, (128, 128, 128))
        cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 2)
        label = f"{r.class_name} {r.confidence:.2f}"
        cv2.putText(vis, label, (x1, max(y1 - 6, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"[INFO] Visualisation saved to {out}")
    return out


# ---------------------------------------------------------------------------
# 5. V2 HELPERS (2-CLASS MODEL)
# ---------------------------------------------------------------------------

def filter_by_center_y(
    regions: list[CroppedRegion],
    image_height: int,
    min_ratio: float = 0.25,
    max_ratio: float = 1.0,
) -> list[CroppedRegion]:
    """
    Discard detections whose vertical center falls outside the allowed band.

    Used to suppress ceiling false positives (top 25% of image) and
    optionally floor regions (bottom portion).
    """
    filtered = []
    for r in regions:
        center_y = (r.bbox_xyxy[1] + r.bbox_xyxy[3]) / 2
        ratio = center_y / image_height
        if min_ratio <= ratio <= max_ratio:
            filtered.append(r)
    return filtered


def merge_nearby_equations(
    equations: list[CroppedRegion],
    image: np.ndarray,
    iou_threshold: float = 0.05,
    gap_ratio: float = 0.03,
) -> list[CroppedRegion]:
    """
    Merge equation bboxes that overlap or are horizontally adjacent.

    YOLO often fragments a single handwritten equation into multiple small
    detections.  This function iteratively merges boxes whose IoU exceeds
    `iou_threshold` OR whose horizontal gap is smaller than `gap_ratio`
    times the image width (and they share a similar vertical band).

    Parameters:
        equations:     List of equation CroppedRegion instances.
        image:         Original RGB image (used to re-crop after merging).
        iou_threshold: Minimum IoU to consider boxes overlapping.
        gap_ratio:     Max horizontal gap (as fraction of image width)
                       for boxes at similar height to be merged.

    Returns:
        Merged list of CroppedRegion instances with updated bboxes and crops.
    """
    if len(equations) <= 1:
        return equations

    img_h, img_w = image.shape[:2]
    max_gap = int(img_w * gap_ratio)

    # Work with mutable bbox lists
    boxes = [list(eq.bbox_xyxy) for eq in equations]
    confs = [eq.confidence for eq in equations]
    merged_flags = [False] * len(boxes)

    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            if merged_flags[i]:
                continue
            for j in range(i + 1, len(boxes)):
                if merged_flags[j]:
                    continue

                bx1, by1, bx2, by2 = boxes[i]
                cx1, cy1, cx2, cy2 = boxes[j]

                # Check IoU
                ix1 = max(bx1, cx1)
                iy1 = max(by1, cy1)
                ix2 = min(bx2, cx2)
                iy2 = min(by2, cy2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                area_b = (bx2 - bx1) * (by2 - by1)
                area_c = (cx2 - cx1) * (cy2 - cy1)
                union = area_b + area_c - inter
                iou = inter / max(union, 1)

                # Check horizontal adjacency at similar height
                h_gap = max(0, max(bx1, cx1) - min(bx2, cx2))
                v_overlap = min(by2, cy2) - max(by1, cy1)
                min_h = min(by2 - by1, cy2 - cy1)
                similar_height = v_overlap > 0.3 * max(min_h, 1)

                if iou > iou_threshold or (h_gap < max_gap and similar_height):
                    # Merge j into i
                    boxes[i] = [
                        min(bx1, cx1), min(by1, cy1),
                        max(bx2, cx2), max(by2, cy2),
                    ]
                    confs[i] = max(confs[i], confs[j])
                    merged_flags[j] = True
                    changed = True

    # Build merged regions with fresh crops
    result = []
    for i in range(len(boxes)):
        if merged_flags[i]:
            continue
        x1, y1, x2, y2 = boxes[i]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w, x2)
        y2 = min(img_h, y2)
        crop = image[y1:y2, x1:x2].copy()
        result.append(CroppedRegion(
            class_id=0,
            class_name="equation",
            confidence=confs[i],
            bbox_xyxy=(x1, y1, x2, y2),
            crop=crop,
        ))

    result.sort(key=lambda r: (r.bbox_xyxy[1], r.bbox_xyxy[0]))
    return result


def equations_inside_whiteboards(
    equations: list[CroppedRegion],
    whiteboards: list[CroppedRegion],
) -> list[CroppedRegion]:
    """
    Keep only equation detections whose center falls inside at least
    one whiteboard bounding box. If no whiteboards are detected,
    return all equations (graceful degradation).
    """
    if not whiteboards:
        return equations

    valid = []
    for eq in equations:
        eq_cx = (eq.bbox_xyxy[0] + eq.bbox_xyxy[2]) / 2
        eq_cy = (eq.bbox_xyxy[1] + eq.bbox_xyxy[3]) / 2
        for wb in whiteboards:
            wx1, wy1, wx2, wy2 = wb.bbox_xyxy
            if wx1 <= eq_cx <= wx2 and wy1 <= eq_cy <= wy2:
                valid.append(eq)
                break
    return valid


def draw_detections_v2(
    image: np.ndarray,
    regions: list[CroppedRegion],
    latex_texts: dict[int, str] | None = None,
    output_path: str | Path = "output/detections_v2.png",
) -> Path:
    """
    Draw v2 bounding boxes (red=whiteboard, blue=equation) with optional
    LaTeX text overlay above equation boxes.

    Parameters:
        image:       RGB numpy array.
        regions:     Detected regions with v2 class IDs.
        latex_texts: Optional dict mapping region index -> LaTeX string.
        output_path: Where to save the annotated image.
    """
    vis = image.copy()

    for i, r in enumerate(regions):
        x1, y1, x2, y2 = r.bbox_xyxy
        colour = V2_PALETTE.get(r.class_id, (128, 128, 128))
        thickness = 3 if r.class_id == 1 else 2  # thicker for whiteboard
        cv2.rectangle(vis, (x1, y1), (x2, y2), colour, thickness)

        label = f"{r.class_name} {r.confidence:.2f}"
        cv2.putText(vis, label, (x1, max(y1 - 6, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

        # Overlay LaTeX text if available
        if latex_texts and i in latex_texts:
            latex = latex_texts[i][:80]  # truncate for display
            cv2.putText(vis, latex, (x1, min(y2 + 18, image.shape[0] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 180, 0), 1)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"[INFO] V2 visualisation saved to {out}")
    return out


# ---------------------------------------------------------------------------
# 6. STANDALONE EXECUTION (DEMO)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import json

    input_path  = sys.argv[1] if len(sys.argv) > 1 else "data/images/whiteboard_0001.png"
    model_path  = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL_PATH

    # Load whiteboard image
    img = image_file_to_array(input_path)
    img = preprocess_whiteboard(img)

    # Detect
    model   = load_yolo_model(model_path)
    regions = run_inference(model, img)

    # Visualisation
    draw_detections(img, regions, output_path="output/detections.png")

    # JSON Output
    output = []
    for r in regions:
        output.append({
            "class_name": r.class_name,
            "confidence": round(r.confidence, 3),
            "bbox_xyxy": r.bbox_xyxy
        })
    print(json.dumps(output, indent=2))
