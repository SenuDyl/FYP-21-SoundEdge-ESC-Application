import base64
import io
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


def _save_fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def _write_bytes(path: str, data: bytes) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return path


def plot_spectrogram(spectrogram: np.ndarray) -> bytes:
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(spectrogram, cmap="jet", aspect="auto")
    ax.set_xlabel("Time Frames")
    ax.set_ylabel("Mel Frequency Bins")
    ax.set_ylim(0, spectrogram.shape[0] - 2)
    plt.colorbar(im, ax=ax)
    return _save_fig_to_png_bytes(fig)


def save_gradcam(spectrogram: np.ndarray, cam_resized: np.ndarray) -> bytes:
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(cam_resized, cmap="jet", alpha=0.6, aspect="auto")
    ax.set_xlabel("Time Frames")
    ax.set_ylabel("Mel Frequency Bins")
    ax.set_ylim(0, spectrogram.shape[0] - 2)
    plt.colorbar(im, ax=ax)
    return _save_fig_to_png_bytes(fig)


def compute_iou(box1, box2) -> float:
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)

    inter_w = max(0, xb - xa)
    inter_h = max(0, yb - ya)
    inter_area = inter_w * inter_h

    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area + 1e-8

    return inter_area / union_area


def suppress_overlapping_boxes(boxes, iou_threshold=0.3):
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept = []
    for box in boxes:
        if all(compute_iou(box, k) <= iou_threshold for k in kept):
            kept.append(box)
    return kept


def remove_nested_boxes(boxes):
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept = []
    for box in boxes:
        x, y, w, h = box
        x2, y2 = x + w, y + h

        inside = False
        for kx, ky, kw, kh in kept:
            kx2, ky2 = kx + kw, ky + kh
            if x >= kx and y >= ky and x2 <= kx2 and y2 <= ky2:
                inside = True
                break

        if not inside:
            kept.append(box)
    return kept


def merge_close_boxes(boxes, time_gap_thresh=20):
    boxes = sorted(boxes, key=lambda b: b[0])
    merged = []

    for box in boxes:
        if not merged:
            merged.append(box)
            continue

        x, y, w, h = box
        mx, my, mw, mh = merged[-1]

        if x <= mx + mw + time_gap_thresh:
            new_x = mx
            new_y = min(my, y)
            new_w = max(mx + mw, x + w) - new_x
            new_h = max(my + mh, y + h) - new_y
            merged[-1] = (new_x, new_y, new_w, new_h)
        else:
            merged.append(box)

    return merged


def _extract_boxes_from_cam(cam_resized: np.ndarray, threshold: float, padding: int):
    _, thresh = cv2.threshold(cam_resized.astype(np.float32), threshold, 1, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        thresh.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    boxes = []
    H, W = cam_resized.shape[:2]

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        x -= padding
        y -= padding
        w += 2 * padding
        h += 2 * padding

        x = max(0, x)
        y = max(0, y)
        w = min(w, W - x)
        h = min(h, H - y)

        if w > 1 and h > 1:
            boxes.append((x, y, w, h))

    boxes = remove_nested_boxes(boxes)
    boxes = suppress_overlapping_boxes(boxes, iou_threshold=0.3)
    boxes = merge_close_boxes(boxes, time_gap_thresh=20)
    return boxes


def draw_bounding_box_on_heatmap(spectrogram: np.ndarray, cam_resized: np.ndarray, threshold: float = 0.7, padding: int = 10) -> bytes:
    boxes = _extract_boxes_from_cam(cam_resized, threshold, padding)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(cam_resized, cmap="jet", alpha=0.6, aspect="auto")

    for x, y, w, h in boxes:
        ax.add_patch(
            plt.Rectangle((x, y), w, h, linewidth=2, edgecolor="red", facecolor="none")
        )

    ax.set_xlabel("Time Frames")
    ax.set_ylabel("Mel Frequency Bins")
    ax.set_ylim(0, spectrogram.shape[0] - 2)
    plt.colorbar(im, ax=ax)
    return _save_fig_to_png_bytes(fig)


def draw_bounding_box_on_spectrogram(spectrogram: np.ndarray, cam_resized: np.ndarray, threshold: float = 0.7, padding: int = 10) -> bytes:
    boxes = _extract_boxes_from_cam(cam_resized, threshold, padding)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(spectrogram, cmap="jet", aspect="auto")

    for x, y, w, h in boxes:
        ax.add_patch(
            plt.Rectangle((x, y), w, h, linewidth=2, edgecolor="red", facecolor="none")
        )

    ax.set_xlabel("Time Frames")
    ax.set_ylabel("Mel Frequency Bins")
    ax.set_ylim(0, spectrogram.shape[0] - 2)
    plt.colorbar(im, ax=ax)
    return _save_fig_to_png_bytes(fig)


def generate_and_save_images(spectrogram: np.ndarray, cam_resized: np.ndarray, file_prefix: str, output_dir: str):
    """
    Saves PNGs into: <output_dir>/visuals/
    Returns:
      encoded_spec, encoded_heatmap, encoded_spec_bb, encoded_heatmap_bb, saved_paths(dict)
    """
    visuals_dir = os.path.join(output_dir, "visuals")
    os.makedirs(visuals_dir, exist_ok=True)

    spec_png = plot_spectrogram(spectrogram)
    heat_png = save_gradcam(spectrogram, cam_resized)
    spec_bb_png = draw_bounding_box_on_spectrogram(spectrogram, cam_resized)
    heat_bb_png = draw_bounding_box_on_heatmap(spectrogram, cam_resized)

    saved_paths = {
        "spectrogram_png": _write_bytes(os.path.join(visuals_dir, f"{file_prefix}_spectrogram.png"), spec_png),
        "gradcam_png": _write_bytes(os.path.join(visuals_dir, f"{file_prefix}_gradcam.png"), heat_png),
        "spectrogram_bboxes_png": _write_bytes(os.path.join(visuals_dir, f"{file_prefix}_spectrogram_bboxes.png"), spec_bb_png),
        "gradcam_bboxes_png": _write_bytes(os.path.join(visuals_dir, f"{file_prefix}_gradcam_bboxes.png"), heat_bb_png),
    }

    encoded_spec = base64.b64encode(spec_png).decode("utf-8")
    encoded_heatmap = base64.b64encode(heat_png).decode("utf-8")
    encoded_spec_bb = base64.b64encode(spec_bb_png).decode("utf-8")
    encoded_heatmap_bb = base64.b64encode(heat_bb_png).decode("utf-8")

    return encoded_spec, encoded_heatmap, encoded_spec_bb, encoded_heatmap_bb, saved_paths
