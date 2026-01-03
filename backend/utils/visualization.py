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


def plot_spectrogram(spectrogram: np.ndarray) -> bytes:
    """
    spectrogram: np.ndarray [F, T]
    returns: PNG bytes
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(spectrogram, cmap="jet", aspect="auto")
    ax.set_xlabel("Time Frames")
    ax.set_ylabel("Mel Frequency Bins")
    ax.set_ylim(0, spectrogram.shape[0] - 2)
    plt.colorbar(im, ax=ax)
    return _save_fig_to_png_bytes(fig)


def save_gradcam(spectrogram: np.ndarray, cam_resized: np.ndarray) -> bytes:
    """
    cam_resized: np.ndarray [F, T] already normalized to [0,1] and already resized to spectrogram shape
    returns: PNG bytes
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(cam_resized, cmap="jet", alpha=0.6, aspect="auto")
    ax.set_xlabel("Time Frames")
    ax.set_ylabel("Mel Frequency Bins")
    ax.set_ylim(0, spectrogram.shape[0] - 2)
    plt.colorbar(im, ax=ax)
    return _save_fig_to_png_bytes(fig)


def overlay_gradcam_on_spectrogram(spectrogram: np.ndarray, cam_resized: np.ndarray) -> bytes:
    """
    cam_resized: np.ndarray [F, T] already normalized and resized
    returns: PNG bytes
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(spectrogram, cmap="jet", aspect="auto")
    im = ax.imshow(cam_resized, cmap="jet", alpha=0.3, aspect="auto")
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


def _extract_boxes_from_cam(
    cam_resized: np.ndarray,
    threshold: float,
    padding: int,
):
    """
    cam_resized: np.ndarray [F, T] normalized 0..1
    Returns list of (x, y, w, h) in image coordinates (x=time, y=freq)
    """
    # Threshold in [0,1] directly (no renorm, no resize)
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

        # padding
        x -= padding
        y -= padding
        w += 2 * padding
        h += 2 * padding

        # clamp to bounds
        x = max(0, x)
        y = max(0, y)
        w = min(w, W - x)
        h = min(h, H - y)

        # skip degenerate boxes
        if w > 1 and h > 1:
            boxes.append((x, y, w, h))

    boxes = remove_nested_boxes(boxes)
    boxes = suppress_overlapping_boxes(boxes, iou_threshold=0.3)
    boxes = merge_close_boxes(boxes, time_gap_thresh=20)
    return boxes


def draw_bounding_box_on_heatmap(
    spectrogram: np.ndarray,
    cam_resized: np.ndarray,
    threshold: float = 0.7,
    padding: int = 10,
) -> bytes:
    """
    Draw bboxes extracted from cam_resized ON TOP of a heatmap render.

    cam_resized must already be:
      - same shape as spectrogram [F, T]
      - normalized to [0,1]
    """
    boxes = _extract_boxes_from_cam(cam_resized, threshold, padding)

    fig, ax = plt.subplots(figsize=(10, 6))

    # heatmap
    im = ax.imshow(cam_resized, cmap="jet", alpha=0.6, aspect="auto")

    # boxes
    for x, y, w, h in boxes:
        ax.add_patch(
            plt.Rectangle(
                (x, y),
                w,
                h,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )
        )

    ax.set_xlabel("Time Frames")
    ax.set_ylabel("Mel Frequency Bins")
    ax.set_ylim(0, spectrogram.shape[0] - 2)
    plt.colorbar(im, ax=ax)
    return _save_fig_to_png_bytes(fig)


def draw_bounding_box_on_spectrogram(
    spectrogram: np.ndarray,
    cam_resized: np.ndarray,
    threshold: float = 0.7,
    padding: int = 10,
) -> bytes:
    """
    Draw bboxes extracted from cam_resized ON TOP of the spectrogram.

    cam_resized must already be:
      - same shape as spectrogram [F, T]
      - normalized to [0,1]
    """
    boxes = _extract_boxes_from_cam(cam_resized, threshold, padding)

    fig, ax = plt.subplots(figsize=(10, 6))

    # spectrogram
    im = ax.imshow(spectrogram, cmap="jet", aspect="auto")

    # boxes
    for x, y, w, h in boxes:
        ax.add_patch(
            plt.Rectangle(
                (x, y),
                w,
                h,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )
        )

    ax.set_xlabel("Time Frames")
    ax.set_ylabel("Mel Frequency Bins")
    ax.set_ylim(0, spectrogram.shape[0] - 2)
    plt.colorbar(im, ax=ax)
    return _save_fig_to_png_bytes(fig)


def generate_and_save_images(spectrogram: np.ndarray, cam_resized: np.ndarray, file_prefix: str):
    """
    spectrogram: np.ndarray [F, T]
    cam_resized: np.ndarray [F, T] normalized 0..1

    Returns:
      encoded_spec, encoded_heatmap, encoded_spec_bb, encoded_heatmap_bb
    """
    output_folder = f"visualizations/{file_prefix}_outputs"
    os.makedirs(output_folder, exist_ok=True)

    # Spectrogram
    spec_png = plot_spectrogram(spectrogram)
    encoded_spec = base64.b64encode(spec_png).decode("utf-8")

    # Heatmap only
    heat_png = save_gradcam(spectrogram, cam_resized)
    encoded_heatmap = base64.b64encode(heat_png).decode("utf-8")

    # Spectrogram + boxes
    spec_bb_png = draw_bounding_box_on_spectrogram(spectrogram, cam_resized)
    encoded_spec_bb = base64.b64encode(spec_bb_png).decode("utf-8")

    # Heatmap + boxes
    heat_bb_png = draw_bounding_box_on_heatmap(spectrogram, cam_resized)
    encoded_heatmap_bb = base64.b64encode(heat_bb_png).decode("utf-8")

    # Optional: save to disk (commented out since you mostly return base64)
    # with open(os.path.join(output_folder, f"{file_prefix}_spectrogram.png"), "wb") as f: f.write(spec_png)
    # with open(os.path.join(output_folder, f"{file_prefix}_gradcam.png"), "wb") as f: f.write(heat_png)
    # with open(os.path.join(output_folder, f"{file_prefix}_spec_with_bboxes.png"), "wb") as f: f.write(spec_bb_png)
    # with open(os.path.join(output_folder, f"{file_prefix}_gradcam_with_bboxes.png"), "wb") as f: f.write(heat_bb_png)

    return encoded_spec, encoded_heatmap, encoded_spec_bb, encoded_heatmap_bb
