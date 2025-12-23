import base64
import io
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

from audio_utils import waveform_to_model_input

def save_gradcam(spectrogram, grad_cam, file_name, ax=None):
    """Overlay the Grad-CAM heatmap on the log-mel spectrogram with more distinction."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Normalize Grad-CAM image
    grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min())  # Normalize to [0, 1]
    
    # Resize the Grad-CAM to match spectrogram dimensions
    grad_cam_resized = cv2.resize(grad_cam, (spectrogram.shape[1], spectrogram.shape[0]), interpolation=cv2.INTER_CUBIC)
    
    ax.imshow(grad_cam_resized, cmap='jet', alpha=0.6)

    # Add axes labels
    ax.set_xlabel("Time Frames")
    ax.set_ylabel("Mel Frequency Bins")

    # ax.invert_yaxis()

    ax.set_ylim(0, spectrogram.shape[0]-2)  # Manually set y-axis limits (adjust as needed)

    plt.colorbar(ax.imshow(grad_cam_resized, cmap='jet', origin='lower', alpha=0.6, aspect='auto'), ax=ax)
    
    # plt.savefig(file_name, bbox_inches='tight', pad_inches=0.1)
    # plt.close()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    
    return buf.getvalue()  # raw PNG bytes


def plot_spectrogram(spectrogram, file_name, ax=None):
    """Plot the log-mel spectrogram with axes labeled."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Display the spectrogram
    ax.imshow(spectrogram, cmap='jet', aspect='equal')
    
    # Label the axes
    ax.set_xlabel("Time Frames")
    ax.set_ylabel("Mel Frequency Bins")

    ax.set_ylim(0, spectrogram.shape[0]-2)

    # Show the color bar for reference
    plt.colorbar(ax.imshow(spectrogram, cmap='jet', aspect='auto'), ax=ax)
    # plt.savefig(file_name, bbox_inches='tight', pad_inches=0.1)
    # plt.close()

    # Save figure to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    
    return buf.getvalue()  # raw PNG bytes

def overlay_gradcam_on_spectrogram(spectrogram, grad_cam, file_name, ax=None):
    """Overlay the Grad-CAM heatmap on the log-mel spectrogram with more distinction."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Normalize Grad-CAM image
    grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min())  # Normalize to [0, 1]
    
    # Resize the Grad-CAM to match spectrogram dimensions
    grad_cam_resized = cv2.resize(grad_cam, (spectrogram.shape[1], spectrogram.shape[0]), interpolation=cv2.INTER_CUBIC)
    
    # Overlay the heatmap on the spectrogram (change color map to 'inferno' or 'hot' for better contrast)
    ax.imshow(spectrogram, cmap='jet', aspect='auto')  # Spectrogram on the bottom

    ax.imshow(grad_cam_resized, cmap='jet', alpha=0.3, aspect='auto')  # Grad-CAM on top
    
    # Add axes labels
    ax.set_xlabel("Time Frames")
    ax.set_ylabel("Mel Frequency Bins")

    ax.set_ylim(0, spectrogram.shape[0]-2) 

    plt.colorbar(ax.imshow(grad_cam_resized, cmap='jet', alpha=0.6, aspect='auto'), ax=ax)
    
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0.1)
    plt.close()

# def draw_bounding_box_on_heatmap(spectrogram, grad_cam, file_name, ax=None, threshold=0.7, padding=10):
#     """Draw bounding boxes on GradCAM heatmaps with more distinction."""
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(10, 6))
    
#     # Normalize Grad-CAM image
#     grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min())  # Normalize to [0, 1]
    
#     # Resize the Grad-CAM to match spectrogram dimensions
#     grad_cam_resized = cv2.resize(grad_cam, (spectrogram.shape[1], spectrogram.shape[0]), interpolation=cv2.INTER_CUBIC)
    
#     # Threshold the Grad-CAM image to create a binary mask
#     _, thresh = cv2.threshold(grad_cam_resized, threshold, 1, cv2.THRESH_BINARY)

#     # Find contours in the thresholded image
#     contours, _ = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Draw bounding boxes around high-attention regions
#     for contour in contours:
#         # Get the bounding box coordinates for each contour
#         x, y, w, h = cv2.boundingRect(contour)
        
#         # Add padding to the bounding box
#         x -= padding  # Reduce the left side of the box
#         y -= padding  # Reduce the top side of the box
#         w += 2 * padding  # Increase the width of the box
#         h += 2 * padding  # Increase the height of the box
        
#         # Ensure the coordinates are within the image bounds
#         x = max(x, 0)
#         y = max(y, 0.5)
#         w = min(w, spectrogram.shape[1] - x)

#         h = min(h, spectrogram.shape[0] - y - 2.5)  # Ensure the height with padding does not go beyond the bottom edge
#         print(f"Bounding box before adjustment: x={x}, y={y}, w={w}, h={h}")

#         # Draw the bounding box on the spectrogram
#         ax.add_patch(plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none'))

#     ax.imshow(grad_cam_resized, cmap='jet', alpha=0.6, aspect='auto')  # Grad-CAM on top
    
#     # Add axes labels
#     ax.set_xlabel("Time Frames")
#     ax.set_ylabel("Mel Frequency Bins")

#     ax.set_ylim(0, spectrogram.shape[0]-2) 

#     # Optional: Add a color bar for Grad-CAM
#     plt.colorbar(ax.imshow(grad_cam_resized, cmap='jet', alpha=0.6, aspect='auto'), ax=ax)
    
#     plt.savefig(file_name, bbox_inches='tight', pad_inches=0.1)
#     plt.close()

def compute_iou(box1, box2):
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
    # Sort by area (largest first)
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept = []

    for box in boxes:
        overlap = False
        for k in kept:
            if compute_iou(box, k) > iou_threshold:
                overlap = True
                break

        if not overlap:
            kept.append(box)

    return kept


# --- Remove nested bounding boxes ---
def remove_nested_boxes(boxes):
    # Sort largest → smallest
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept = []

    for box in boxes:
        x, y, w, h = box
        x2, y2 = x + w, y + h

        inside = False
        for k in kept:
            kx, ky, kw, kh = k
            kx2, ky2 = kx + kw, ky + kh

            if (
                x >= kx and y >= ky and
                x2 <= kx2 and y2 <= ky2
            ):
                inside = True
                break

        if not inside:
            kept.append(box)

    return kept

def merge_close_boxes(boxes, time_gap_thresh=20):
    boxes = sorted(boxes, key=lambda b: b[0])  # sort by x (time)
    merged = []

    for box in boxes:
        if not merged:
            merged.append(box)
            continue

        x, y, w, h = box
        mx, my, mw, mh = merged[-1]

        # if close in time
        if x <= mx + mw + time_gap_thresh:
            new_x = mx
            new_y = min(my, y)
            new_w = max(mx + mw, x + w) - new_x
            new_h = max(my + mh, y + h) - new_y
            merged[-1] = (new_x, new_y, new_w, new_h)
        else:
            merged.append(box)

    return merged


def draw_bounding_box_on_heatmap(
    spectrogram,
    grad_cam,
    file_name,
    ax=None,
    threshold=0.7,
    padding=10
):
    """Draw bounding boxes on Grad-CAM heatmaps and remove nested boxes."""

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # --- Normalize Grad-CAM ---
    grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min() + 1e-8)

    # --- Resize Grad-CAM to spectrogram size ---
    grad_cam_resized = cv2.resize(
        grad_cam,
        (spectrogram.shape[1], spectrogram.shape[0]),
        interpolation=cv2.INTER_CUBIC
    )

    # --- Threshold Grad-CAM ---
    _, thresh = cv2.threshold(
        grad_cam_resized, threshold, 1, cv2.THRESH_BINARY
    )

    # --- Find contours ---
    contours, _ = cv2.findContours(
        thresh.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # --- Collect bounding boxes ---
    boxes = []
    H, W = spectrogram.shape[:2]

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Padding
        x -= padding
        y -= padding
        w += 2 * padding
        h += 2 * padding

        # Clamp to image bounds
        x = max(1, x)
        y = max(0.5, y)
        w = min(w, W - x - 1)
        h = min(h, H - y - 2.5)

        boxes.append((x, y, w, h))
    
    boxes = remove_nested_boxes(boxes)
    boxes = suppress_overlapping_boxes(boxes, iou_threshold=0.3)
    boxes = merge_close_boxes(boxes, time_gap_thresh=20)

    # --- Draw bounding boxes ---
    for x, y, w, h in boxes:
        ax.add_patch(
            plt.Rectangle(
                (x, y),
                w,
                h,
                linewidth=2,
                edgecolor="red",
                facecolor="none"
            )
        )

    # --- Overlay Grad-CAM ---
    ax.imshow(grad_cam_resized, cmap='jet', alpha=0.6, aspect='auto')

    ax.set_xlabel("Time Frames")
    ax.set_ylabel("Mel Frequency Bins")
    ax.set_ylim(0, spectrogram.shape[0] - 2)

    plt.colorbar(ax.imshow(grad_cam_resized, cmap='jet', alpha=0.6, aspect='auto'), ax=ax)

    # plt.savefig(file_name, bbox_inches="tight", pad_inches=0.1)
    # plt.close()
        # Save figure to bytes buffer

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    
    return buf.getvalue()  # raw PNG bytes


def draw_bounding_box_on_spectrogram(spectrogram, grad_cam, file_name, ax=None, threshold=0.7, padding=10):
    """Draw bounding boxes on the spectrogram based on GradCAM heatmap."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Normalize Grad-CAM image
    grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min())  # Normalize to [0, 1]
    
    # Resize the Grad-CAM to match spectrogram dimensions
    grad_cam_resized = cv2.resize(grad_cam, (spectrogram.shape[1], spectrogram.shape[0]), interpolation=cv2.INTER_CUBIC)
    
   # --- Threshold Grad-CAM ---
    _, thresh = cv2.threshold(
        grad_cam_resized, threshold, 1, cv2.THRESH_BINARY
    )

    # --- Find contours ---
    contours, _ = cv2.findContours(
        thresh.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # --- Collect bounding boxes ---
    boxes = []
    H, W = spectrogram.shape[:2]

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Padding
        x -= padding
        y -= padding
        w += 2 * padding
        h += 2 * padding

        # Clamp to image bounds
        x = max(1, x)
        y = max(0.5, y)
        w = min(w, W - x - 1)
        h = min(h, H - y - 2.5)

        boxes.append((x, y, w, h))
    
    boxes = remove_nested_boxes(boxes)
    boxes = suppress_overlapping_boxes(boxes, iou_threshold=0.3)
    boxes = merge_close_boxes(boxes, time_gap_thresh=20)

    # --- Draw bounding boxes ---
    for x, y, w, h in boxes:
        ax.add_patch(
            plt.Rectangle(
                (x, y),
                w,
                h,
                linewidth=2,
                edgecolor="red",
                facecolor="none"
            )
        )

    ax.imshow(spectrogram, cmap='jet', aspect='auto')  # Spectrogram on the bottom

    # Add axes labels
    ax.set_xlabel("Time Frames")
    ax.set_ylabel("Mel Frequency Bins")

    ax.set_ylim(0, spectrogram.shape[0]-2) 

    plt.colorbar(ax.imshow(spectrogram, cmap='jet', alpha=0.6, aspect='auto'), ax=ax)

    # Save the figure with the spectrogram and bounding boxes
    # plt.savefig(file_name, bbox_inches='tight', pad_inches=0.1)
    # plt.close()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    
    return buf.getvalue()  # raw PNG bytes


def generate_and_save_images(audio_bytes, model, grad_cam, target_layer, file_prefix):
    """Generate the spectrogram with Grad-CAM overlay and attention regions."""

    output_folder = f"visualizations/{file_prefix}_outputs"
    os.makedirs(output_folder, exist_ok=True)

    # Convert waveform to model input
    x = waveform_to_model_input(audio_bytes)
    
    # Get Grad-CAM heatmap
    cam, _ = grad_cam(x)
    grad_cam_image = cam.squeeze().cpu().numpy()
    
    # Get the log-mel spectrogram (for visualization)
    spectrogram = x.squeeze().cpu().numpy()
    print(f"Spectrogram shape: {spectrogram.shape}")  # Should be [num_mel_bins, num_time_steps]

    spectrogram_filename = os.path.join(output_folder, f"{file_prefix}_spectrogram.png")
    spec_image_bytes = plot_spectrogram( spectrogram, spectrogram_filename)
    encoded_spec = base64.b64encode(spec_image_bytes).decode('utf-8')       
    
    normal_gradcam_filename = os.path.join(output_folder, f"{file_prefix}_gradcam.png")
    heatmap_image_bytes = save_gradcam(spectrogram, grad_cam_image, normal_gradcam_filename)
    encoded_heatmap = base64.b64encode(heatmap_image_bytes).decode('utf-8')

    # Overlay Grad-CAM on the spectrogram
    gradcam_overlay_filename = os.path.join(output_folder, f"{file_prefix}_gradcam_overlay.png")
    overlay_gradcam_on_spectrogram(spectrogram, grad_cam_image, gradcam_overlay_filename)

    # Draw bounding boxes on spectrograms for high-attention regions
    spec_with_bboxes_filename = os.path.join(output_folder, f"{file_prefix}_spec_with_bboxes.png")
    spec_bb_image = draw_bounding_box_on_spectrogram(spectrogram, grad_cam_image, spec_with_bboxes_filename)
    encoded_spec_bb = base64.b64encode(spec_bb_image).decode('utf-8')

    # Draw bounding boxes on heatmaps for high-attention regions
    gradcam_with_bboxes_filename = os.path.join(output_folder, f"{file_prefix}_gradcam_with_bboxes.png")
    heatmap_bb_image = draw_bounding_box_on_heatmap(spectrogram, grad_cam_image, gradcam_with_bboxes_filename)
    encoded_heatmap_bb = base64.b64encode(heatmap_bb_image).decode('utf-8')

    return encoded_spec, encoded_heatmap, encoded_spec_bb, encoded_heatmap_bb
 
