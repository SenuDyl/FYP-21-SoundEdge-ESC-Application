import numpy as np
import os
import cv2
import json

from visualization_utils import waveform_to_model_input
from time_segment_utils import generate_explained_audio_segments

def compute_time_importance(cam_resized):
    """
    cam_resized: np.ndarray of shape [mel_bins, time_frames]
    """
    # Average over frequency → importance per time frame
    time_importance = cam_resized.mean(axis=0)  # [time_frames]

    # Normalize
    time_importance = time_importance / (time_importance.max() + 1e-8)
    
    print("Time importance stats - min:", time_importance.min(), "max:", time_importance.max(), "mean:", time_importance.mean())

    return time_importance

def extract_time_segments(
    time_importance,
    threshold=0.6,
    min_frames=5
):
    """
    Returns list of (start_frame, end_frame)
    """
    active = time_importance >= threshold
    segments = []

    start = None
    for i, is_active in enumerate(active):
        if is_active and start is None:
            start = i
        elif not is_active and start is not None:
            if i - start >= min_frames:
                segments.append((start, i - 1))
            start = None

    # Handle trailing segment
    if start is not None and len(active) - start >= min_frames:
        segments.append((start, len(active) - 1))

    return segments

def frames_to_seconds(segments, hop_length, sample_rate):
    """
    Convert frame indices to time in seconds
    """
    time_segments = []

    for start_f, end_f in segments:
        start_t = start_f * hop_length / sample_rate
        end_t = end_f * hop_length / sample_rate
        time_segments.append((round(start_t, 2), round(end_t, 2)))

    # time_segments.append({
    #         "start_sec": round(start_t, 2),
    #         "end_sec": round(end_t, 2),
    #         "duration_sec": round(end_t - start_t, 2)
    #     })

    print("[DEBUG] Time segments (sec):", time_segments)

    return time_segments

def save_time_insight_json(
    audio_file_name,
    time_segments,
    threshold,
    output_root="time_insights"
):
    os.makedirs(output_root, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(audio_file_name))[0]
    json_path = os.path.join(output_root, f"{base_name}_time_insight.json")

    data = {
        "audio_file": audio_file_name,
        "time_explanation": {
            "threshold": threshold,
            "important_segments": time_segments
        }
    }

    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"[INFO] Time insight saved → {json_path}")
    return json_path
    

def generate_and_save_time_explanation(
    grad_cam,
    audio_file_name,
    hop_length,
    sample_rate,
    audio_bytes,
    threshold=0.6
):
    x = waveform_to_model_input(audio_bytes)
    spectrogram = x.squeeze().cpu().numpy()
    
    # Get Grad-CAM heatmap
    cam, _ = grad_cam(x)
    cam_np = cam.squeeze(0).squeeze(0).cpu().numpy()

    cam_np = (cam_np - cam_np.min()) / (cam_np.max() + 1e-8)

    cam_resized = cv2.resize(cam_np, (spectrogram.shape[1], spectrogram.shape[0]), interpolation=cv2.INTER_CUBIC)

    print("Heatmap shape before resizing:", cam_np.shape)
    print("Heatmap shape after resizing:", cam_resized.shape)

    time_importance = compute_time_importance(cam_resized)

    frame_segments = extract_time_segments(
        time_importance,
        threshold=threshold
    )

    time_segments = frames_to_seconds(
        frame_segments,
        hop_length,
        sample_rate
    )

    json_path = save_time_insight_json(
        audio_file_name,
        time_segments,
        threshold
    )

    json_output = generate_explained_audio_segments(
        audio_bytes=audio_bytes,
        time_json_path=json_path
    )


    return time_segments, json_output
