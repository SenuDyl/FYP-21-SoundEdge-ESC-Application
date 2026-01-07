import os
import json

from .time_segment import generate_explained_audio_segments


def compute_time_importance(cam_resized):
    time_importance = cam_resized.mean(axis=0)
    time_importance = time_importance / (time_importance.max() + 1e-8)
    return time_importance


def extract_time_segments(time_importance, threshold=0.6, min_frames=5):
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

    if start is not None and len(active) - start >= min_frames:
        segments.append((start, len(active) - 1))

    return segments


def frames_to_seconds(segments, hop_length, sample_rate):
    time_segments = []
    for start_f, end_f in segments:
        start_t = start_f * hop_length / sample_rate
        end_t = end_f * hop_length / sample_rate
        time_segments.append((round(start_t, 2), round(end_t, 2)))
    return time_segments


def save_time_insight_json(audio_file_name, time_segments, threshold, output_dir):
    time_dir = os.path.join(output_dir, "time")
    os.makedirs(time_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(audio_file_name))[0]
    json_path = os.path.join(time_dir, f"{base_name}_time_insight.json")

    data = {
        "audio_file": audio_file_name,
        "time_explanation": {
            "threshold": threshold,
            "important_segments": time_segments
        }
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    return json_path


def generate_and_save_time_explanation(
    audio_bytes,
    audio_file_name,
    cam_resized,
    hop_length,
    sample_rate,
    output_dir,
    threshold=0.6
):
    time_importance = compute_time_importance(cam_resized)
    frame_segments = extract_time_segments(time_importance, threshold=threshold)
    time_segments = frames_to_seconds(frame_segments, hop_length, sample_rate)

    time_json_path = save_time_insight_json(audio_file_name, time_segments, threshold, output_dir)

    json_output, segments_json_path = generate_explained_audio_segments(
        audio_bytes=audio_bytes,
        time_json_path=time_json_path,
        output_dir=output_dir,
    )

    return time_segments, json_output, time_json_path, segments_json_path
