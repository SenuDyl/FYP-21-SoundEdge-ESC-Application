import base64
import librosa
import soundfile as sf
import os
import io
import json
import numpy as np

def load_audio_from_bytes(audio_bytes, sr=None):
    """
    Load audio from uploaded bytes.
    """
    
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=sr)
    print("[DEBUG] Audio loaded:", y.shape, "Sample rate:", sr)
    return y, sr


def extract_audio_segment(
    y,
    sr,
    start_sec,
    end_sec,
    pad_sec=0.3,
    min_duration=1.0
):
    """
    Slice waveform between start_sec and end_sec,
    add contextual padding, and enforce a minimum duration.
    """

    audio_duration = len(y) / sr

    # Step 1: Apply padding
    padded_start = start_sec - pad_sec
    padded_end = end_sec + pad_sec

    # Step 2: Enforce minimum duration
    current_duration = padded_end - padded_start

    if current_duration < min_duration:
        center = (padded_start + padded_end) / 2
        half_min = min_duration / 2

        padded_start = center - half_min
        padded_end = center + half_min

    # Step 3: Clamp to audio boundaries
    padded_start = max(0.0, padded_start)
    padded_end = min(audio_duration, padded_end)

    # Step 4: Convert to samples
    start_sample = int(padded_start * sr)
    end_sample = int(padded_end * sr)

    print(
        f"[DEBUG] Extracting samples {start_sample}:{end_sample} "
        f"(time {padded_start:.2f}s – {padded_end:.2f}s, "
        f"duration {(padded_end - padded_start):.2f}s)"
    )

    return y[start_sample:end_sample]

def merge_close_time_segments(segments, gap_threshold=0.5):
    """
    Merge time segments that are close together.

    Args:
        segments (list): list of [start_sec, end_sec] or dicts
        gap_threshold (float): max allowed gap (seconds) to merge

    Returns:
        merged_segments (list): list of (start_sec, end_sec)
    """

    # Normalize input
    normalized = []
    for seg in segments:
        if isinstance(seg, dict):
            normalized.append((seg["start_sec"], seg["end_sec"]))
        else:
            normalized.append((seg[0], seg[1]))

    # Sort by start time
    normalized.sort(key=lambda x: x[0])

    if not normalized:
        return []

    merged = []
    current_start, current_end = normalized[0]

    for next_start, next_end in normalized[1:]:
        gap = next_start - current_end

        if gap < gap_threshold:
            # Merge segments
            current_end = max(current_end, next_end)
        else:
            # Commit current segment
            merged.append((current_start, current_end))
            current_start, current_end = next_start, next_end

    # Append last segment
    merged.append((current_start, current_end))

    return merged


def save_audio_segment(
    segment_audio,
    sr,
    output_path
):
    sf.write(output_path, segment_audio, sr)
    print(f"[INFO] Saved audio segment → {output_path}")


def classify_temporal_pattern(raw_segments, merged_segments, total_audio_duration):
    """
    Classify audio based on temporal characteristics using both raw and merged segments.

    Args:
        raw_segments (list of tuples): [(start_sec, end_sec), ...]
        merged_segments (list of tuples): [(start_sec, end_sec), ...]
        total_audio_duration (float): total audio duration in seconds

    Returns:
        category (str)
    """
    if not raw_segments:
        return "Unknown"

    # Use merged_segments to check for continuous
    merged_durations = [end - start for start, end in merged_segments]
    merged_gaps = [merged_segments[i+1][0] - merged_segments[i][1] for i in range(len(merged_segments)-1)]
    merged_coverage = sum(merged_durations) / total_audio_duration

    # Continuous: most of the audio is active and gaps are small
    if merged_coverage > 0.5 and (not merged_gaps or max(merged_gaps) < 0.3):
        return "Continuous"

    # Use raw_segments to check for repetitive or bursts
    raw_durations = [end - start for start, end in raw_segments]
    raw_gaps = [raw_segments[i+1][0] - raw_segments[i][1] for i in range(len(raw_segments)-1)]
    raw_std_gap = np.std(raw_gaps) if raw_gaps else 0

    # Repetitive: many short raw segments, uniform gaps
    if len(raw_segments) > 3 and max(raw_durations) < 0.5 and raw_std_gap < 0.15:
        return "Repetitive"

    # Bursts: very few short raw segments
    if len(raw_segments) <= 3 and max(raw_durations) < 0.5:
        return "Bursts"

    # Intermittent: default if segments are irregular
    return "Intermittent"


def generate_explained_audio_segments(
    audio_bytes,
    time_json_path,
    output_root="explained_audio_segments"
):

    # Load audio
    y, sr = load_audio_from_bytes(audio_bytes)
    total_audio_duration = len(y) / sr

    # Load time insights
    with open(time_json_path, "r") as f:
        data = json.load(f)

    segments = data["time_explanation"]["important_segments"]

    merged_segments = merge_close_time_segments(
        segments,
        gap_threshold=0.5
    )

    temporal_category = classify_temporal_pattern(segments, merged_segments, total_audio_duration)

    print("[DEBUG] Merged time segments:", merged_segments)

    # Extract base name and class
    audio_file = data["audio_file"]
    base_name = os.path.splitext(os.path.basename(audio_file))[0]

    # Example: "1-sneezing" → "sneezing"
    class_name = base_name.split("-", 1)[-1]

    # Create class-specific folder
    class_output_dir = os.path.join(output_root, class_name)
    os.makedirs(class_output_dir, exist_ok=True)

    finalized_segments_info = []

    for i, seg in enumerate(merged_segments):
        start_sec = seg[0]
        end_sec = seg[1]

        segment_audio = extract_audio_segment(y, sr, start_sec, end_sec)

        # Encode segment as WAV in-memory
        out_bytes = io.BytesIO()
        sf.write(out_bytes, segment_audio, sr, format='WAV')
        out_bytes.seek(0)
        audio_b64 = base64.b64encode(out_bytes.read()).decode("utf-8")

        out_path = os.path.join(
            class_output_dir,
            f"{base_name}_segment_{i+1}_{start_sec:.2f}_{end_sec:.2f}.wav"
        )

        finalized_segments_info.append({
            "segment_index": i + 1,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "duration_sec": round(end_sec - start_sec, 2),
            "audio_base64": audio_b64 
        })

        save_audio_segment(segment_audio, sr, out_path)

        print(f"[INFO] Saved explained segment → {out_path}")

    
    json_output = {
        "audio_file": audio_file,
        "class_name": class_name,
        "temporal_category": temporal_category,
        "finalized_segments": finalized_segments_info
    }
    json_path = os.path.join(class_output_dir, f"{base_name}_segments.json")
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=4)
    print(f"[INFO] Segment info saved to JSON → {json_path}")

    return json_output



