import base64
import librosa
import soundfile as sf
import os
import io
import json
import numpy as np


def load_audio_from_bytes(audio_bytes, sr=None):
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=sr)
    return y, sr


def extract_audio_segment(y, sr, start_sec, end_sec, pad_sec=0.3, min_duration=1.0):
    audio_duration = len(y) / sr

    padded_start = start_sec - pad_sec
    padded_end = end_sec + pad_sec

    current_duration = padded_end - padded_start
    if current_duration < min_duration:
        center = (padded_start + padded_end) / 2
        half_min = min_duration / 2
        padded_start = center - half_min
        padded_end = center + half_min

    padded_start = max(0.0, padded_start)
    padded_end = min(audio_duration, padded_end)

    start_sample = int(padded_start * sr)
    end_sample = int(padded_end * sr)

    return y[start_sample:end_sample]


def merge_close_time_segments(segments, gap_threshold=0.5):
    normalized = []
    for seg in segments:
        if isinstance(seg, dict):
            normalized.append((seg["start_sec"], seg["end_sec"]))
        else:
            normalized.append((seg[0], seg[1]))

    normalized.sort(key=lambda x: x[0])
    if not normalized:
        return []

    merged = []
    current_start, current_end = normalized[0]

    for next_start, next_end in normalized[1:]:
        gap = next_start - current_end
        if gap < gap_threshold:
            current_end = max(current_end, next_end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = next_start, next_end

    merged.append((current_start, current_end))
    return merged


def save_audio_segment(segment_audio, sr, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, segment_audio, sr)


def classify_temporal_pattern(raw_segments, merged_segments, total_audio_duration):
    if not raw_segments:
        return "Unknown"

    merged_durations = [end - start for start, end in merged_segments]
    merged_gaps = [merged_segments[i+1][0] - merged_segments[i][1] for i in range(len(merged_segments)-1)]
    merged_coverage = sum(merged_durations) / total_audio_duration

    if merged_coverage > 0.5 and (not merged_gaps or max(merged_gaps) < 0.3):
        return "Continuous"

    raw_durations = [end - start for start, end in raw_segments]
    raw_gaps = [raw_segments[i+1][0] - raw_segments[i][1] for i in range(len(raw_segments)-1)]
    raw_std_gap = np.std(raw_gaps) if raw_gaps else 0

    if len(raw_segments) > 3 and max(raw_durations) < 0.5 and raw_std_gap < 0.15:
        return "Repetitive"

    if len(raw_segments) <= 3 and max(raw_durations) < 0.5:
        return "Bursts"

    return "Intermittent"


def generate_explained_audio_segments(audio_bytes, time_json_path, output_dir):
    y, sr = load_audio_from_bytes(audio_bytes)
    total_audio_duration = len(y) / sr

    with open(time_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data["time_explanation"]["important_segments"]
    merged_segments = merge_close_time_segments(segments, gap_threshold=0.5)
    temporal_category = classify_temporal_pattern(segments, merged_segments, total_audio_duration)

    audio_file = data["audio_file"]
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    class_name = base_name.split("-", 1)[-1]

    segments_dir = os.path.join(output_dir, "time", "segments")
    os.makedirs(segments_dir, exist_ok=True)

    finalized_segments_info = []

    for i, (start_sec, end_sec) in enumerate(merged_segments):
        segment_audio = extract_audio_segment(y, sr, start_sec, end_sec)

        out_bytes = io.BytesIO()
        sf.write(out_bytes, segment_audio, sr, format="WAV")
        out_bytes.seek(0)
        audio_b64 = base64.b64encode(out_bytes.read()).decode("utf-8")

        wav_path = os.path.join(
            segments_dir,
            f"{base_name}_segment_{i+1}_{start_sec:.2f}_{end_sec:.2f}.wav"
        )
        save_audio_segment(segment_audio, sr, wav_path)

        finalized_segments_info.append({
            "segment_index": i + 1,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "duration_sec": round(end_sec - start_sec, 2),
            "audio_base64": audio_b64,
            "wav_path": wav_path,
        })

    json_output = {
        "audio_file": audio_file,
        "class_name": class_name,
        "temporal_category": temporal_category,
        "finalized_segments": finalized_segments_info
    }

    segments_json_path = os.path.join(segments_dir, f"{base_name}_segments.json")
    with open(segments_json_path, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=4)

    return json_output, segments_json_path
