import numpy as np
import os


def top_k_mean(array, k=3):
    if len(array) == 0:
        return 0.0
    k = min(k, len(array))
    top_k_values = np.sort(array)[-k:]
    return top_k_values.mean()


def generate_stats(cam_resized, top_k=3, sum_weight=0.7, peak_weight=0.3):
    freq_importance = cam_resized.mean(axis=1)
    freq_importance /= freq_importance.sum() + 1e-8

    low_bins = slice(0, 13)
    mid_bins = slice(13, 26)
    high_bins = slice(26, 40)

    low_energy = sum_weight * freq_importance[low_bins].sum() + peak_weight * top_k_mean(freq_importance[low_bins], k=top_k)
    mid_energy = sum_weight * freq_importance[mid_bins].sum() + peak_weight * top_k_mean(freq_importance[mid_bins], k=top_k)
    high_energy = sum_weight * freq_importance[high_bins].sum() + peak_weight * top_k_mean(freq_importance[high_bins], k=top_k)

    energies = {"low": low_energy, "mid": mid_energy, "high": high_energy}
    return energies, freq_importance


def classify_frequency_category_soft(energies):
    low, mid, high = energies["low"], energies["mid"], energies["high"]
    bands = {"low": low, "mid": mid, "high": high}

    sorted_bands = sorted(bands.items(), key=lambda x: x[1], reverse=True)
    top1, top2 = sorted_bands[0], sorted_bands[1]

    if top1[0] in ["low", "mid"] and top2[0] in ["low", "mid"]:
        category = "Low-mid frequencies"
    elif top1[0] in ["mid", "high"] and top2[0] in ["mid", "high"]:
        if top1[0] == "mid" and top1[1] > 0.45:
            category = "Mostly mid frequencies"
        else:
            category = "Mid-high frequencies"
    elif top1[0] == "low" and top1[1] > 0.45:
        category = "Mostly low frequencies"
    elif top1[0] == "high" and top1[1] > 0.45:
        category = "Mostly high frequencies"
    else:
        category = "Broad frequency range"

    return category


def generate_frequency_explanation(class_name, freq_category, freq_importance, top_k=5):
    top_bins = freq_importance.argsort()[-top_k:][::-1]

    def bin_to_band(bin_idx):
        if 0 <= bin_idx <= 12:
            return "low"
        if 13 <= bin_idx <= 25:
            return "mid"
        return "high"

    top_bands = [bin_to_band(b) for b in top_bins]
    top_bands_set = sorted(set(top_bands), key=lambda x: ["low", "mid", "high"].index(x))
    top_bands_str = " and ".join(top_bands_set)

    clean_name = class_name.replace("_", " ")

    if freq_category == "Broad frequency range":
        return (
            f"The model attended to a broad range of frequencies, which is typical for broadband sounds like {clean_name}. "
            f"Top peaks are in {top_bands_str} frequencies."
        )
    if freq_category == "Mostly low frequencies":
        return f"The model focused mainly on low-frequency components, characteristic of deep, sustained sounds such as {clean_name}."
    if freq_category == "Mostly mid frequencies":
        return f"Mid-frequency components were most influential, capturing the dominant spectral content typical of {clean_name} sounds."
    if freq_category == "Mostly high frequencies":
        return f"High-frequency components were most influential, reflecting the sharp and high-pitched nature of {clean_name} sounds."
    if freq_category == "Low-mid frequencies":
        return f"The model emphasized low-to-mid frequency regions, capturing the dominant spectral content typical of {clean_name} sounds."
    if freq_category == "Mid-high frequencies":
        return f"The model emphasized mid-to-high frequency regions, matching the spectral profile of {clean_name} sounds."
    return "Frequency profile could not be determined."


def save_frequency_feedback(audio_file_name, explanation, output_dir):
    freq_dir = os.path.join(output_dir, "frequency")
    os.makedirs(freq_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(audio_file_name))[0]
    file_path = os.path.join(freq_dir, f"{base_name}_freq_feedback.txt")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(explanation)

    return file_path


def generate_and_save_frequency_explanation(class_name, audio_file_name, cam_resized, output_dir):
    energies, freq_importance = generate_stats(cam_resized)
    freq_category = classify_frequency_category_soft(energies)

    explanation = generate_frequency_explanation(class_name, freq_category, freq_importance)
    txt_path = save_frequency_feedback(audio_file_name, explanation, output_dir)

    return explanation, txt_path
