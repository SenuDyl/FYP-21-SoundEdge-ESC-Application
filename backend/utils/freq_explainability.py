import numpy as np
import os


def top_k_mean(array, k=3):
    """
    Compute the mean of the top-k values in a 1D array.
    """
    if len(array) == 0:
        return 0.0
    k = min(k, len(array))
    top_k_values = np.sort(array)[-k:]  # largest k values
    return top_k_values.mean()

def generate_stats(cam_resized, top_k=3, sum_weight=0.7, peak_weight=0.3):
    """
    Compute frequency band energies using weighted sum + top-k mean approach.
    
    Returns:
        energies (dict): energies per band
        freq_importance (np.ndarray): normalized importance per mel-bin
    """
    # Average over time → frequency importance
    freq_importance = cam_resized.mean(axis=1)  # shape: [mel_bins]
    freq_importance /= freq_importance.sum() + 1e-8  # normalize

    # Define mel bin ranges
    low_bins = slice(0, 13)
    mid_bins = slice(13, 26)
    high_bins = slice(26, 40)

    # Compute weighted energy
    low_energy = sum_weight * freq_importance[low_bins].sum() + peak_weight * top_k_mean(freq_importance[low_bins], k=top_k)
    mid_energy = sum_weight * freq_importance[mid_bins].sum() + peak_weight * top_k_mean(freq_importance[mid_bins], k=top_k)
    high_energy = sum_weight * freq_importance[high_bins].sum() + peak_weight * top_k_mean(freq_importance[high_bins], k=top_k)

    energies = {"low": low_energy, "mid": mid_energy, "high": high_energy}

    return energies, freq_importance


def classify_frequency_category_soft(energies):
    """
    Improved classification with soft thresholds and top-2 band logic.
    """

    low, mid, high = energies["low"], energies["mid"], energies["high"]
    bands = {"low": low, "mid": mid, "high": high}

    # Sort bands by energy descending
    sorted_bands = sorted(bands.items(), key=lambda x: x[1], reverse=True)
    top1, top2 = sorted_bands[0], sorted_bands[1]

    # Case 1: Low-mid frequencies → top bands are low & mid
    if top1[0] in ["low", "mid"] and top2[0] in ["low", "mid"]:
        category = "Low-mid frequencies"

    # Case 2: Mid-high frequencies → top bands are mid & high
    elif top1[0] in ["mid", "high"] and top2[0] in ["mid", "high"]:
        # If mid dominates strongly, we can classify as Mostly mid
        if top1[0] == "mid" and top1[1] > 0.45:
            category = "Mostly mid frequencies"
        else:
            category = "Mid-high frequencies"

    # Case 3: Mostly low/high frequencies → one band clearly dominant (>0.45)
    elif top1[0] == "low" and top1[1] > 0.45:
        category = "Mostly low frequencies"
    elif top1[0] == "high" and top1[1] > 0.45:
        category = "Mostly high frequencies"

    # Case 4: Broad frequency range → fallback
    else:
        category = "Broad frequency range"

    return category


def generate_frequency_explanation(class_name, freq_category, freq_importance, top_k=5):
    """
    Generate a human-readable frequency explanation for a sound clip.
    Combines category info with actual top-k frequency peaks for better insight.
    """
    # Identify top-k mel-bin indices
    top_bins = freq_importance.argsort()[-top_k:][::-1]  # descending order
    
    # Map bins to rough bands
    def bin_to_band(bin_idx):
        if 0 <= bin_idx <= 12:
            return "low"
        elif 13 <= bin_idx <= 25:
            return "mid"
        else:
            return "high"

    top_bands = [bin_to_band(b) for b in top_bins]
    top_bands_set = sorted(set(top_bands), key=lambda x: ["low", "mid", "high"].index(x))
    top_bands_str = " and ".join(top_bands_set)

    # Generate explanation
    if freq_category == "Broad frequency range":
        explanation = (
            f"The model attended to a broad range of frequencies, "
            f"which is typical for broadband sounds like {class_name.replace('_', ' ')}. "
            f"Top peaks are in {top_bands_str} frequencies."
        )
    elif freq_category == "Mostly low frequencies":
        explanation = (
            f"The model focused mainly on low-frequency components, "
            f"characteristic of deep, sustained sounds such as {class_name.replace('_', ' ')}."
        )
    elif freq_category == "Mostly mid frequencies":
        explanation = (
            f"Mid-frequency components were most influential, "
            f"capturing the dominant spectral content typical of {class_name.replace('_', ' ')} sounds."
        )
    elif freq_category == "Mostly high frequencies":
        explanation = (
            f"High-frequency components were most influential, "
            f"reflecting the sharp and high-pitched nature of {class_name.replace('_', ' ')} sounds."
        )
    elif freq_category == "Low-mid frequencies":
        explanation = (
            f"The model emphasized low-to-mid frequency regions, "
            f"capturing the dominant spectral content typical of {class_name.replace('_', ' ')} sounds."
        )
    elif freq_category == "Mid-high frequencies":
        explanation = (
            f"The model emphasized mid-to-high frequency regions, "
            f"matching the spectral profile of {class_name.replace('_', ' ')} sounds."
        )
    else:
        explanation = "Frequency profile could not be determined."

    return explanation


def save_frequency_feedback(audio_file_name, explanation, output_root="frequency_feedback"):
    # create output directory if it doesn't exist
    os.makedirs(output_root, exist_ok=True)
    # Extract file name without extension
    base_name = os.path.splitext(os.path.basename(audio_file_name))[0]

    # Save explanation to a text file
    file_path = os.path.join(output_root, f"{base_name}_freq_feedback.txt")
    with open(file_path, "w") as f:
        f.write(explanation)

    print(f"[INFO] Frequency feedback saved to: {file_path}")

def generate_and_save_frequency_explanation(class_name, audio_file_name, cam_resized):
    """
    cam_resized: np.ndarray [F, T] normalized 0..1
    """
    energies, freq_importance = generate_stats(cam_resized)
    freq_category = classify_frequency_category_soft(energies)

    explanation = generate_frequency_explanation(
        class_name, freq_category, freq_importance
    )
    save_frequency_feedback(audio_file_name, explanation)
    return explanation
