# explainability_shared.py
import torch
import numpy as np
import cv2

from .audio_transform import waveform_to_model_input

def compute_explainability_tensors(audio_bytes, grad_cam):
    """
    Runs waveform->mel and GradCAM exactly once.

    Returns:
      spectrogram: np.ndarray [mel_bins, time_frames]
      cam_resized: np.ndarray [mel_bins, time_frames] (0..1)
      probs: torch.Tensor [num_classes]
      idx: int (predicted class)
      conf: float
    """
    # 1) waveform -> model input
    x = waveform_to_model_input(audio_bytes)  # expected shape your model wants

    # 2) GradCAM forward+backward once
    cam, logits = grad_cam(x)

    probs = torch.softmax(logits, dim=1)[0]
    idx = int(torch.argmax(probs).item())
    conf = float(probs[idx].item())

    # 3) Convert to numpy
    spectrogram = x.squeeze().detach().cpu().numpy()          # [F, T]
    cam_np = cam.squeeze().detach().cpu().numpy()             # [H, W] (layer space)

    # 4) Normalize and resize CAM to spectrogram resolution
    cam_np = (cam_np - cam_np.min()) / (cam_np.max() + 1e-8)
    cam_resized = cv2.resize(
        cam_np,
        (spectrogram.shape[1], spectrogram.shape[0]),
        interpolation=cv2.INTER_CUBIC
    )

    return spectrogram, cam_resized, probs, idx, conf
