import cv2
import numpy as np
import torch

from .audio_transform import waveform_to_model_input
from .types import ExplainOutput

def compute_explainability_tensors(audio_bytes, grad_cam):
    device = next(grad_cam.model.parameters()).device
    x = waveform_to_model_input(audio_bytes).to(device)

    cam, logits = grad_cam(x)
    probs = torch.softmax(logits, dim=1)[0]
    idx = int(torch.argmax(probs).item())
    conf = float(probs[idx].item())

    spectrogram = x.squeeze().detach().cpu().numpy()      # [F, T]
    cam_np = cam.squeeze().detach().cpu().numpy()         # [H, W]

    cam_np = (cam_np - cam_np.min()) / (cam_np.max() + 1e-8)
    cam_resized = cv2.resize(
        cam_np,
        (spectrogram.shape[1], spectrogram.shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )

    return ExplainOutput(
        spectrogram=spectrogram,
        cam_resized=cam_resized,
        probs=probs,
        idx=idx,
        conf=conf,
    )
