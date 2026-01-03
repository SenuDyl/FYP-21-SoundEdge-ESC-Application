from dataclasses import dataclass
import numpy as np
import torch

@dataclass(frozen=True)
class ExplainOutput:
    spectrogram: np.ndarray      # [F, T]
    cam_resized: np.ndarray      # [F, T] 0..1
    probs: torch.Tensor          # [C]
    idx: int
    conf: float
