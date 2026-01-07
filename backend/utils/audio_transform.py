import io
import torch
import torch.nn as nn
import torchaudio
import soundfile as sf
import json

from .config import TARGET_SR, N_MELS, HOP_LENGTH, N_FFT, DURATION_SEC, STATS_JSON_PATH

NUM_SAMPLES = int(TARGET_SR * DURATION_SEC)

class NormalizeMeanStd(nn.Module):
    def __init__(self, mean: float, std: float, eps: float = 1e-6):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + self.eps)


def mel_transform(
    sample_rate: int = TARGET_SR,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    n_mels: int = N_MELS,
) -> nn.Sequential:
    with open(STATS_JSON_PATH, "r", encoding="utf-8") as f:
        s = json.load(f)
    return nn.Sequential(
        torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        ),
        torchaudio.transforms.AmplitudeToDB(),
        NormalizeMeanStd(mean=s["mean"], std=s["std"]),
    )


_MEL_PIPELINE = mel_transform()


# =========================
# Helpers
# =========================
def trim_or_pad(waveform: torch.Tensor, num_samples: int) -> torch.Tensor:
    """
    waveform: [1, T]
    returns:  [1, num_samples]
    """
    T = waveform.shape[1]
    if T < num_samples:
        waveform = torch.nn.functional.pad(waveform, (0, num_samples - T))
    else:
        waveform = waveform[:, :num_samples]
    return waveform


def waveform_to_model_input(audio_bytes: bytes) -> torch.Tensor:
    # Load audio with soundfile (works well for .wav)
    data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)  # [T, C]
    data = data.mean(axis=1)  # mono -> [T]
    wav = torch.from_numpy(data).unsqueeze(0)  # [1, T]
    # Resample to training sample rate
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    # Fixed duration
    wav = trim_or_pad(wav, NUM_SAMPLES)

    mel = _MEL_PIPELINE(wav)    # [1, 40, T_frames]
    x = mel.unsqueeze(0)        # [1, 1, 40, T_frames]
    return x