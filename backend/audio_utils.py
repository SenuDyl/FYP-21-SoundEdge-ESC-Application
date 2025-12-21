import torch
import torch.nn as nn
import torchaudio

torchaudio.set_audio_backend("sox_io")

# Load audio
def load_audio(file_path: str, sr: int = 44100) -> torch.Tensor:
    waveform, orig_sr = torchaudio.load(file_path)
    if orig_sr != sr:
        waveform = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=sr)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform  # [1, time]

# Trim or pad waveform to fixed length
def trim_or_pad_waveform(waveform, num_samples):
    current_length = waveform.shape[1]
    if current_length < num_samples:
        pad_size = num_samples - current_length
        waveform = torch.nn.functional.pad(waveform, (0, pad_size))
    else:
        waveform = waveform[:, :num_samples]
    return waveform

# Mel-spectrogram transformation with normalization
class NormalizeMinus1To1(nn.Module):
    """Min-max normalize each example to [-1, 1]."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Works for [*, F, T] or [*, T]
        dims = list(range(x.dim()))
        # compute per-example min/max (keep channel/freq/time jointly)
        # If batch exists, reduce over all but batch dim
        if x.dim() >= 2:
            reduce_dims = tuple(dims[1:])
        else:
            reduce_dims = tuple(dims)

        x_min = x.amin(dim=reduce_dims, keepdim=True)
        x_max = x.amax(dim=reduce_dims, keepdim=True)
        x = (x - x_min) / (x_max - x_min + 1e-6)
        return (x * 2.0) - 1.0

# Mel-spectrogram transformation pipeline
def mel_transform(
    sample_rate: int = 44100,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 40,
) -> nn.Sequential:
    return nn.Sequential(
        torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        ),
        torchaudio.transforms.AmplitudeToDB(),
        NormalizeMinus1To1(),
    )

# Full preprocessing for a single audio file
def preprocess_audio(file_path: str, sr: int = 44100, duration: float = 5.0) -> torch.Tensor:
    waveform = load_audio(file_path, sr)
    waveform = trim_or_pad_waveform(waveform, int(sr * duration))
    mel_module = mel_transform()
    log_mel = mel_module(waveform)  # [channel, n_mels, time]
    log_mel = log_mel.unsqueeze(0)  # Add batch dim: [1, channel, n_mels, time]
    return log_mel