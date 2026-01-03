import io
import torch
import torch.nn as nn
import torchaudio
import soundfile as sf

# =========================
# Config (match training)
# =========================
TARGET_SR = 44100
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 40

# Set this to what you used in training (common: 5.0 sec)
DURATION_SEC = 5.0
NUM_SAMPLES = int(TARGET_SR * DURATION_SEC)


# =========================
# Normalization (your code)
# =========================
class NormalizeMinus1To1(nn.Module):
    """Min-max normalize each example to [-1, 1]."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Works for [*, F, T] or [*, T]
        dims = list(range(x.dim()))
        # If batch exists, reduce over all but batch dim
        if x.dim() >= 2:
            reduce_dims = tuple(dims[1:])
        else:
            reduce_dims = tuple(dims)

        x_min = x.amin(dim=reduce_dims, keepdim=True)
        x_max = x.amax(dim=reduce_dims, keepdim=True)
        x = (x - x_min) / (x_max - x_min + 1e-6)
        return (x * 2.0) - 1.0


def mel_transform(
    sample_rate: int = TARGET_SR,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    n_mels: int = N_MELS,
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
    print(f"Original SR: {sr}, shape: {data.shape}")
    wav = torch.from_numpy(data).unsqueeze(0)  # [1, T]
    print(f"Waveform shape: {wav.shape}")
    # Resample to training sample rate
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)

    # Fixed duration
    wav = trim_or_pad(wav, NUM_SAMPLES)

    mel = _MEL_PIPELINE(wav)    # [1, 40, T_frames]
    x = mel.unsqueeze(0)        # [1, 1, 40, T_frames]
    return x