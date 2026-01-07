# utils/api_helpers.py
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Any

import torch
from fastapi import HTTPException

from model.CNN_PSK import CNN_PCAw_SSRPMS_KAN
from model.GradCAM import GradCAM


SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9._-]+")


@dataclass(frozen=True)
class AppConfig:
    labels_path: Path
    results_root: Path
    checkpoints_dir: Path
    samples_dir: Path
    cors_origins: Tuple[str, ...]


def safe_name(name: str) -> str:
    name = os.path.basename(name)
    name = SAFE_NAME_RE.sub("_", name).strip("_")
    return name or "audio"


def ensure_within_base(base: Path, target: Path, not_found_msg: str) -> Path:
    base = base.resolve()
    target = target.resolve()

    if base not in target.parents or not target.is_file():
        raise HTTPException(status_code=404, detail=not_found_msg)

    return target


def resolve_sample(cfg: AppConfig, sample_name: str) -> Path:
    if not sample_name:
        raise HTTPException(status_code=400, detail="Invalid sample name.")

    p = ensure_within_base(
        cfg.samples_dir,
        cfg.samples_dir / sample_name,
        not_found_msg="Sample not found.",
    )

    if p.suffix.lower() != ".wav":
        raise HTTPException(status_code=400, detail="Unsupported audio format.")

    return p


def resolve_checkpoint(cfg: AppConfig, model_name: str) -> Path:
    if not model_name:
        raise HTTPException(status_code=400, detail="Invalid model name.")

    ckpt_name = f"{model_name}.pth"
    p = ensure_within_base(
        cfg.checkpoints_dir,
        cfg.checkpoints_dir / ckpt_name,
        not_found_msg="Model checkpoint not found.",
    )
    return p


def make_run_dir(cfg: AppConfig, sample_file_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    stem = Path(sample_file_name).stem
    run_dir = cfg.results_root / f"{timestamp}_{stem}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


class ModelStore:
    """
    Simple in-memory cache:
      model_name -> (model, grad_cam)
    """

    def __init__(self, device: str, num_classes: int):
        self.device = device
        self.num_classes = num_classes
        self._cache: Dict[str, Tuple[CNN_PCAw_SSRPMS_KAN, GradCAM]] = {}

    def get(self, cfg: AppConfig, model_name: str) -> Tuple[CNN_PCAw_SSRPMS_KAN, GradCAM]:
        if model_name in self._cache:
            return self._cache[model_name]

        ckpt_path = resolve_checkpoint(cfg, model_name)

        m = CNN_PCAw_SSRPMS_KAN(num_classes=self.num_classes)
        state: Any = torch.load(str(ckpt_path), map_location=self.device)
        m.load_state_dict(state)
        m.to(self.device)
        m.eval()

        target_layer = m.conv3[0]
        cam = GradCAM(m, target_layer)

        self._cache[model_name] = (m, cam)
        return m, cam
