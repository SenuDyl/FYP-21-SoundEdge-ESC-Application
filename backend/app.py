from __future__ import annotations

import json
from pathlib import Path

import torch
from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from utils.config import HOP_LENGTH, TARGET_SR, LABELS_JSON_PATH, RESULTS_DIR, CHECKPOINTS_DIR, SAMPLES_DIR
from utils.explainability_shared import compute_explainability_tensors
from utils.freq_explainability import generate_and_save_frequency_explanation
from utils.time_explainability import generate_and_save_time_explanation
from utils.visualization import generate_and_save_images

from utils.api_helpers import (
    AppConfig,
    ModelStore,
    resolve_sample,
    safe_name,
    make_run_dir,
)

# ---------------------------CONFIG---------------------------

CFG = AppConfig(
    labels_path=Path(LABELS_JSON_PATH),
    results_root=Path(RESULTS_DIR),
    checkpoints_dir=Path(CHECKPOINTS_DIR),
    samples_dir=Path(SAMPLES_DIR),
    cors_origins=("http://localhost:5173",),
)

CFG.results_root.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with CFG.labels_path.open("r", encoding="utf-8") as f:
    LABELS = json.load(f)

MODEL_STORE = ModelStore(device=DEVICE, num_classes=len(LABELS))

# ---------------------------APP---------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(CFG.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------ENDPOINTS---------------------------

@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}


@app.get("/samples")
def list_samples():
    if not CFG.samples_dir.exists():
        return {"samples": []}

    samples = sorted(
        [
            p.name
            for p in CFG.samples_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".wav"
        ],
        key=lambda s: s.lower(),
    )
    return {"samples": samples}


@app.get("/samples/{sample_name}")
def get_sample_audio(sample_name: str):
    p = resolve_sample(CFG, sample_name)
    return FileResponse(path=str(p), media_type="audio/*", filename=p.name)


@app.get("/models")
def list_models():
    if not CFG.checkpoints_dir.exists():
        return {"models": []}

    models = sorted(
        [p.stem for p in CFG.checkpoints_dir.glob("*.pth") if p.is_file()],
        key=lambda s: s.lower(),
    )
    return {"models": models}


@app.post("/predict")
async def predict(sample_name: str = Form(...), model_name: str = Form(...)):
    try:
        sample_path = resolve_sample(CFG, sample_name)
        audio_bytes = sample_path.read_bytes()
        safe_file = safe_name(sample_path.name)

        _, grad_cam = MODEL_STORE.get(CFG, model_name)

        run_dir = make_run_dir(CFG, safe_file)

        explain_output = compute_explainability_tensors(audio_bytes, grad_cam)

        spec_encoded, heatmap_encoded, spec_bb_encoded, heatmap_bb_encoded, saved_pngs = (
            generate_and_save_images(
                spectrogram=explain_output.spectrogram,
                cam_resized=explain_output.cam_resized,
                file_prefix=Path(safe_file).stem,
                output_dir=str(run_dir),
            )
        )

        freq_explanation, freq_txt_path = generate_and_save_frequency_explanation(
            class_name=LABELS[explain_output.idx],
            audio_file_name=safe_file,
            cam_resized=explain_output.cam_resized,
            output_dir=str(run_dir),
        )

        _, json_output, time_json_path, segments_json_path = generate_and_save_time_explanation(
            audio_bytes=audio_bytes,
            audio_file_name=safe_file,
            cam_resized=explain_output.cam_resized,
            hop_length=HOP_LENGTH,
            sample_rate=TARGET_SR,
            output_dir=str(run_dir),
        )

        return {
            "label": LABELS[explain_output.idx],
            "confidence": explain_output.conf,
            "spectrogram": f"data:image/png;base64,{spec_encoded}",
            "gradcam_heatmap": f"data:image/png;base64,{heatmap_encoded}",
            "spectrogram_bboxes": f"data:image/png;base64,{spec_bb_encoded}",
            "gradcam_heatmap_bboxes": f"data:image/png;base64,{heatmap_bb_encoded}",
            "frequency_explanation": freq_explanation,
            "time_explanation": json_output,
            "run_dir": str(run_dir),
            "used_sample": sample_name,
            "used_model": model_name,
            "saved_files": {
                "pngs": saved_pngs,
                "frequency_txt": freq_txt_path,
                "time_insight_json": time_json_path,
                "segments_json": segments_json_path,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[DEBUG] Exception in /predict: {e}")
        raise HTTPException(status_code=400, detail=str(e))
