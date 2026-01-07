# ---------------------------app.py---------------------------
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import torch
import json
import os
import re
from datetime import datetime
from pathlib import Path

from utils.explainability_shared import compute_explainability_tensors
from utils.visualization import generate_and_save_images
from utils.freq_explainability import generate_and_save_frequency_explanation
from utils.time_explainability import generate_and_save_time_explanation
from utils.config import TARGET_SR, HOP_LENGTH
from model.CNN_PSK import CNN_PCAw_SSRPMS_KAN
from model.GradCAM import GradCAM

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with open("./class_labels/ESC10.json", "r") as f:
    LABELS = json.load(f)

RESULTS_ROOT = "results"
CHECKPOINTS_DIR = Path("./available_models/clean")

os.makedirs(RESULTS_ROOT, exist_ok=True)

MODEL_CACHE = {}  # model_name -> (model, grad_cam)

def _resolve_checkpoint(model_name: str) -> Path:
    if not model_name:
        raise HTTPException(status_code=400, detail="Invalid model name.")
    model_name += ".pth"
    ckpt = (CHECKPOINTS_DIR / model_name).resolve()
    base = CHECKPOINTS_DIR.resolve()

    # prevent path traversal
    if base not in ckpt.parents or not ckpt.is_file():
        raise HTTPException(status_code=404, detail="Model checkpoint not found.")

    return ckpt

def get_model_and_cam(model_name: str):
    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name]

    ckpt_path = _resolve_checkpoint(model_name)

    m = CNN_PCAw_SSRPMS_KAN(num_classes=len(LABELS))
    state = torch.load(str(ckpt_path), map_location=DEVICE)
    m.load_state_dict(state)
    m.to(DEVICE)
    m.eval()

    target_layer = m.conv3[0]
    cam = GradCAM(m, target_layer)

    MODEL_CACHE[model_name] = (m, cam)
    return m, cam


def _safe_name(name: str) -> str:
    # keep letters, numbers, dash, underscore, dot
    name = os.path.basename(name)
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_")
    return name or "audio"

@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}

@app.get("/models")
def list_models():
    """
    Returns available model checkpoint filenames in ./model/checkpoints
    """
    if not CHECKPOINTS_DIR.exists():
        return {"models": []}
    models = sorted(
        [p.stem for p in CHECKPOINTS_DIR.glob("*.pth") if p.is_file()],
        key=lambda s: s.lower()
    )
    return {"models": models}


@app.post("/predict")
async def predict(file: UploadFile = File(...), model_name: str = Form(...)):
    try:
        audio_bytes = await file.read()
        safe_file = _safe_name(file.filename)
        model, grad_cam = get_model_and_cam(model_name)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = os.path.join(RESULTS_ROOT, f"{timestamp}_{os.path.splitext(safe_file)[0]}")
        os.makedirs(run_dir, exist_ok=True)

        explain_output = compute_explainability_tensors(audio_bytes, grad_cam)

        spec_encoded, heatmap_encoded, spec_bb_encoded, heatmap_bb_encoded, saved_pngs = generate_and_save_images(
            spectrogram=explain_output.spectrogram,
            cam_resized=explain_output.cam_resized,
            file_prefix=os.path.splitext(safe_file)[0],
            output_dir=run_dir,
        )

        freq_explanation, freq_txt_path = generate_and_save_frequency_explanation(
            class_name=LABELS[explain_output.idx],
            audio_file_name=safe_file,
            cam_resized=explain_output.cam_resized,
            output_dir=run_dir,
        )

        time_segments, json_output, time_json_path, segments_json_path = generate_and_save_time_explanation(
            audio_bytes=audio_bytes,
            audio_file_name=safe_file,
            cam_resized=explain_output.cam_resized,
            hop_length=HOP_LENGTH,
            sample_rate=TARGET_SR,
            output_dir=run_dir,
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
            "run_dir": run_dir,
            "saved_files": {
                "pngs": saved_pngs,
                "frequency_txt": freq_txt_path,
                "time_insight_json": time_json_path,
                "segments_json": segments_json_path,
            }
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
