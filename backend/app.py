# ---------------------------app.py---------------------------
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import json
import os
import re
from datetime import datetime

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

model = CNN_PCAw_SSRPMS_KAN(num_classes=len(LABELS))
model.load_state_dict(torch.load("./model/checkpoints/best_2025-12-22.pth", map_location=DEVICE))

target_layer = model.conv3[0]
grad_cam = GradCAM(model, target_layer)

model.to(DEVICE)
model.eval()

RESULTS_ROOT = "results"
os.makedirs(RESULTS_ROOT, exist_ok=True)

def _safe_name(name: str) -> str:
    # keep letters, numbers, dash, underscore, dot
    name = os.path.basename(name)
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_")
    return name or "audio"

@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        safe_file = _safe_name(file.filename)

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
