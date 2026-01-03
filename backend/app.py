from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import json
import os

from utils.explainability_shared import compute_explainability_tensors
from model.CNN_PSK import CNN_PCAw_SSRPMS_KAN
from model.GradCAM import GradCAM
from utils.visualization import generate_and_save_images
from utils.freq_explainability import generate_and_save_frequency_explanation
from utils.time_explainability import generate_and_save_time_explanation

app = FastAPI()

# Allow your frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load labels
with open("./class_labels/ESC10.json", "r") as f:
    LABELS = json.load(f)

# Load model
model = CNN_PCAw_SSRPMS_KAN(num_classes=len(LABELS))
model.load_state_dict(torch.load("./model/checkpoints/best_2025-12-22.pth", map_location=DEVICE))

# Create GradCAM object
target_layer = model.conv3[0]
grad_cam = GradCAM(model, target_layer)

model.to(DEVICE)
model.eval()

# Folder to save heatmaps
HEATMAPS_DIR = "heatmaps/"
os.makedirs(HEATMAPS_DIR, exist_ok=True)

VIS_RESULTS = "visualizations/"
os.makedirs(VIS_RESULTS, exist_ok=True)

@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        file_name = f"{file.filename}"

        # Compute ONCE
        spectrogram, cam_resized, probs, idx, conf = compute_explainability_tensors(audio_bytes, grad_cam)

        # Reuse everywhere
        spec_encoded, heatmap_encoded, spec_bb_encoded, heatmap_bb_encoded = generate_and_save_images(
            spectrogram=spectrogram,
            cam_resized=cam_resized,
            file_prefix=file_name
        )

        freq_explanation = generate_and_save_frequency_explanation(
            class_name=LABELS[idx],
            audio_file_name=file_name,
            cam_resized=cam_resized
        )

        time_segments, json_output = generate_and_save_time_explanation(
            audio_bytes=audio_bytes,
            audio_file_name=file_name,
            cam_resized=cam_resized,
            hop_length=256,
            sample_rate=44100,
            threshold=0.6
        )

        return {
            "label": LABELS[idx],
            "confidence": conf,
            "spectrogram": f"data:image/png;base64,{spec_encoded}",
            "gradcam_heatmap": f"data:image/png;base64,{heatmap_encoded}",
            "spectrogram_bboxes": f"data:image/png;base64,{spec_bb_encoded}",
            "gradcam_heatmap_bboxes": f"data:image/png;base64,{heatmap_bb_encoded}",
            "frequency_explanation": freq_explanation,
            "time_explanation": json_output
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
