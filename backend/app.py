from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import matplotlib.pyplot as plt
import torch
import json
import io
import os

from audio_utils import waveform_to_model_input
from model_components.model import CNN_PCAw_SSRPMS_KAN  # rename to your actual class
from grad_cam import GradCAM
from visualization_utils import generate_and_save_images
from freq_explainability_utils import generate_and_save_frequency_explanation
from time_explainability_utils import generate_and_save_time_explanation

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
with open("labels.json", "r") as f:
    LABELS = json.load(f)

# Load model
model = CNN_PCAw_SSRPMS_KAN(num_classes=len(LABELS))
model.load_state_dict(torch.load("checkpoints/best_2025-12-22.pth", map_location=DEVICE))

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
        # Read the audio file
        audio_bytes = await file.read()
        print(f"file name: {file.filename}")
        
        # Convert waveform to model input
        x = waveform_to_model_input(audio_bytes).to(DEVICE)
        print(f"shape: {x.shape}")
        print(f"min: {x.min().item()}, max: {x.max().item()}, mean: {x.mean().item()}, std: {x.std().item()}")

        # Get model's predictions
        with torch.no_grad():
            logits = model(x)
            print(f"logits shape: {logits.shape}, logits: {logits}")
            probs = torch.softmax(logits, dim=1)[0]
            print(f"probs: {probs}")
            idx = int(torch.argmax(probs).item())
            conf = float(probs[idx].item())

        file_name = f"{file.filename}"

        spec_encoded, heatmap_encoded, spec_bb_encoded, heatmap_bb_encoded = generate_and_save_images(audio_bytes, model, grad_cam, target_layer, file_name)

        freq_explanation = generate_and_save_frequency_explanation(grad_cam, LABELS[idx], file_name, audio_bytes)

        time_segments, json_output = generate_and_save_time_explanation(
            grad_cam, 
            file_name, 
            hop_length=256, 
            sample_rate=44100, 
            audio_bytes=audio_bytes, 
            threshold=0.6)

        return {
            "label": LABELS[idx], 
            "confidence": conf, 
            "class_index": idx,
            "spectrogram": f"data:image/png;base64,{spec_encoded}", 
            "gradcam_heatmap": f"data:image/png;base64,{heatmap_encoded}",
            "spectrogram_bboxes": f"data:image/png;base64,{spec_bb_encoded}",
            "gradcam_heatmap_bboxes": f"data:image/png;base64,{heatmap_bb_encoded}",
            "frequency_explanation": freq_explanation,
            "time_explanation": json_output
            }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))