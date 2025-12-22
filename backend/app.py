from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
import json
import io

from audio_utils import waveform_to_model_input
from model_components.model import CNN_PCAw_SSRPMS_KAN  # rename to your actual class

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

model.to(DEVICE)
model.eval()

@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        print(f"file name: {file.filename}")
        # get the frequency of the audio
        

        x = waveform_to_model_input(audio_bytes).to(DEVICE)
        print(f"shape: {x.shape}")
        print(f"min: {x.min().item()}, max: {x.max().item()}, mean: {x.mean().item()}, std: {x.std().item()}")
        with torch.no_grad():
            logits = model(x)
            print(f"logits shape: {logits.shape}, logits: {logits}")
            probs = torch.softmax(logits, dim=1)[0]
            print(f"probs: {probs}")
            idx = int(torch.argmax(probs).item())
            conf = float(probs[idx].item())

        return {"label": LABELS[idx], "confidence": conf, "class_index": idx}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))