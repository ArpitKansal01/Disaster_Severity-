# app.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models

# ============================
# 1️⃣ App & Device
# ============================
app = FastAPI(title="Disaster + Severity Prediction API")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================
# 2️⃣ Labels
# ============================
disaster_types = ["damaged_buildings", "fallen_trees", "fire", "flood", "landslide", "non_disaster"]
severity_levels = ["no_damage", "low", "medium", "severe"]

# ============================
# 3️⃣ Model
# ============================
class MultiTaskResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = models.resnet18(pretrained=True)
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()
        self.type_head = nn.Linear(in_features, len(disaster_types))
        self.severity_head = nn.Linear(in_features, len(severity_levels))

    def forward(self, x):
        features = self.base_model(x)
        type_out = self.type_head(features)
        severity_out = self.severity_head(features)
        return type_out, severity_out

model_path = "disaster_model.pth"  # path to your saved model
model = MultiTaskResNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ============================
# 4️⃣ Transforms
# ============================
inference_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ============================
# 5️⃣ Entropy Function
# ============================
def compute_entropy(logits):
    probs = F.softmax(logits, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
    return entropy.item()


# ============================
# 6️⃣ Prediction Function (STRICT APPROVAL)
# ============================

STRICT_ENTROPY_THRESHOLD = 0.4
CONFIDENCE_THRESHOLD = 0.75

def predict_disaster_severity_strict(image: Image.Image):
    img_tensor = inference_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        type_out, sev_out = model(img_tensor)

        # entropy
        type_entropy = compute_entropy(type_out)
        sev_entropy = compute_entropy(sev_out)

        # confidence
        type_probs = F.softmax(type_out, dim=1)
        sev_probs = F.softmax(sev_out, dim=1)

        type_conf = torch.max(type_probs).item()
        sev_conf = torch.max(sev_probs).item()

        # predicted indices
        pred_type_idx = torch.argmax(type_probs, dim=1).item()
        pred_sev_idx = torch.argmax(sev_probs, dim=1).item()

        # STRICT APPROVAL CONDITION
        if (
            type_entropy <= STRICT_ENTROPY_THRESHOLD and
            sev_entropy <= STRICT_ENTROPY_THRESHOLD and
            type_conf >= CONFIDENCE_THRESHOLD and
            sev_conf >= CONFIDENCE_THRESHOLD
        ):
            status = "approved"
            pred_type = disaster_types[pred_type_idx]
            pred_sev = severity_levels[pred_sev_idx]
        else:
            status = "rejected"
            pred_type = "uncertain"
            pred_sev = "uncertain"

    return {
        "predicted_disaster": pred_type,
        "predicted_severity": pred_sev,
        "type_entropy": round(type_entropy, 4),
        "severity_entropy": round(sev_entropy, 4),
        "type_confidence": round(type_conf, 4),
        "severity_confidence": round(sev_conf, 4),
        "status": status
    }
# ============================
# 7️⃣ API Endpoint
# ============================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        result = predict_disaster_severity_strict(img)

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=400
        )

@app.get("/")
async def root():
    return {"message": "Disaster + Severity Prediction API is running!"}
