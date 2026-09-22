import io
import json

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

backend = Flask(__name__)
CORS(backend)  
MODEL_PATH = r"current_model.pth"
CLASS_NAMES_PATH = r"class_names.json"
NUM_CLASSES = 262
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)  


model = models.resnet50(weights=None)
model.fc = nn.Sequential(
    nn.Linear(2048, 1024),
    nn.BatchNorm1d(1024),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),

    nn.Linear(1024, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),

    nn.Linear(512, NUM_CLASSES)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


@backend.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    image = Image.open(io.BytesIO(file.read())).convert("RGB")

    input_tensor = transform(image).unsqueeze(0).to(device)  # add batch dimension

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        top5_probs, top5_idx = torch.topk(probs, 5)

    results = [
        {"label": class_names[idx.item()], "confidence": round(prob.item() * 100, 2)}
        for prob, idx in zip(top5_probs[0], top5_idx[0])
    ]

    return jsonify({"predictions": results})


if __name__ == "__main__":
    backend.run(host="0.0.0.0", port=5000, debug=True)