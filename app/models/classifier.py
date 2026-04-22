import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

CLASSES = ["Dent", "Scratch", "Crack", "Glass Shatter", "Lamp Broken", "Tire Flat"]

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def build_model():
    model = models.efficientnet_b3(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 6),
    )
    return model

def load_classifier(weights_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model()
    state = torch.load(weights_path, map_location=device)
    # Handle both raw state_dict and checkpoint dicts
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def run_classifier(model, image: Image.Image) -> dict:
    device = next(model.parameters()).device
    tensor = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.sigmoid(logits).squeeze().cpu().tolist()
    return {cls: float(p) for cls, p in zip(CLASSES, probs)}
