from io import BytesIO
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from cv.model import ShelterPetResNet18, ANIMAL_TYPE_CLASSES, SIZE_CLASSES, COLOR_CLASSES

MODEL_PATH = Path(__file__).resolve().parent / "models" / "shelter_pet_resnet18_multitask.pth"

_device = torch.device("cpu")
_model = None

ANIMAL_TYPE_THRESHOLD = 80.0

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def load_model():
    global _model

    if _model is not None:
        return _model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Не знайдено файл моделі: {MODEL_PATH}")

    model = ShelterPetResNet18(pretrained=False)
    state_dict = torch.load(MODEL_PATH, map_location=_device)
    model.load_state_dict(state_dict)
    model.to(_device)
    model.eval()

    _model = model
    return _model

def predict_head(logits, classes):
    probs = torch.softmax(logits, dim=1)[0]
    predicted_index = int(torch.argmax(probs).item())
    confidence = round(float(probs[predicted_index].item()) * 100, 2)

    return {
        "label": classes[predicted_index],
        "confidence": confidence
    }

def predict_animal_type_with_threshold(logits):
    probs = torch.softmax(logits, dim=1)[0]
    predicted_index = int(torch.argmax(probs).item())
    confidence = round(float(probs[predicted_index].item()) * 100, 2)
    predicted_label = ANIMAL_TYPE_CLASSES[predicted_index]

    if confidence < ANIMAL_TYPE_THRESHOLD:
        return {
            "label": "Інше",
            "confidence": confidence,
            "allow_apply": False
        }

    return {
        "label": predicted_label,
        "confidence": confidence,
        "allow_apply": True
    }

def predict_animal_fields(file_storage):
    model = load_model()

    image_bytes = file_storage.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    x = _transform(image).unsqueeze(0).to(_device)

    with torch.no_grad():
        outputs = model(x)

    animal_type = predict_animal_type_with_threshold(outputs["animal_type"])
    size = predict_head(outputs["size"], SIZE_CLASSES)
    color = predict_head(outputs["color"], COLOR_CLASSES)

    return {
        "animal_type": animal_type,
        "size": {
            "label": size["label"],
            "confidence": size["confidence"],
            "allow_apply": True
        },
        "color": {
            "label": color["label"],
            "confidence": color["confidence"],
            "allow_apply": True
        }
    }