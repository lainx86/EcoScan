import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io

import os

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "coral_bleaching_resnet18.pt")
CLASSES = ["Bleached", "Healthy"]

def load_network(device):
    """
    Loads the ResNet18 model architecture and weights.
    """
    print(f"Loading model on {device}...")
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    # Binary classification
    model.fc = torch.nn.Linear(num_ftrs, 2)
    
    # Load state dict
    try:
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        raise RuntimeError(f"Model file not found at {MODEL_PATH}")
        
    model.to(device)
    model.eval()
    return model

def get_transform():
    """
    Returns the preprocessing pipeline.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def preprocess_image(image_bytes):
    """
    Converts raw image bytes to a tensor.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transform = get_transform()
    return transform(image).unsqueeze(0) # Add batch dimension

def predict(model, input_tensor, device):
    """
    Runs inference and returns label and confidence.
    """
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        
    label = CLASSES[predicted_idx.item()]
    conf_score = confidence.item()
    
    return label, conf_score, predicted_idx.item()
