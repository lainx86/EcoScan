import io
import torch
import torchvision.transforms as transforms
from torchvision import models
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from contextlib import asynccontextmanager
from typing import Dict, Any

# --- Configuration ---
MODEL_PATH = "coral_bleaching_resnet18.pt"
CLASSES = ["Bleached", "Healthy"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Global Model Variable ---
model = None

# --- Lifecycle Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the model on startup and clean up on shutdown.
    """
    global model
    try:
        print(f"Loading model from {MODEL_PATH} on {DEVICE}...")
        # Initialize architecture
        model = models.resnet18(weights=None)
        # Modify the final layer to match binary classification (2 classes)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 2)
        
        # Load weights
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        
        model.to(DEVICE)
        model.eval() # Set to evaluation mode
        print("Model loaded successfully.")
        yield
    except Exception as e:
        print(f"Error loading model: {e}")
        raise RuntimeError(f"Failed to load model: {e}")
    finally:
        # Clean up resources if needed
        pass

# --- FastAPI App ---
app = FastAPI(
    title="Coral Bleaching Detection API",
    description="API for classifying coral images as Bleached or Healthy using ResNet18.",
    version="1.0.0",
    lifespan=lifespan
)

# --- Preprocessing ---
def get_transform():
    """
    Returns the transformation pipeline used for training.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

# --- Endpoints ---

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify backend status.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "device": str(DEVICE)}

@app.post("/predict", response_model=Dict[str, Any])
async def predict(file: UploadFile = File(...)):
    """
    Accepts an image file and returns the predicted class and confidence.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        # Read and open image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Preprocess
        transform = get_transform()
        input_tensor = transform(image).unsqueeze(0).to(DEVICE) # Add batch dimension

        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        predicted_label = CLASSES[predicted_idx.item()]
        confidence_score = confidence.item()

        return {
            "predicted_label": predicted_label,
            "confidence": confidence_score
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
