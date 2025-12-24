import os
import uuid
import torch
from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager

import model
import gradcam

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STATIC_DIR = "static"
OUTPUT_DIR = os.path.join(STATIC_DIR, "outputs")

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Global Model ---
net = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global net
    # Load model
    try:
        net = model.load_network(DEVICE)
        # Register hooks for Grad-CAM
        gradcam.register_hooks(net)
        print("Model loaded and hooks registered.")
    except Exception as e:
        print(f"Failed to load model: {e}")
    yield
    # Cleanup
    net = None

app = FastAPI(lifespan=lifespan)

# --- Mounts & Templates ---
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory="templates")

# --- Endpoints ---

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    if net is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.content_type.startswith("image/"):
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": "File must be an image."
        })

    try:
        content = await file.read()
        
        # 1. Preprocess
        input_tensor = model.preprocess_image(content)
        
        # 2. Predict (Enable grad for Grad-CAM preparation if needed, though usually just forward is enough, 
        # but for Grad-CAM backward pass we need to handle it carefully).
        # We will do a standard forward pass first to get the class.
        # Note: gradcam implementation requires a forward pass where gradients are tracked? 
        # Actually, if we are in inference mode (torch.no_grad), .backward() will fail.
        # So we must switch context.
        
        # We need to run locally with gradients enabled for Grad-CAM
        with torch.enable_grad():
             # Forward pass for prediction & capturing activations
             input_tensor = input_tensor.to(DEVICE)
             input_tensor.requires_grad = True # Important for backprop to input if we wanted it, but essential for cam sometimes
             
             # Re-run forward to ensure hooks capture this specific batch
             outputs = net(input_tensor)
             probabilities = torch.nn.functional.softmax(outputs, dim=1)
             confidence, predicted_idx = torch.max(probabilities, 1)
             
             label = model.CLASSES[predicted_idx.item()]
             conf_score = confidence.item()
             
             # 3. Generate Grad-CAM
             # We use the prediction index as the target class
             heatmap = gradcam.generate_gradcam(net, input_tensor, predicted_idx.item())
             
        # 4. Save Result
        filename = f"{uuid.uuid4()}.jpg"
        output_path = os.path.join(OUTPUT_DIR, filename)
        gradcam.save_gradcam(heatmap, content, output_path)
        
        # Return result
        # Web path is relative to static mount
        web_path = f"/static/outputs/{filename}"
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "label": label,
            "confidence": f"{conf_score:.2%}",
            "gradcam_image": web_path,
            "image_uploaded": True
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": f"Error processing image: {str(e)}"
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
