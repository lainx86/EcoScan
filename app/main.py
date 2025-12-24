import os
import torch
import base64
import io
from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager

from app.ml import model
from app.ml import gradcam

# --- Configuration ---
DEVICE = torch.device("cpu") # Force CPU for HF Spaces (unless GPU space selected)
# Note: "cuda" checks are fine, but robust deployment usually just defaults to what's available safely.
# If the space has GPU, torch.cuda.is_available() works. If not, CPU.
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")

# --- Global Model ---
net = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global net
    try:
        net = model.load_network(DEVICE)
        gradcam.register_hooks(net)
        print(f"Model loaded on {DEVICE}")
    except Exception as e:
        print(f"Failed to load model: {e}")
    yield
    net = None

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ... (Previous imports remain same, just ensuring this goes to the top is tricky with replace_file_content if I don't select the whole block. I'll stick to a calculated replace or just use multi_replace carefully)

app = FastAPI(lifespan=lifespan)

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (including feby.dev)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="app/templates")

# --- Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/predict")
async def predict_api(file: UploadFile = File(...)):
    """
    JSON Endpoint for external frontends (e.g. feby.dev)
    """
    if net is None:
        return JSONResponse(content={"error": "Model not loaded"}, status_code=503)
    
    if not file.content_type.startswith("image/"):
        return JSONResponse(content={"error": "Invalid file type"}, status_code=400)

    try:
        content = await file.read()
        
        # 1. Preprocess
        input_tensor = model.preprocess_image(content)
        
        # 2. Predict & Grad-CAM
        with torch.enable_grad():
             input_tensor = input_tensor.to(DEVICE)
             input_tensor.requires_grad = True
             
             outputs = net(input_tensor)
             probabilities = torch.nn.functional.softmax(outputs, dim=1)
             confidence, predicted_idx = torch.max(probabilities, 1)
             
             label = model.CLASSES[predicted_idx.item()]
             conf_score = confidence.item()
             
             heatmap = gradcam.generate_gradcam(net, input_tensor, predicted_idx.item())
             
        # 3. Process Grad-CAM Image to Base64
        result_pil = gradcam.apply_heatmap(heatmap, content)
        
        buffered = io.BytesIO()
        result_pil.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        img_base64 = f"data:image/jpeg;base64,{img_str}"
        
        return {
            "label": label,
            "confidence": conf_score,
            "gradcam_image": img_base64
        }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    if net is None:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": "Model is not loaded. Please contact administrator."
        })
    
    if not file.content_type.startswith("image/"):
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": "File must be an image."
        })

    try:
        content = await file.read()
        
        # 1. Preprocess
        input_tensor = model.preprocess_image(content)
        
        # 2. Predict & Grad-CAM
        # Enable grad for Grad-CAM computation
        with torch.enable_grad():
             input_tensor = input_tensor.to(DEVICE)
             input_tensor.requires_grad = True # Track gradients for input if strictly needed, mainly for model weights
             
             # Forward
             outputs = net(input_tensor)
             probabilities = torch.nn.functional.softmax(outputs, dim=1)
             confidence, predicted_idx = torch.max(probabilities, 1)
             
             label = model.CLASSES[predicted_idx.item()]
             conf_score = confidence.item()
             
             # Grad-CAM
             heatmap = gradcam.generate_gradcam(net, input_tensor, predicted_idx.item())
             
        # 3. Process Grad-CAM Image to Base64
        result_pil = gradcam.apply_heatmap(heatmap, content)
        
        # Save to buffer
        buffered = io.BytesIO()
        result_pil.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        img_base64 = f"data:image/jpeg;base64,{img_str}"
        
        # Return result with embedded image (stateless)
        return templates.TemplateResponse("index.html", {
            "request": request,
            "label": label,
            "confidence": f"{conf_score:.2%}",
            "gradcam_image": img_base64, # Base64 string
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
    # Hugging Face Spaces expects port 7860
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
