import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import io

# --- Globals for Hooks ---
activations = None
gradients = None

def forward_hook(module, input, output):
    global activations
    activations = output

def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]

def register_hooks(model):
    """
    Registers hooks on layer4 of ResNet18.
    """
    target_layer = model.layer4
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)

def generate_gradcam(model, input_tensor, target_class):
    """
    Computes the Grad-CAM heatmap.
    """
    input_tensor.requires_grad = True # Ensure gradients
    
    # Forward pass
    output = model(input_tensor)
    
    # Zero grads
    model.zero_grad()
    
    # Target score
    score = output[0, target_class]
    score.backward()
    
    # gradients: [1, 512, 7, 7]
    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
    
    # activations: [1, 512, 7, 7]
    cam = torch.sum(weights * activations, dim=1, keepdim=True)
    cam = F.relu(cam)
    
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-7)
    
    return cam.squeeze().detach().cpu().numpy()

def apply_heatmap(heatmap, original_image_bytes):
    """
    Overlays heatmap on original image and returns a PIL Image.
    """
    # Load original image
    img = Image.open(io.BytesIO(original_image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_np = np.array(img) / 255.0
    
    # Resize heatmap
    heatmap_img = Image.fromarray(heatmap)
    heatmap_img = heatmap_img.resize((224, 224), resample=Image.BILINEAR)
    heatmap_np = np.array(heatmap_img)
    
    # Colormap
    colormap = plt.get_cmap("jet")
    heatmap_colored = colormap(heatmap_np)
    heatmap_colored = heatmap_colored[:, :, :3] # RGB only
    
    # Overlay
    alpha = 0.4
    overlayed = (1 - alpha) * img_np + alpha * heatmap_colored
    overlayed = np.clip(overlayed, 0, 1)
    
    # Convert back to uint8 PIL Image
    result_img = Image.fromarray((overlayed * 255).astype(np.uint8))
    return result_img
