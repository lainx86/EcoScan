import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

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
    # Clear previous hooks if any (though typically we just register once)
    # targeting layer4 (last residual block)
    target_layer = model.layer4
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)

def generate_gradcam(model, input_tensor, target_class):
    """
    Computes the Grad-CAM heatmap.
    """
    # Ensure gradients are enabled for this specific pass even if model is in eval mode/no_grad context globally
    # However, since we might be inside torch.no_grad() in the caller, we need to be careful.
    # Actually, Grad-CAM requires gradients. So we must enable grad.
    
    input_tensor.requires_grad = True # Ensure input tracks grad if needed, or just model parameters
    
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

def save_gradcam(heatmap, original_image_bytes, output_path):
    """
    Overlays heatmap on original image and saves to disk.
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
    
    # Save
    plt.imsave(output_path, overlayed)
    return output_path

import io # imported here for the save_gradcam function
