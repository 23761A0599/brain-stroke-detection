import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms

from model.config import DEVICE, IMAGE_SIZE, CLASS_NAMES

GRADCAM_DIR = Path("outputs") / "gradcam"
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.forward_hook = target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


def generate_gradcam_map(model, image_tensor):
    """
    Computes standard Grad-CAM activation map from EfficientNet last features block.
    """
    model.eval()
    
    # Target last conv layer of EfficientNet
    target_layer = model.features[-1]
    gradcam = GradCAM(model, target_layer)

    model.zero_grad()
    output = model(image_tensor)
    predicted_class = output.argmax(dim=1)
    
    score = output[0, predicted_class]
    score.backward()

    gradients = gradcam.gradients.detach().cpu().numpy()[0]
    activations = gradcam.activations.detach().cpu().numpy()[0]
    gradcam.remove_hooks()

    weights = np.mean(gradients, axis=(1, 2))
    cam = np.zeros(activations.shape[1:], dtype=np.float32)

    for i, weight in enumerate(weights):
        cam += weight * activations[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))
    
    # Normalize safely
    cam_max = cam.max()
    if cam_max > 0:
        cam = cam / cam_max

    return cam, predicted_class.item()