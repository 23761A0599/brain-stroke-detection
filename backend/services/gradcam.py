import cv2
import numpy as np

import torch

from pathlib import Path
from PIL import Image
from torchvision import transforms

from model.config import (
    DEVICE,
    IMAGE_SIZE,
    CLASS_NAMES
)

# ==========================================================
# Output Directory
# ==========================================================

GRADCAM_DIR = Path("outputs") / "gradcam"
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# Image Transform
# ==========================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================================================
# Load Image
# ==========================================================

def load_image(image_path):
    """
    Load MRI image and convert it into a tensor.
    """

    image = Image.open(image_path).convert("RGB")

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(DEVICE)

    return image, image_tensor


# ==========================================================
# GradCAM Class
# ==========================================================

class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        self.forward_hook = target_layer.register_forward_hook(
            self.save_activation
        )

        self.backward_hook = target_layer.register_full_backward_hook(
            self.save_gradient
        )

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()
        # ==========================================================
# Generate Grad-CAM Heatmap
# ==========================================================

def generate_gradcam(model, image_tensor, target_layer):

    gradcam = GradCAM(model, target_layer)

    model.zero_grad()

    output = model(image_tensor)

    predicted_class = output.argmax(dim=1)

    score = output[:, predicted_class]

    score.backward()

    gradients = gradcam.gradients.detach().cpu().numpy()[0]

    activations = gradcam.activations.detach().cpu().numpy()[0]

    weights = np.mean(
        gradients,
        axis=(1, 2)
    )

    cam = np.zeros(
        activations.shape[1:],
        dtype=np.float32
    )

    for i, weight in enumerate(weights):
        cam += weight * activations[i]

    cam = np.maximum(cam, 0)

    cam = cv2.resize(
        cam,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    cam -= cam.min()

    cam /= (cam.max() + 1e-8)

    gradcam.remove_hooks()

    return cam, predicted_class.item()


# ==========================================================
# Overlay Heatmap
# ==========================================================

def overlay_heatmap(original_image, cam):

    image = np.array(
        original_image.resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        )
    )

    heatmap = np.uint8(255 * cam)

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(
        image,
        0.6,
        heatmap,
        0.4,
        0
    )

    return overlay
# ==========================================================
# Save Grad-CAM Result
# ==========================================================

def save_gradcam(model, image_path):
    """
    Generate and save Grad-CAM visualization.

    Args:
        model: Loaded PyTorch model.
        image_path: Path to the MRI image.

    Returns:
        dict containing:
            - prediction
            - gradcam_path
    """

    target_layer = model.features[-1]

    original_image, image_tensor = load_image(image_path)

    cam, predicted_class = generate_gradcam(
        model,
        image_tensor,
        target_layer
    )

    overlay = overlay_heatmap(
        original_image,
        cam
    )

    # CHANGED: per-image filename instead of a fixed name that gets
    # overwritten on every prediction
    image_stem = Path(image_path).stem
    output_path = GRADCAM_DIR / f"gradcam_{image_stem}.jpg"

    cv2.imwrite(
        str(output_path),
        cv2.cvtColor(
            overlay,
            cv2.COLOR_RGB2BGR
        )
    )

    return {
        "prediction": CLASS_NAMES[predicted_class],
        "gradcam_path": str(output_path)
    }