import torch

from model.model_loader import load_model
from model.config import CLASS_NAMES, DEVICE
from services.preprocess import preprocess_image

# Load the model only once when this module is imported
model = load_model("efficientnetb0")
model.eval()


def predict_image(image_path):
    """
    Predict the class of an MRI image.

    Args:
        image_path (str): Path to the uploaded MRI image.

    Returns:
        dict: Prediction results.
    """

    image = preprocess_image(image_path).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probabilities, dim=1)

    prediction = CLASS_NAMES[predicted.item()]

    confidence = round(confidence.item() * 100, 2)

    class_probabilities = {
        CLASS_NAMES[i]: round(probabilities[0][i].item() * 100, 2)
        for i in range(len(CLASS_NAMES))
    }

    return {
        "prediction": prediction,
        "confidence": confidence,
        "class_probabilities": class_probabilities
    }
def get_model():
    """
    Return the loaded model instance.
    """
    return model