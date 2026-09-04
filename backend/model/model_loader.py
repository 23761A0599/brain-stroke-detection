import torch
from pathlib import Path

from model.config import DEVICE, MODELS_DIR
from model.model_builder import build_model


def load_model(model_name):

    """
    Loads the best trained model.

    Supported Models:
        - resnet50
        - resnet101
        - densenet121
        - efficientnetb0
    """

    model = build_model(model_name)

    model_path = (
    MODELS_DIR /
    "EfficientNetB0" /
    f"best_{model_name}.pth"
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"Model not found:\n{model_path}"
        )

    model.load_state_dict(

        torch.load(
            model_path,
            map_location=DEVICE
        )

    )

    model.to(DEVICE)

    model.eval()

    return model