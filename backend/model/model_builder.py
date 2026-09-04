import torch.nn as nn

from torchvision.models import (
    resnet50,
    resnet101,
    densenet121,
    efficientnet_b0,
    ResNet50_Weights,
    ResNet101_Weights,
    DenseNet121_Weights,
    EfficientNet_B0_Weights
)

from model.config import NUM_CLASSES


def build_model(model_name):

    model_name = model_name.lower()

    if model_name == "resnet50":

        model = resnet50(weights=ResNet50_Weights.DEFAULT)

        model.fc = nn.Linear(
            model.fc.in_features,
            NUM_CLASSES
        )

    elif model_name == "resnet101":

        model = resnet101(weights=ResNet101_Weights.DEFAULT)

        model.fc = nn.Linear(
            model.fc.in_features,
            NUM_CLASSES
        )

    elif model_name == "densenet121":

        model = densenet121(weights=DenseNet121_Weights.DEFAULT)

        model.classifier = nn.Linear(
            model.classifier.in_features,
            NUM_CLASSES
        )

    elif model_name == "efficientnetb0":

        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            NUM_CLASSES
        )

    else:

        raise ValueError(
            f"Unsupported Model : {model_name}"
        )

    return model