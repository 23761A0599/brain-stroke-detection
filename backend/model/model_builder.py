import torch
import torch.nn as nn
from torchvision.models import (
    resnet50, ResNet50_Weights,
    resnet101, ResNet101_Weights,
    densenet121, DenseNet121_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
)

from model.config import NUM_CLASSES

# name -> (constructor, ImageNet weights enum)
_REGISTRY = {
    "resnet50":       (resnet50,        ResNet50_Weights.IMAGENET1K_V2),
    "resnet101":      (resnet101,       ResNet101_Weights.IMAGENET1K_V2),
    "densenet121":    (densenet121,     DenseNet121_Weights.IMAGENET1K_V1),
    "efficientnetb0": (efficientnet_b0, EfficientNet_B0_Weights.IMAGENET1K_V1),
}


def _replace_head(model: nn.Module, name: str) -> nn.Module:
    if name in ("resnet50", "resnet101"):
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    elif name == "densenet121":
        model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
    elif name == "efficientnetb0":
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    return model


def _head_params(model: nn.Module, name: str):
    if name in ("resnet50", "resnet101"):
        return model.fc.parameters()
    return model.classifier.parameters()          # densenet121, efficientnetb0


def _last_blocks(model: nn.Module, name: str):
    """Modules unfrozen for partial fine-tuning (last stage + head)."""
    if name in ("resnet50", "resnet101"):
        return [model.layer4, model.fc]
    if name == "densenet121":
        return [model.features.denseblock4, model.features.norm5, model.classifier]
    return [model.features[-3:], model.classifier]  # efficientnetb0: last 3 blocks


def build_model(
    model_name: str,
    pretrained: bool = True,
    mode: str = "frozen",          # "frozen" | "finetune" | "full"
    weights_path: str | None = None,
) -> nn.Module:
    """
    pretrained   : ImageNet weights (True) or random init (False)
    mode         : frozen   -> only the new head trains
                   finetune -> last stage + head train (use a lower LR for the backbone)
                   full     -> everything trains
    weights_path : optional local .pth state_dict for machines with no internet
    """
    name = model_name.lower()
    if name not in _REGISTRY:
        raise ValueError(f"Unsupported Model : {model_name}")

    ctor, w_enum = _REGISTRY[name]

    if pretrained and weights_path:
        model = ctor(weights=None)
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    elif pretrained:
        model = ctor(weights=w_enum)              # downloads / caches ImageNet weights
    else:
        model = ctor(weights=None)

    model = _replace_head(model, name)            # new head is always trainable

    if mode == "full":
        return model

    for p in model.parameters():
        p.requires_grad = False
    if mode == "frozen":
        for p in _head_params(model, name):
            p.requires_grad = True
    elif mode == "finetune":
        for blk in _last_blocks(model, name):
            for p in blk.parameters():
                p.requires_grad = True
    else:
        raise ValueError(f"Unsupported mode : {mode}")

    return model


def set_train_mode(model: nn.Module, mode: str = "frozen"):
    """
    Use INSTEAD of model.train() at the start of every training epoch.
    Any BatchNorm layer whose parameters are all frozen is kept in eval()
    so its running statistics are not updated by the small MRI dataset.
    """
    model.train()
    if mode != "full":
        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                if not any(p.requires_grad for p in m.parameters()):
                    m.eval()