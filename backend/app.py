import os
import torch
import torchvision.models as models

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Ensure this matches the exact folder structure in your git repository
MODEL_PATH = os.path.join(BASE_DIR, "models", "EfficientNetB0", "best_efficientnetb0.pth")

def get_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

    try:
        # 1. Attempt loading as a state dictionary with weights-only safe loading disabled
        state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"), weights_only=False)
        
        # If the file was saved as a full model object directly
        if isinstance(state_dict, torch.nn.Module):
            model = state_dict
        else:
            # Instantiate architecture and load state_dict
            model = models.efficientnet_b0(weights=None)
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = torch.nn.Linear(num_ftrs, 2)
            model.load_state_dict(state_dict)

        model.eval()
        return model
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise e

model = get_model()