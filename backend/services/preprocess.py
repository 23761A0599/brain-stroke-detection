from PIL import Image
from torchvision import transforms


def get_transform():
    """
    Image preprocessing pipeline for EfficientNet-B0.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def preprocess_image(image_path):
    """
    Loads and preprocesses an MRI image.
    Returns a tensor of shape (1, 3, 224, 224).
    """
    image = Image.open(image_path).convert("RGB")

    transform = get_transform()

    image = transform(image)

    image = image.unsqueeze(0)

    return image