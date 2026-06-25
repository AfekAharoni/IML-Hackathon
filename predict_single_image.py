from pathlib import Path
import sys

import torch
from PIL import Image
from torchvision import transforms

from labels import IDX_TO_NAME


PROJECT_ROOT = Path(__file__).resolve().parent
TEAM_DIR = PROJECT_ROOT / "submissions" / "my_team"
IMAGE_PATH = PROJECT_ROOT / "guitar.jpg"
WEIGHTS_PATH = TEAM_DIR / "weights.joblib"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def main():
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Weights not found: {WEIGHTS_PATH}")

    sys.path.insert(0, str(TEAM_DIR))

    # Import torch before predict/model loading to avoid WinPython DLL issues.
    from predict import Model

    image = Image.open(IMAGE_PATH).convert("RGB")
    x = build_transform()(image).unsqueeze(0)

    model = Model()
    model.load(str(WEIGHTS_PATH))

    pred = model.predict(x).item()
    class_name = IDX_TO_NAME[pred]

    print(f"Image: {IMAGE_PATH}")
    print(f"Predicted index: {pred}")
    print(f"Predicted class: {class_name}")


if __name__ == "__main__":
    main()
