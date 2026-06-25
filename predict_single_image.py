from pathlib import Path
import sys

import torch
from PIL import Image
from torchvision import transforms

from labels import IDX_TO_NAME


PROJECT_ROOT = Path(__file__).resolve().parent
TEAM_DIR = PROJECT_ROOT / "submissions" / "my_team"
IMAGES_DIR = PROJECT_ROOT / "tricky_images"
WEIGHTS_PATH = TEAM_DIR / "weights.joblib"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

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
    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"Images folder not found: {IMAGES_DIR}")

    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Weights not found: {WEIGHTS_PATH}")

    image_paths = sorted(
        path for path in IMAGES_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        print(f"No images found in: {IMAGES_DIR}")
        return

    sys.path.insert(0, str(TEAM_DIR))

    # Import torch before predict/model loading to avoid WinPython DLL issues.
    from predict import Model

    model = Model()
    model.load(str(WEIGHTS_PATH))

    transform = build_transform()

    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        x = transform(image).unsqueeze(0)

        pred = model.predict(x).item()
        class_name = IDX_TO_NAME[pred]

        print(f"Image: {image_path.name}")
        print(f"Predicted index: {pred}")
        print(f"Predicted class: {class_name}")
        print()


if __name__ == "__main__":
    main()
