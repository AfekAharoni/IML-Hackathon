"""
Expected submissions layout:
  submissions/
    team_a/
      train.py
      model.py
      predict.py
      weights.joblib

Run:
  python evaluate.py
"""
import importlib.util
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms

from labels import (
    HF_INDEX_TO_NAME,
    HF_INDEX_TO_IDX,
    TARGET_HF_INDICES,
)


DATA_ROOT = Path("dataset")
SUBMISSIONS_DIR = Path("submissions")
BATCH_SIZE = 64
WEIGHTS_FILENAME = "weights.joblib"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageNetSubset(Dataset):
    """Loads the 20 target classes from a split folder."""

    def __init__(self, root: Path, split: str = "validation", transform=None):
        self.transform = transform
        self.samples = []

        split_root = root / split

        if not split_root.exists():
            raise FileNotFoundError(
                f"Folder not found: {split_root}\n"
                f"Expected structure: {root}/{split}/<class_name>/*.jpg"
            )

        for hf_idx in sorted(TARGET_HF_INDICES):
            class_name = HF_INDEX_TO_NAME[hf_idx]
            class_dir = split_root / class_name

            if not class_dir.exists():
                raise FileNotFoundError(f"Class folder not found: {class_dir}")

            local_idx = HF_INDEX_TO_IDX[hf_idx]

            image_paths = []
            image_paths.extend(class_dir.glob("*.jpg"))
            image_paths.extend(class_dir.glob("*.jpeg"))
            image_paths.extend(class_dir.glob("*.JPEG"))
            image_paths.extend(class_dir.glob("*.png"))

            for img_path in sorted(image_paths):
                self.samples.append((img_path, local_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def build_eval_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_clean_split(split: str):
    dataset = ImageNetSubset(
        DATA_ROOT / "train",
        split=split,
        transform=build_eval_transform(),
    )
    print(f"Loaded {len(dataset)} clean {split} images.")
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)


def load_augmentation_split(split: str):
    augmentations_root = DATA_ROOT / "augmentations" / split

    if not augmentations_root.exists():
        raise FileNotFoundError(
            f"Augmentation {split} folder not found: {augmentations_root}"
        )

    datasets = []
    for augmentation_dir in sorted(p for p in augmentations_root.iterdir() if p.is_dir()):
        dataset = ImageNetSubset(
            augmentations_root,
            split=augmentation_dir.name,
            transform=build_eval_transform(),
        )
        datasets.append(dataset)

    if not datasets:
        raise RuntimeError(
            f"No augmentation {split} folders found in {augmentations_root}"
        )

    combined = ConcatDataset(datasets)
    print(f"Loaded {len(combined)} augmentation {split} images.")
    return DataLoader(combined, batch_size=BATCH_SIZE, shuffle=False)


def load_submission(team_dir: Path):
    predict_path = team_dir / "predict.py"
    model_path = team_dir / "model.py"
    weights_path = team_dir / WEIGHTS_FILENAME

    if not predict_path.exists():
        raise FileNotFoundError(f"Missing predict.py in {team_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model.py in {team_dir}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing {WEIGHTS_FILENAME} in {team_dir}")

    sys.path.insert(0, str(team_dir))
    sys.modules.pop("model", None)

    try:
        spec = importlib.util.spec_from_file_location(
            f"{team_dir.name}_predict",
            predict_path,
        )

        if spec is None or spec.loader is None:
            raise ImportError(f"Could not import predict.py from {team_dir}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, "Model"):
            raise AttributeError(
                f"predict.py in {team_dir} must define a class named Model"
            )

        model = module.Model()
        model.load(str(weights_path))

    finally:
        sys.path.pop(0)
        sys.modules.pop("model", None)

    return model


@torch.no_grad()
def evaluate(model, loader):
    correct = 0
    total = 0

    for x, y in loader:
        preds = model.predict(x)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return correct / total


def main():
    splits = ["validation", "test"]

    loaders = {}
    for split in splits:
        print(f"Preparing clean {split} set...")
        loaders[("clean", split)] = load_clean_split(split)

        print(f"Preparing augmentation {split} set...")
        loaders[("augmentations", split)] = load_augmentation_split(split)
        print()

    print()

    team_dirs = sorted(d for d in SUBMISSIONS_DIR.iterdir() if d.is_dir())
    if not team_dirs:
        print(f"No submissions found in {SUBMISSIONS_DIR}/")
        sys.exit(1)

    results = []
    for team_dir in team_dirs:
        print(f"Evaluating {team_dir.name}...", end=" ", flush=True)
        try:
            model = load_submission(team_dir)
            scores = {}
            for dataset_name in ["clean", "augmentations"]:
                for split in splits:
                    scores[(dataset_name, split)] = evaluate(
                        model,
                        loaders[(dataset_name, split)],
                    )

            combined_acc = sum(scores.values()) / len(scores)
            results.append((team_dir.name, scores, combined_acc))

            print(
                f"clean_val: {scores[('clean', 'validation')]:.4f} | "
                f"aug_val: {scores[('augmentations', 'validation')]:.4f} | "
                f"clean_test: {scores[('clean', 'test')]:.4f} | "
                f"aug_test: {scores[('augmentations', 'test')]:.4f} | "
                f"combined: {combined_acc:.4f}"
            )
        except Exception as e:
            print(f"FAILED - {e}")
            results.append((team_dir.name, None, None))

    print("\n--- Leaderboard ---")
    ranked = sorted(
        (r for r in results if r[2] is not None),
        key=lambda r: r[2],
        reverse=True,
    )

    for rank, (team, scores, combined_acc) in enumerate(
            ranked,
            start=1):
        print(
            f"  {rank}. {team:<20} "
            f"clean_val={scores[('clean', 'validation')]:.4f} "
            f"aug_val={scores[('augmentations', 'validation')]:.4f} "
            f"clean_test={scores[('clean', 'test')]:.4f} "
            f"aug_test={scores[('augmentations', 'test')]:.4f} "
            f"combined={combined_acc:.4f}"
        )

    for team, scores, combined_acc in results:
        if combined_acc is None:
            print(f"  --  {team:<20} FAILED")


if __name__ == "__main__":
    main()
