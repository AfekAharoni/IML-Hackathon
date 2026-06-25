from pathlib import Path
import argparse
import random
import shutil


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SPLITS = {
    "train": 0.60,
    "validation": 0.20,
    "test": 0.10,
    "other": 0.10,
}

SPLIT_NAMES = set(SPLITS.keys())


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def split_counts(n: int) -> dict[str, int]:
    train_count = int(n * 0.60)
    validation_count = int(n * 0.20)
    test_count = int(n * 0.10)
    other_count = n - train_count - validation_count - test_count

    return {
        "train": train_count,
        "validation": validation_count,
        "test": test_count,
        "other": other_count,
    }


def safe_move(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")

    shutil.move(str(src), str(dst))


def split_list_randomly(paths: list[Path], rng: random.Random) -> dict[str, list[Path]]:
    paths = list(paths)
    rng.shuffle(paths)

    counts = split_counts(len(paths))

    result = {}
    start = 0

    for split_name, count in counts.items():
        result[split_name] = paths[start:start + count]
        start += count

    return result


def get_class_dirs(source_dir: Path) -> list[Path]:
    return sorted(
        p for p in source_dir.iterdir()
        if p.is_dir() and p.name not in SPLIT_NAMES
    )


def split_regular_train(source_train: Path, output_root: Path, rng: random.Random) -> None:
    if not source_train.exists():
        print(f"[SKIP] Missing train source folder: {source_train}")
        return

    print()
    print(f"Splitting train images from: {source_train}")

    class_dirs = get_class_dirs(source_train)
    if not class_dirs:
        print(f"[SKIP] No unsplit class folders found in: {source_train}")
        return

    for class_dir in class_dirs:
        class_name = class_dir.name
        image_paths = sorted(p for p in class_dir.iterdir() if is_image(p))

        if not image_paths:
            print(f"[WARN] No images found in {class_dir}")
            continue

        split_images = split_list_randomly(image_paths, rng)
        counts = {name: len(items) for name, items in split_images.items()}

        for split_name, paths in split_images.items():
            for src in paths:
                dst = output_root / "train" / split_name / class_name / src.name
                safe_move(src, dst)

        print(
            f"{class_name}: {len(image_paths)} -> "
            f"train={counts['train']}, "
            f"validation={counts['validation']}, "
            f"test={counts['test']}, "
            f"other={counts['other']}"
        )


def split_augmentations(source_augmentations: Path, output_root: Path, rng: random.Random) -> None:
    if not source_augmentations.exists():
        print(f"[SKIP] Missing augmentations source folder: {source_augmentations}")
        return

    print()
    print(f"Splitting augmentation images from: {source_augmentations}")

    augmentation_dirs = get_class_dirs(source_augmentations)
    if not augmentation_dirs:
        print(f"[SKIP] No unsplit augmentation folders found in: {source_augmentations}")
        return

    for augmentation_dir in augmentation_dirs:
        augmentation_name = augmentation_dir.name

        class_dirs = get_class_dirs(augmentation_dir)

        for class_dir in class_dirs:
            class_name = class_dir.name
            image_paths = sorted(p for p in class_dir.iterdir() if is_image(p))

            if not image_paths:
                print(f"[WARN] No images found in {class_dir}")
                continue

            split_images = split_list_randomly(image_paths, rng)
            counts = {name: len(items) for name, items in split_images.items()}

            for split_name, paths in split_images.items():
                for src in paths:
                    dst = (
                        output_root
                        / "augmentations"
                        / split_name
                        / augmentation_name
                        / class_name
                        / src.name
                    )
                    safe_move(src, dst)

            print(
                f"{augmentation_name}/{class_name}: {len(image_paths)} -> "
                f"train={counts['train']}, "
                f"validation={counts['validation']}, "
                f"test={counts['test']}, "
                f"other={counts['other']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-train", default="dataset/train")
    parser.add_argument("--source-augmentations", default="dataset/augmentations")
    parser.add_argument("--output", default="dataset")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source_train = Path(args.source_train)
    source_augmentations = Path(args.source_augmentations)
    output_root = Path(args.output)

    rng = random.Random(args.seed)

    print("Split ratios:")
    print("  train: 60%")
    print("  validation: 20%")
    print("  test: 10%")
    print("  other: 10%")
    print()
    print("[MODE] Moving files randomly")
    print("[SAFE] This script does not delete existing folders")

    output_root.mkdir(parents=True, exist_ok=True)

    split_regular_train(source_train, output_root, rng)
    split_augmentations(source_augmentations, output_root, rng)

    print()
    print("[SUCCESS] Done.")


if __name__ == "__main__":
    main()
