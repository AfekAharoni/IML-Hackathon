from pathlib import Path
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms
from base_model import ImageNetSubset

# ===> CONSTANTS <===
IMAGE_SIZE = 224
BATCH_SIZE = 64
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_train_transform():
    """
    Creates random in-memory image manipulations for training.

    The goal is to make the model robust to changes that should not change the
    class label, such as lighting, color tone, small crops, small shifts, and
    partial occlusions.

    Important:
        The transformed images are not saved to disk. Each image is loaded,
        changed temporarily in memory, converted to a tensor, normalized, and
        then passed to the model.

    Note:
        More advanced background replacement, where the object is segmented and
        placed on a different background, is a useful robustness idea. However,
        that requires an object segmentation/background-generation pipeline, so
        it should be implemented separately if we decide to use it.
    """
    return transforms.Compose([
        # Resize first so all following augmentations work from a stable size.
        transforms.Resize(256),

        # Randomly crop a small part of the image and resize back to 224x224.
        # This simulates the object being slightly closer/farther or shifted.
        # The crop is mild: we keep 85%-100% of the original resized image.
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.85, 1.0)),

        # Mirror flip left-right with 50% probability. This is the main
        # "mirror rotation" augmentation we want to keep.
        transforms.RandomHorizontalFlip(p=0.5),

        # Mirror flip top-bottom with low probability. This is more aggressive
        # and can be unnatural for some classes, so we keep it rare.
        transforms.RandomVerticalFlip(p=0.05),

        # Random color/lighting changes. This helps the model avoid relying too
        # much on exact brightness, contrast, saturation, or color tone.
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.45,
                contrast=0.45,
                saturation=0.45,
                hue=0.06)
        ], p=0.75),

        # 10% probability to remove color information. This encourages the
        # model to learn shape/texture cues in addition to color cues.
        transforms.RandomGrayscale(p=0.10),

        # Very small rotation only. We explicitly avoid large rotations such as
        # 90 degrees because those can create unrealistic images for this task.
        transforms.RandomRotation(degrees=5),

        # Convert PIL image to a tensor with shape [3, 224, 224].
        transforms.ToTensor(),

        # Randomly erase a small rectangle from the tensor. This simulates
        # partial occlusion or a small accidental crop without saving new files.
        transforms.RandomErasing(
            p=0.20,
            scale=(0.02, 0.08),
            ratio=(0.5, 2.0),
            value="random"),

        # Normalize with ImageNet statistics, matching the evaluator format.
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

def build_eval_transform():
    """
    Creates deterministic preprocessing for validation/test
    Validation should be stable, so this does not include random transforms
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def build_datasets(data_root=Path("dataset/train")):
    """
    Build the train dataset and validation dataset, where:
        1. Validation dataset is stable without any probabilities
        2. Train dataset should changed in each run
    """
    data_root = Path(data_root)
    train_dataset = ImageNetSubset(root=data_root, split="train", 
                                   transform=build_train_transform())
    validation_dataset = ImageNetSubset(root=data_root, split="validation",
                                         transform=build_eval_transform())
    return train_dataset, validation_dataset

def build_dataloaders(data_root=Path("dataset/train"), batch_size=BATCH_SIZE,
                      num_workers=2):
    train_dataset, validation_dataset = build_datasets(data_root)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, 
                                   num_workers=num_workers)
    return train_loader, validation_loader

def build_robust_validation_loader(
        clean_data_root=Path("dataset/train"),
        augmentations_root=Path("dataset/augmentations/validation"),
        batch_size=BATCH_SIZE,
        num_workers=2):
    """
    Build a robust validation loader from two sources:
        1. Real validation images with random in-memory manipulations.
        2. Saved validation augmentation folders, such as color_jitter and
           random_rotation, with stable evaluation preprocessing.

    This lets clean validation measure real images only, while robust
    validation measures manipulated images.
    """
    clean_data_root = Path(clean_data_root)
    augmentations_root = Path(augmentations_root)

    robust_datasets = [
        ImageNetSubset(
            root=clean_data_root,
            split="validation",
            transform=build_train_transform())
    ]

    if augmentations_root.exists():
        for augmentation_dir in sorted(p for p in augmentations_root.iterdir()
                                       if p.is_dir()):
            robust_datasets.append(
                ImageNetSubset(
                    root=augmentations_root,
                    split=augmentation_dir.name,
                    transform=build_eval_transform()))

    robust_dataset = ConcatDataset(robust_datasets)
    return DataLoader(robust_dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers)

if __name__ == "__main__":
    train_loader, validation_loader = build_dataloaders(num_workers=0)
    x, y = next(iter(train_loader))
    print("Train batch shape:", tuple(x.shape))
    print("Train labels shape:", tuple(y.shape))
    print("Validation batches:", len(validation_loader))
