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
    Creates random in-memory image manipulations for training
    Each image is loaded, randomly transformed,
    converted to a tensor, normalized, and then passed to the model
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5), # 50% probability for flip the image
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.08)
        ], p=0.7), # 70% probability for changing tone/contrast etc
        transforms.RandomGrayscale(p=0.15), #15% change to grayscale
        transforms.RandomRotation(degrees=15), # rotate between -15 to 15 degrees
        transforms.ToTensor(), 
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
