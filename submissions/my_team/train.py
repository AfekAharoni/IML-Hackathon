from pathlib import Path

import joblib
import torch
import torch.nn as nn
import torch.optim as optim  # Added for Adam optimizer
from torch.utils.data import DataLoader, ConcatDataset  # Added ConcatDataset to combine training data
from torchvision import transforms
from tqdm import tqdm  # Added for a nice progress bar during training

from base_model import ImageNetSubset
from model import ModelArchitecture

DATA_ROOT = Path("../../dataset")
OUTPUT = Path("weights.joblib")

IMAGE_SIZE = 128  # Set to 128x128 pixels to match our lightweight CNN architecture
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10  # Number of full passes over the training dataset


def main():
    """
    Full training pipeline.

    This script must create weights.joblib.
    """
    # Determine the computing device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # TODO: load dataset (you might want to use ImageNetSubset)
    # Define transformations for standardizing the input images
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Construct the BASE paths for clean data
    clean_base_dir = DATA_ROOT / "train"

    # Initialize clean datasets
    clean_train_dataset = ImageNetSubset(root=clean_base_dir, split="train", transform=transform)
    clean_val_dataset = ImageNetSubset(root=clean_base_dir, split="validation", transform=transform)

    # --- THE FIX FOR AUGMENTATIONS ---
    # Set the base paths to the folder JUST BEFORE the augmentation types
    aug_train_base = DATA_ROOT / "augmentations" / "train"
    aug_val_base = DATA_ROOT / "augmentations" / "validation"

    # Load each augmentation type separately by passing its folder name as the 'split' parameter
    aug_train_color = ImageNetSubset(root=aug_train_base, split="color_jitter", transform=transform)
    aug_train_rot = ImageNetSubset(root=aug_train_base, split="random_rotation", transform=transform)

    # Assuming validation has the same internal folder structure
    aug_val_color = ImageNetSubset(root=aug_val_base, split="color_jitter", transform=transform)
    aug_val_rot = ImageNetSubset(root=aug_val_base, split="random_rotation", transform=transform)

    # Combine the clean training data with ALL the augmented training sets
    full_train_dataset = ConcatDataset([clean_train_dataset, aug_train_color, aug_train_rot])

    # Combine the augmented validation sets to create one robust testing set
    aug_val_dataset = ConcatDataset([aug_val_color, aug_val_rot])

    # Create DataLoaders
    train_loader = DataLoader(full_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    clean_val_loader = DataLoader(clean_val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    aug_val_loader = DataLoader(aug_val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # TODO: create your model
    # Instantiate the model and move it to the computed device
    model = ModelArchitecture().to(device)

    # Define Loss function (CrossEntropyLoss) and Optimizer (Adam)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Begin the training loop
    print("Start training")
    for epoch in range(EPOCHS):
        model.train()  # Set the model to training mode
        running_loss = 0.0

        # Iterate over the training batches with a progress bar
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            # Move data to the corresponding device
            images, labels = images.to(device), labels.to(device)

            # Forward pass: compute predicted outputs
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize weights
            optimizer.zero_grad()  # Clear previous gradients
            loss.backward()  # Compute new gradients
            optimizer.step()  # Update weights using Adam

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # Validation step
        model.eval()  # Set the model to evaluation mode

        # --- 1. Evaluate Clean Validation ---
        clean_val_loss = 0.0
        clean_correct = 0
        clean_total = 0

        # Disable gradient calculation for validation to save memory and compute
        with torch.no_grad():
            for images, labels in clean_val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                loss = criterion(outputs, labels)
                clean_val_loss += loss.item()

                # Get the predicted class (the index with the highest logit)
                _, predicted = torch.max(outputs.data, 1)
                clean_total += labels.size(0)
                clean_correct += (predicted == labels).sum().item()

        avg_clean_val_loss = clean_val_loss / len(clean_val_loader)
        clean_val_accuracy = 100 * clean_correct / clean_total

        # --- 2. Evaluate Augmented (Robust) Validation ---
        aug_val_loss = 0.0
        aug_correct = 0
        aug_total = 0

        with torch.no_grad():
            for images, labels in aug_val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                loss = criterion(outputs, labels)
                aug_val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                aug_total += labels.size(0)
                aug_correct += (predicted == labels).sum().item()

        avg_aug_val_loss = aug_val_loss / len(aug_val_loader)
        aug_val_accuracy = 100 * aug_correct / aug_total

        # Print all metrics clearly to monitor both standard and robust performance
        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] - Train Loss: {avg_train_loss:.4f} | "
            f"Clean Val Acc: {clean_val_accuracy:.2f}% | Robust Val Acc: {aug_val_accuracy:.2f}%"
        )

    # TODO: save trained model weights to weights.joblib
    # Move model back to CPU before saving to ensure hardware-independent loading later!
    joblib.dump(model.cpu().state_dict(), OUTPUT)
    print("Saved trained weights.joblib")


if __name__ == "__main__":
    main()