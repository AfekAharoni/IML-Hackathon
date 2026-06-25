from pathlib import Path
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from change_images_for_training import (
    build_dataloaders,
    build_robust_validation_loader,
)
from model import ModelArchitecture

DATA_ROOT = PROJECT_ROOT / "dataset"
OUTPUT = Path(__file__).resolve().parent / "weights.joblib"

BATCH_SIZE = 64
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

    # Build DataLoaders using in-memory image manipulations from
    # change_images_for_training.py. The training images are randomly changed
    # every time they are loaded, then converted to tensors for the CNN.
    train_loader, clean_val_loader = build_dataloaders(
        data_root=DATA_ROOT / "train",
        batch_size=BATCH_SIZE,
        num_workers=0,
    )
    robust_val_loader = build_robust_validation_loader(
        clean_data_root=DATA_ROOT / "train",
        augmentations_root=DATA_ROOT / "augmentations" / "validation",
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

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

        # --- 2. Evaluate Robust Validation ---
        # This loader contains saved augmentation images and real validation
        # images with our random in-memory manipulations applied.
        robust_val_loss = 0.0
        robust_correct = 0
        robust_total = 0

        with torch.no_grad():
            for images, labels in robust_val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                loss = criterion(outputs, labels)
                robust_val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                robust_total += labels.size(0)
                robust_correct += (predicted == labels).sum().item()

        avg_robust_val_loss = robust_val_loss / len(robust_val_loader)
        robust_val_accuracy = 100 * robust_correct / robust_total

        # Print all metrics clearly to monitor both standard and robust performance
        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] - Train Loss: {avg_train_loss:.4f} | "
            f"Clean Val Loss: {avg_clean_val_loss:.4f} | "
            f"Clean Val Acc: {clean_val_accuracy:.2f}% | "
            f"Robust Val Loss: {avg_robust_val_loss:.4f} | "
            f"Robust Val Acc: {robust_val_accuracy:.2f}%"
        )

    # Move model back to CPU before saving to ensure hardware-independent loading later!
    joblib.dump(model.cpu().state_dict(), OUTPUT)
    print("Saved trained weights.joblib")


if __name__ == "__main__":
    main()
