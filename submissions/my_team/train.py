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

BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 15  # Target set for fast but robust convergence


def main():
    """
    Full training pipeline.

    This script must create weights.joblib.
    """
    # Determine the computing device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

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

    # Label smoothing prevents the model from being overconfident on noisy images
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # AdamW incorporates strict weight decay, far superior for deep ResNets
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # OneCycleLR is specifically designed to maximize accuracy in short runs (like 15 epochs)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE * 5,  # Peak learning rate
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS
    )

    # Begin the training loop
    print("Start training")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # OneCycleLR MUST be stepped after every batch, not every epoch!
            scheduler.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # Validation step
        model.eval()

        # --- 1. Evaluate Clean Validation ---
        clean_val_loss = 0.0
        clean_correct = 0
        clean_total = 0

        with torch.no_grad():
            for images, labels in clean_val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                loss = criterion(outputs, labels)
                clean_val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                clean_total += labels.size(0)
                clean_correct += (predicted == labels).sum().item()

        avg_clean_val_loss = clean_val_loss / len(clean_val_loader)
        clean_val_accuracy = 100 * clean_correct / clean_total

        # --- 2. Evaluate Robust Validation ---
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

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] - Train Loss: {avg_train_loss:.4f} | "
            f"Clean Val Loss: {avg_clean_val_loss:.4f} | "
            f"Clean Val Acc: {clean_val_accuracy:.2f}% | "
            f"Robust Val Loss: {avg_robust_val_loss:.4f} | "
            f"Robust Val Acc: {robust_val_accuracy:.2f}% | "
            f"End LR: {current_lr:.6f}"
        )

    joblib.dump(model.cpu().state_dict(), OUTPUT)
    print("Saved trained weights.joblib")


if __name__ == "__main__":
    main()