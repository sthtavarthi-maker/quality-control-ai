import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path

from src.data.dataset_loader import create_dataloaders
from src.models.model import create_model
from src.models.device import get_device


# ==========================================
# Configuration
# ==========================================

NUM_EPOCHS = 10

LEARNING_RATE = 0.001

MODEL_DIR = Path("models")

BEST_MODEL_PATH = MODEL_DIR / "best_model.pth"


# ==========================================
# Training function
# ==========================================

def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update model weights
        optimizer.step()

        # Statistics
        running_loss += loss.item()

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    epoch_loss = (
        running_loss /
        len(train_loader)
    )

    epoch_accuracy = (
        100.0 * correct / total
    )

    return epoch_loss, epoch_accuracy


# ==========================================
# Validation function
# ==========================================

def validate(
    model,
    validation_loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += loss.item()

            _, predicted = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    validation_loss = (
        running_loss /
        len(validation_loader)
    )

    validation_accuracy = (
        100.0 * correct / total
    )

    return (
        validation_loss,
        validation_accuracy
    )


# ==========================================
# Main training
# ==========================================

def main():

    print("=" * 60)
    print("IMAGE RECOGNITION FOR QUALITY CONTROL")
    print("RESNET18 TRAINING")
    print("=" * 60)

    # --------------------------------------
    # Device
    # --------------------------------------

    device = get_device()

    print("\nDevice:", device)

    # --------------------------------------
    # Dataset
    # --------------------------------------

    print("\nLoading dataset...")

    (
        train_loader,
        validation_loader,
        test_loader
    ) = create_dataloaders()

    print(
        "Training images:",
        len(train_loader.dataset)
    )

    print(
        "Validation images:",
        len(validation_loader.dataset)
    )

    print(
        "Test images:",
        len(test_loader.dataset)
    )

    # --------------------------------------
    # Model
    # --------------------------------------

    print("\nCreating ResNet18...")

    model = create_model()

    model = model.to(device)

    # --------------------------------------
    # Loss function
    # --------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------
    # Optimizer
    # --------------------------------------

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------
    # Create model directory
    # --------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------
    # Best validation accuracy
    # --------------------------------------

    best_validation_accuracy = 0.0

    # ======================================
    # Training loop
    # ======================================

    for epoch in range(NUM_EPOCHS):

        print("\n" + "-" * 60)

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS}"
        )

        print("-" * 60)

        # Training
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        # Validation
        validation_loss, validation_accuracy = validate(
            model,
            validation_loader,
            criterion,
            device
        )

        # ----------------------------------
        # Display results
        # ----------------------------------

        print(
            f"Train Loss: {train_loss:.4f}"
        )

        print(
            f"Train Accuracy: {train_accuracy:.2f}%"
        )

        print(
            f"Validation Loss: {validation_loss:.4f}"
        )

        print(
            f"Validation Accuracy: "
            f"{validation_accuracy:.2f}%"
        )

        # ----------------------------------
        # Save best model
        # ----------------------------------

        if validation_accuracy > best_validation_accuracy:

            best_validation_accuracy = validation_accuracy

            torch.save(
                model.state_dict(),
                BEST_MODEL_PATH
            )

            print(
                "\n✓ Best model saved!"
            )

            print(
                "Path:",
                BEST_MODEL_PATH
            )

    # ======================================
    # Training completed
    # ======================================

    print("\n" + "=" * 60)

    print("TRAINING COMPLETED")

    print("=" * 60)

    print(
        f"Best Validation Accuracy: "
        f"{best_validation_accuracy:.2f}%"
    )

    print(
        "Best model:",
        BEST_MODEL_PATH
    )


if __name__ == "__main__":

    main()