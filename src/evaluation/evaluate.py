import torch

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from src.data.dataset_loader import create_dataloaders
from src.models.model import create_model
from src.models.device import get_device


# ==========================================
# Configuration
# ==========================================

MODEL_PATH = Path("models/best_model.pth")


# ==========================================
# Main evaluation
# ==========================================

def main():

    print("=" * 60)
    print("QUALITY CONTROL MODEL EVALUATION")
    print("=" * 60)

    # --------------------------------------
    # Device
    # --------------------------------------

    device = get_device()

    # --------------------------------------
    # Load dataset
    # --------------------------------------

    print("\nLoading test dataset...")

    _, _, test_loader = create_dataloaders()

    print(
        "Test images:",
        len(test_loader.dataset)
    )

    # --------------------------------------
    # Load model
    # --------------------------------------

    print("\nLoading trained model...")

    model = create_model()

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    model = model.to(device)

    model.eval()

    print("Model loaded successfully!")

    # --------------------------------------
    # Predictions
    # --------------------------------------

    all_predictions = []
    all_labels = []

    print("\nRunning predictions...")

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            outputs = model(images)

            _, predictions = torch.max(
                outputs,
                1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.numpy()
            )

    # --------------------------------------
    # Accuracy
    # --------------------------------------

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    print(
        f"\nTest Accuracy: {accuracy * 100:.2f}%"
    )

    # --------------------------------------
    # Classification report
    # --------------------------------------

    class_names = test_loader.dataset.classes

    print("\nClassification Report:")
    print()

    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=class_names
        )
    )

    # --------------------------------------
    # Confusion matrix
    # --------------------------------------

    print("Confusion Matrix:")

    matrix = confusion_matrix(
        all_labels,
        all_predictions
    )

    print(matrix)


if __name__ == "__main__":

    main()