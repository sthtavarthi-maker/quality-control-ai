import torch
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from src.data.dataset_loader import create_dataloaders
from src.models.model import create_model
from src.models.device import get_device


MODEL_PATH = Path("models/best_model.pth")


def main():

    print("=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)

    device = get_device()

    _, _, test_loader = create_dataloaders()

    model = create_model()

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    model = model.to(device)

    model.eval()

    all_predictions = []
    all_labels = []

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

    class_names = test_loader.dataset.classes

    matrix = confusion_matrix(
        all_labels,
        all_predictions
    )

    print("\nClass order:")
    print(class_names)

    print("\nConfusion matrix:")
    print(matrix)

    # --------------------------------------
    # Display matrix
    # --------------------------------------

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names
    )

    display.plot(
        xticks_rotation=45
    )

    plt.title(
        "NEU Steel Defect Classification"
    )

    plt.tight_layout()

    # --------------------------------------
    # Save figure
    # --------------------------------------

    output_dir = Path("results")

    output_dir.mkdir(
        exist_ok=True
    )

    output_path = (
        output_dir /
        "confusion_matrix.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(
        "\nConfusion matrix saved to:"
    )

    print(output_path)

    plt.show()


if __name__ == "__main__":

    main()