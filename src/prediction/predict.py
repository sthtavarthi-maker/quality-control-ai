import sys
import torch

from pathlib import Path
from PIL import Image
from torchvision import transforms

from src.models.model import create_model
from src.models.device import get_device


# ==========================================
# Configuration
# ==========================================

MODEL_PATH = Path("models/best_model.pth")


# ==========================================
# Class names
# ==========================================

# IMPORTANT:
# This order MUST match the ImageFolder
# class_to_idx mapping used during training.
#
# 0 -> Cr
# 1 -> In
# 2 -> PS
# 3 -> Pa
# 4 -> RS
# 5 -> Sc

CLASS_NAMES = {
    0: "Cr",
    1: "In",
    2: "PS",
    3: "Pa",
    4: "RS",
    5: "Sc"
}


# ==========================================
# Human-readable class names
# ==========================================

CLASS_DISPLAY_NAMES = {
    0: "Crazing",
    1: "Inclusion",
    2: "Pitted Surface",
    3: "Patches",
    4: "Rolled-in Scale",
    5: "Scratches"
}


# ==========================================
# Image preprocessing
# ==========================================

transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.Grayscale(
        num_output_channels=3
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# Prediction function
# ==========================================

def predict_image(
    image_path,
    model,
    device
):

    image = Image.open(image_path)

    image_tensor = transform(image)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted_class = torch.max(
            probabilities,
            1
        )

    # --------------------------------------
    # Convert prediction
    # --------------------------------------

    predicted_index = predicted_class.item()

    confidence_value = confidence.item() * 100

    predicted_name = CLASS_NAMES[
        predicted_index
    ]

    predicted_display_name = CLASS_DISPLAY_NAMES[
        predicted_index
    ]

    # --------------------------------------
    # Calculate all class probabilities
    # --------------------------------------

    probability_values = {
        CLASS_NAMES[index]: round(
            probabilities[0][index].item() * 100,
            2
        )
        for index in range(len(CLASS_NAMES))
    }

    # --------------------------------------
    # Calculate display probabilities
    # --------------------------------------

    display_probability_values = {
        CLASS_DISPLAY_NAMES[index]: round(
            probabilities[0][index].item() * 100,
            2
        )
        for index in range(len(CLASS_DISPLAY_NAMES))
    }

    # --------------------------------------
    # Return prediction results
    # --------------------------------------

    return (
        predicted_name,
        predicted_display_name,
        confidence_value,
        probability_values,
        display_probability_values
    )


# ==========================================
# Main
# ==========================================

def main():

    print("=" * 60)
    print("STEEL DEFECT PREDICTION")
    print("=" * 60)

    # --------------------------------------
    # Get image path from command line
    # --------------------------------------

    if len(sys.argv) < 2:

        print("\nUsage:")

        print(
            "python -m src.prediction.predict "
            "<image_path>"
        )

        return

    image_path = Path(
        sys.argv[1]
    )

    # --------------------------------------
    # Check model
    # --------------------------------------

    if not MODEL_PATH.exists():

        print(
            "\nERROR: Model file not found:"
        )

        print(MODEL_PATH)

        return

    # --------------------------------------
    # Check image
    # --------------------------------------

    if not image_path.exists():

        print(
            "\nERROR: Image file not found:"
        )

        print(image_path)

        return

    # --------------------------------------
    # Device
    # --------------------------------------

    device = get_device()

    print(
        f"\nUsing device: {device}"
    )

    # --------------------------------------
    # Load model
    # --------------------------------------

    print("\nLoading model...")

    model = create_model()

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    model = model.to(device)

    model.eval()

    print(
        "Model loaded successfully!"
    )

    # --------------------------------------
    # Predict
    # --------------------------------------

    print("\nRunning prediction...")

    (
        predicted_name,
        predicted_display_name,
        confidence,
        probability_values,
        display_probability_values
    ) = predict_image(
        image_path,
        model,
        device
    )

    # --------------------------------------
    # Display result
    # --------------------------------------

    print("\n" + "=" * 60)

    print("PREDICTION RESULT")

    print("=" * 60)

    print(
        "\nImage:",
        image_path
    )

    print(
        "\nPredicted Defect:",
        predicted_display_name
    )

    print(
        "Class Code:",
        predicted_name
    )

    print(
        f"Confidence: {confidence:.2f}%"
    )

    print("\nClass Probabilities:")

    for class_name, probability in display_probability_values.items():

        print(
            f"{class_name}: {probability:.2f}%"
        )

    print("\n" + "=" * 60)


# ==========================================
# Run program
# ==========================================

if __name__ == "__main__":

    main()