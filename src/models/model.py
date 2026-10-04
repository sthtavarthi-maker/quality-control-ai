import torch
import torch.nn as nn
from torchvision import models


# ==========================================
# Number of defect classes
# ==========================================

NUM_CLASSES = 6


# ==========================================
# Create ResNet18 model
# ==========================================

def create_model():

    # Load pretrained ResNet18
    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    )

    # Get number of inputs to final layer
    input_features = model.fc.in_features

    # Replace final classification layer
    model.fc = nn.Linear(
        input_features,
        NUM_CLASSES
    )

    return model


# ==========================================
# Main test
# ==========================================

if __name__ == "__main__":

    model = create_model()

    print("=" * 50)
    print("ResNet18 Model")
    print("=" * 50)

    print("Architecture: ResNet18")
    print("Pretrained: Yes")
    print("Number of classes:", NUM_CLASSES)

    print("\nFinal layer:")
    print(model.fc)

    print("\nModel created successfully!")