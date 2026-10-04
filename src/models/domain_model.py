import torch
import torch.nn as nn
from torchvision import models


# ==========================================
# Configuration
# ==========================================

NUM_CLASSES = 2


# ==========================================
# Create Domain Classification Model
# ==========================================

def create_domain_model():

    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    )

    input_features = model.fc.in_features

    model.fc = nn.Linear(
        input_features,
        NUM_CLASSES
    )

    return model


# ==========================================
# Test
# ==========================================

if __name__ == "__main__":

    model = create_domain_model()

    print("=" * 50)
    print("STEEL / NON-STEEL DOMAIN MODEL")
    print("=" * 50)

    print("Architecture: ResNet18")
    print("Pretrained: Yes")
    print("Number of classes:", NUM_CLASSES)

    print("\nFinal layer:")
    print(model.fc)

    print("\nModel created successfully!")