from torchvision import transforms


# ==========================================
# Image configuration
# ==========================================

IMAGE_SIZE = 224


# ==========================================
# Training transformations
# ==========================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(degrees=10),

    transforms.Grayscale(num_output_channels=3),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================================
# Validation transformations
# ==========================================

validation_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.Grayscale(num_output_channels=3),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# Test transformations
# ==========================================

test_transform = transforms.Compose([
    
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    
    transforms.ToTensor(),
    
    transforms.Grayscale(
        num_output_channels=3
    ),
    
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


if __name__ == "__main__":
    
    print("Image preprocessing configuration")
    print(f"Image size: {IMAGE_SIZE} x {IMAGE_SIZE}")
    print("Training augmentation: Enabled")
    print("Validation augmentation: Disabled")
    print("Test augmentation: Disabled")
    print("Output channels: 3")