from pathlib import Path

from PIL import Image
import matplotlib.pyplot as plt

from torchvision import transforms


# ==========================================
# Image path
# ==========================================

IMAGE_PATH = Path(
    "dataset/train/Cr/Cr_1.bmp"
)


# ==========================================
# Load image
# ==========================================

image = Image.open(IMAGE_PATH)


# ==========================================
# Training augmentation
# ==========================================

augmentation = transforms.Compose([
    
    transforms.Resize((224, 224)),
    
    transforms.RandomHorizontalFlip(p=0.5),
    
    transforms.RandomRotation(
        degrees=10
    ),
    
])


# ==========================================
# Generate augmented images
# ==========================================

augmented_images = []

for _ in range(4):

    augmented = augmentation(image)

    augmented_images.append(augmented)


# ==========================================
# Display results
# ==========================================

fig, axes = plt.subplots(
    1,
    4,
    figsize=(12, 3)
)


for ax, augmented in zip(
    axes,
    augmented_images
):

    ax.imshow(
        augmented,
        cmap="gray"
    )

    ax.axis("off")


plt.tight_layout()

plt.show()