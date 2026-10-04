from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets

from src.preprocessing.preprocess import (
    train_transform,
    validation_transform,
    test_transform
)


# ==========================================
# Dataset paths
# ==========================================

TRAIN_DIR = Path("dataset/train")
VALIDATION_DIR = Path("dataset/validation")
TEST_DIR = Path("dataset/test")


# ==========================================
# DataLoader configuration
# ==========================================

BATCH_SIZE = 16
NUM_WORKERS = 0


# ==========================================
# Create datasets
# ==========================================

def create_datasets():

    train_dataset = datasets.ImageFolder(
        root=TRAIN_DIR,
        transform=train_transform
    )

    validation_dataset = datasets.ImageFolder(
        root=VALIDATION_DIR,
        transform=validation_transform
    )

    test_dataset = datasets.ImageFolder(
        root=TEST_DIR,
        transform=test_transform
    )

    return (
        train_dataset,
        validation_dataset,
        test_dataset
    )


# ==========================================
# Create DataLoaders
# ==========================================

def create_dataloaders():

    (
        train_dataset,
        validation_dataset,
        test_dataset
    ) = create_datasets()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    return (
        train_loader,
        validation_loader,
        test_loader
    )


# ==========================================
# Test
# ==========================================

if __name__ == "__main__":

    (
        train_loader,
        validation_loader,
        test_loader
    ) = create_dataloaders()

    print("=" * 50)
    print("DATASET LOADER")
    print("=" * 50)

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

    print(
        "\nClasses:",
        train_loader.dataset.classes
    )

    print(
        "Class mapping:",
        train_loader.dataset.class_to_idx
    )

    images, labels = next(iter(train_loader))

    print(
        "\nBatch image shape:",
        images.shape
    )

    print(
        "Batch label shape:",
        labels.shape
    )

    print(
        "First labels:",
        labels[:10]
    )