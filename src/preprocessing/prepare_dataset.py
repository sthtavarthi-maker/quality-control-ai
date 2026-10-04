import random
import shutil
from pathlib import Path

# ==========================================
# Configuration
# ==========================================

RAW_DIR = Path("dataset/raw/NEU-CLS")
OUTPUT_DIR = Path("dataset")

CLASSES = ["Cr", "In", "Pa", "PS", "RS", "Sc"]

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

random.seed(RANDOM_SEED)


# ==========================================
# Find images by class
# ==========================================

def find_images(class_name):

    images = []

    for file in RAW_DIR.iterdir():

        if not file.is_file():
            continue

        if file.suffix.lower() != ".bmp":
            continue

        # Example:
        # Cr_1.bmp
        # In_25.bmp
        # PS_100.bmp

        if file.name.startswith(class_name + "_"):
            images.append(file)

    return images


# ==========================================
# Split dataset
# ==========================================

def split_images(images):

    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)

    validation_end = (
        train_end +
        int(total * VALIDATION_RATIO)
    )

    train_images = images[:train_end]

    validation_images = images[
        train_end:validation_end
    ]

    test_images = images[
        validation_end:
    ]

    return (
        train_images,
        validation_images,
        test_images
    )


# ==========================================
# Copy images
# ==========================================

def copy_images(
    images,
    class_name,
    split_name
):

    destination = (
        OUTPUT_DIR /
        split_name /
        class_name
    )

    destination.mkdir(
        parents=True,
        exist_ok=True
    )

    for image in images:

        destination_file = (
            destination /
            image.name
        )

        shutil.copy2(
            image,
            destination_file
        )


# ==========================================
# Main
# ==========================================

def main():

    print("=" * 55)
    print("NEU-CLS DATASET PREPARATION")
    print("=" * 55)

    # Check raw dataset
    if not RAW_DIR.exists():

        print(
            f"ERROR: Dataset folder not found: {RAW_DIR}"
        )

        return

    total_dataset_images = 0

    for class_name in CLASSES:

        print(
            f"\nProcessing class: {class_name}"
        )

        images = find_images(class_name)

        print(
            f"Total images: {len(images)}"
        )

        if len(images) == 0:

            print(
                f"WARNING: No images found for {class_name}"
            )

            continue

        total_dataset_images += len(images)

        (
            train_images,
            validation_images,
            test_images
        ) = split_images(images)

        print(
            f"Train:       {len(train_images)}"
        )

        print(
            f"Validation:  {len(validation_images)}"
        )

        print(
            f"Test:        {len(test_images)}"
        )

        copy_images(
            train_images,
            class_name,
            "train"
        )

        copy_images(
            validation_images,
            class_name,
            "validation"
        )

        copy_images(
            test_images,
            class_name,
            "test"
        )

    print("\n" + "=" * 55)

    print(
        f"Total dataset images: {total_dataset_images}"
    )

    print(
        "DATASET PREPARATION COMPLETED"
    )

    print("=" * 55)


# ==========================================
# Run program
# ==========================================

if __name__ == "__main__":
    main()