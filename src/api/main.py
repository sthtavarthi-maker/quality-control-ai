from io import BytesIO
from pathlib import Path

import torch

from PIL import Image, UnidentifiedImageError

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from torchvision import transforms

from src.models.model import create_model
from src.models.device import get_device


# ==========================================
# Configuration
# ==========================================

MODEL_PATH = Path("models/best_model.pth")

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
}


# ==========================================
# Class names
# ==========================================

CLASS_NAMES = {
    0: "Crazing",
    1: "Inclusion",
    2: "Pitted Surface",
    3: "Patches",
    4: "Rolled-in Scale",
    5: "Scratches"
}


# ==========================================
# FastAPI application
# ==========================================

app = FastAPI(
    title="Steel Defect Quality Control API",
    description=(
        "AI-powered steel surface defect "
        "classification using ResNet18."
    ),
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]
)
# ==========================================
# Device
# ==========================================

device = get_device()


# ==========================================
# Load trained model
# ==========================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )


model = create_model()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()


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
# Root endpoint
# ==========================================

@app.get("/")
def root():

    return {
        "message": "Steel Defect Quality Control API",
        "status": "running"
    }


# ==========================================
# Health endpoint
# ==========================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": "ResNet18",
        "device": str(device)
    }


# ==========================================
# Prediction endpoint
# ==========================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # --------------------------------------
    # Check filename
    # --------------------------------------

    if not file.filename:

        return JSONResponse(
            status_code=400,
            content={
                "error": "No filename provided."
            }
        )


    # --------------------------------------
    # Check file extension
    # --------------------------------------

    extension = Path(
        file.filename
    ).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:

        return JSONResponse(
            status_code=400,
            content={
                "error": (
                    "Unsupported file type. "
                    "Allowed formats: JPG, JPEG, "
                    "PNG, BMP."
                )
            }
        )


    try:

        # ----------------------------------
        # Read uploaded file
        # ----------------------------------

        contents = await file.read()

        if not contents:

            return JSONResponse(
                status_code=400,
                content={
                    "error": "Uploaded file is empty."
                }
            )


        # ----------------------------------
        # Open image
        # ----------------------------------

        image = Image.open(
            BytesIO(contents)
        )

        # Force loading to detect corruption
        image.load()


        # ----------------------------------
        # Preprocess image
        # ----------------------------------

        image_tensor = transform(
            image
        )

        image_tensor = (
            image_tensor.unsqueeze(0)
        )

        image_tensor = image_tensor.to(
            device
        )


        # ----------------------------------
        # Model prediction
        # ----------------------------------

        with torch.no_grad():

            outputs = model(
                image_tensor
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence, predicted_class = (
                torch.max(
                    probabilities,
                    1
                )
            )


        # ----------------------------------
        # Convert prediction
        # ----------------------------------

        predicted_index = (
            predicted_class.item()
        )

        predicted_name = CLASS_NAMES[
            predicted_index
        ]

        confidence_value = (
            confidence.item() * 100
        )
        probability_values = {
        CLASS_NAMES[index]: round(
            probabilities[0][index].item() * 100,
            2
        )
        for index in range(len(CLASS_NAMES))
        }


        # ----------------------------------
        # Return result
        # ----------------------------------

        return {
            "filename": file.filename,
            "predicted_defect": predicted_name,
            "confidence": round(
                confidence_value,
                2
            ),
            "probabilities": probability_values
        }


    except UnidentifiedImageError:

        return JSONResponse(
            status_code=400,
            content={
                "error": (
                    "The uploaded file is not "
                    "a valid image."
                )
            }
        )


    except Exception as e:

        return JSONResponse(
            status_code=500,
            content={
                "error": (
                    "Prediction failed."
                ),
                "details": str(e)
            }
        )