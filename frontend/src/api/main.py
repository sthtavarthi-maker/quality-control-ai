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


MODEL_PATH = Path("models/best_model.pth")

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
}


CLASS_NAMES = {
    0: "Crazing",
    1: "Inclusion",
    2: "Pitted Surface",
    3: "Patches",
    4: "Rolled-in Scale",
    5: "Scratches"
}


app = FastAPI(
    title="Steel Defect Quality Control API",
    description="AI-powered steel surface defect classification using ResNet18.",
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