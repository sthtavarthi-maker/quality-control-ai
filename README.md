# Steel Defect Quality Control AI

An AI-powered steel surface defect classification system built using deep learning. The system accepts a steel surface image and predicts the type of defect using a pretrained ResNet18 model.

The project provides a web-based interface where users can upload an image and view the predicted defect, confidence score, and class-wise probabilities.

## Project Overview

Steel surface defects can affect the quality and reliability of manufactured steel products. Manual inspection can be time-consuming and may depend on the experience of the inspector.

This project uses a deep learning image classification model to automatically identify common steel surface defects from images.

The system consists of:

- React + Vite frontend
- FastAPI backend
- PyTorch deep learning model
- ResNet18 architecture
- NEU-CLS steel surface defect dataset

## Defect Classes

The model classifies images into six steel surface defect categories:

| Code | Defect |
|------|--------|
| Cr | Crazing |
| In | Inclusion |
| Pa | Patches |
| PS | Pitted Surface |
| RS | Rolled-in Scale |
| Sc | Scratches |

## System Architecture

```text
User
  |
  v
React + Vite Frontend
  |
  | Image Upload
  v
FastAPI Backend
  |
  v
Image Preprocessing
  |
  v
ResNet18 Model
  |
  v
Defect Classification
  |
  +-----------------------+
  |                       |
  v                       v
Predicted Defect     Class Probabilities
  |
  v
Confidence Score
  |
  v
Frontend Result
```
  ## Project Structure

```bash
quality-control-ai/
│
├── frontend/
│   ├── public/
│   └── src/
│       ├── assets/
│       ├── api/
│       ├── App.jsx
│       ├── App.css
│       ├── index.css
│       └── main.jsx
│
├── notebooks/
│   └── 01_dataset_exploration.ipynb
│
├── src/
│   ├── api/
│   │   └── main.py
│   ├── data/
│   │   └── dataset_loader.py
│   ├── evaluation/
│   │   ├── confusion_matrix.py
│   │   └── evaluate.py
│   ├── models/
│   │   ├── device.py
│   │   ├── model.py
│   │   └── domain_model.py
│   ├── preprocessing/
│   │   ├── prepare_dataset.py
│   │   ├── preprocess.py
│   │   └── test_augmentation.py
│   └── training/
│       └── train.py
│
├── .gitignore
├── requirements.txt
├── package.json
├── package-lock.json
└── README.md
```
## Technologies Used
Machine Learning
Python
PyTorch
Torchvision
ResNet18
Pillow
Backend
FastAPI
Uvicorn
Python
Frontend
React
Vite
JavaScript
CSS
Tools
Git
GitHub
Jupyter Notebook

## Installation
Clone the Repository
git clone https://github.com/sthtavarthi-maker/quality-control-ai.git
cd quality-control-ai
Create a Python Virtual Environment

On Windows:

python -m venv venv

Activate the environment:

.\venv\Scripts\Activate.ps1
Install Python Dependencies
pip install -r requirements.txt
Dataset Setup

Place the NEU-CLS dataset inside the project directory using the following structure:

```text
dataset/
├── train/
│   ├── Cr/
│   ├── In/
│   ├── Pa/
│   ├── PS/
│   ├── RS/
│   └── Sc/
│
├── validation/
│   ├── Cr/
│   ├── In/
│   ├── Pa/
│   ├── PS/
│   ├── RS/
│   └── Sc/
│
└── test/
    ├── Cr/
    ├── In/
    ├── Pa/
    ├── PS/
    ├── RS/
    └── Sc/
```

    Training the Model

The training script can be executed using:
```powershell
.\venv\Scripts\python.exe -m src.training.train
```
The training process evaluates the model on the validation dataset and saves the best-performing model.

The trained model is saved as:

models/best_model.pth

Model files are excluded from Git using .gitignore.

## Running the Backend

Start the FastAPI backend using:

```powershell
.\venv\Scripts\python.exe -m uvicorn src.api.main:app --reload
```

The backend runs at:

http://127.0.0.1:8000

FastAPI interactive API documentation:

http://127.0.0.1:8000/docs
Prediction API

The main prediction endpoint is:

POST /predict

The endpoint accepts an image and returns:

Predicted defect
Confidence score
Probability of each defect class

Example response:
```json
{
  "predicted_defect": "Pitted Surface",
  "confidence": 99.39,
  "probabilities": {
    "Crazing": 0.12,
    "Inclusion": 0.18,
    "Patches": 0.03,
    "Pitted Surface": 99.39,
    "Rolled-in Scale": 0.20,
    "Scratches": 0.08
  }
}
```
## Running the Frontend

Open another terminal and navigate to the frontend:

```powershell
cd frontend
```

Install the frontend dependencies:

```powershell
npm install
```

Start the development server:

```powershell
npm run dev
```

The frontend is normally available at:

http://localhost:5173
Prediction Workflow
The user opens the web application.
The user uploads a steel surface image.
The frontend sends the image to the FastAPI backend.
The backend preprocesses the image.
The ResNet18 model performs classification.
The system calculates the predicted defect and confidence.
Class-wise probabilities are returned.
The frontend displays the prediction results.
Current Limitations

The current system is a closed-set six-class image classifier.

It is designed to classify images into the six supported NEU-CLS steel defect categories. It does not currently contain a dedicated Steel/Non-Steel rejection classifier or an out-of-distribution detection mechanism.

Therefore, an image outside the training distribution may still be assigned to one of the six available defect classes.

Future Improvements

Possible future improvements include:

Steel/Non-Steel image classification
Out-of-distribution image detection
Larger and more diverse datasets
Model performance optimization
Cloud deployment
Model monitoring
Improved frontend visualization
Additional defect categories
## Author

Keerthana Thtavarthi

## License

This project is developed for educational and research purposes.

