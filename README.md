# Duality Detector

A YOLOv8-based object detection system for safety equipment detection in industrial environments.

[![CI](https://github.com/your-username/duality-detector/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/duality-detector/actions/workflows/ci.yml)

## Quick Start

1. Create and activate virtual environment:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:
```powershell
pip install -r requirements.txt
```

3. Train the model (produces `runs/baseline/weights/best.pt`):
```powershell
python scripts/train.py
```

4. Start the web server:
```powershell
uvicorn app.main:app --reload
```

5. Open http://127.0.0.1:8000 and upload an image for detection.

## Dataset Setup

Place your images and labels under the following structure with YOLO format:

```
data/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Ensure `data/config.yaml` matches your dataset structure and class names:
- OxygenTank
- NitrogenTank  
- FirstAidBox
- FireAlarm
- SafetySwitchPanel
- EmergencyPhone
- FireExtinguisher

## Scripts

- `scripts/train.py` - Train YOLOv8 model
- `scripts/predict.py` - Run inference on images
- `app/main.py` - FastAPI web application

## Project Structure

```
duality-detector/
├── app/                    # Web application
│   ├── main.py            # FastAPI server
│   ├── index.html         # Frontend UI
│   └── style.css          # Styles
├── data/                   # Dataset and configuration
│   ├── config.yaml        # YOLO dataset configuration
│   ├── yolo_params.yaml   # Training parameters
│   ├── train/             # Training data
│   ├── valid/             # Validation data
│   └── test/              # Test data
├── models/                 # Model artifacts
├── runs/                   # Training outputs
│   └── baseline/          # Training results
│       └── weights/       # Model weights
├── scripts/                # Training and inference scripts
│   ├── train.py           # Training script
│   ├── predict.py         # Inference script
│   └── package_submission.ps1  # Packaging script
├── .github/                # CI/CD
│   └── workflows/         
│       └── ci.yml         # GitHub Actions
├── requirements.txt        # Dependencies
├── .gitignore             # Git ignore rules
└── README.md              # This file
```

## Usage

### Web Interface
1. Start the server: `uvicorn app.main:app --reload`
2. Open http://127.0.0.1:8000
3. Upload an image to detect safety equipment

### Command Line
```bash
# Run inference on a single image
python scripts/predict.py --source path/to/image.png

# Run inference on a folder
python scripts/predict.py --source path/to/folder

# Train the model (requires proper dataset)
python scripts/train.py --epochs 30 --batch 16
```

## Safety Equipment Classes
1. **OxygenTank** - Oxygen storage containers
2. **NitrogenTank** - Nitrogen storage containers  
3. **FirstAidBox** - Emergency medical supplies
4. **FireAlarm** - Fire detection systems
5. **SafetySwitchPanel** - Emergency stop controls
6. **EmergencyPhone** - Emergency communication devices
7. **FireExtinguisher** - Fire suppression equipment
