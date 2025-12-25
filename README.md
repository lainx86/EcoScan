---
title: EcoScan
emoji: 🪸
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# EcoScan: Coral Bleaching Detection

**EcoScan** is a desktop application designed to assess coral health. Using deep learning (ResNet18) and computer vision (Grad-CAM), it classifies coral images as **Healthy** or **Bleached** and provides visual explainability for its diagnoses.

## Features

- **Desktop Application**: A standalone Windows Desktop App built with Tkinter.
- **AI-Powered Diagnosis**: Utilizing a fine-tuned ResNet18 model for accurate binary classification.
- **Explainability**: Integrated Grad-CAM (Gradient-Weighted Class Activation Mapping) to visualize which parts of the coral the model is focusing on.
- **Modern Dark UI**: Sleek, dark-themed interface.
- **Real-time Inference**: Fast, local inference on CPU or GPU.

## Tech Stack

- **Core**: Python, PyTorch, Torchvision
- **GUI**: Tkinter
- **Utilities**: Pillow, Matplotlib, Numpy

## Getting Started

### Prerequisites

- Python 3.8+
- pip
- Git LFS (for downloading larger model files)

### Installation

1. Clone the repository and pull large files:
   ```bash
   git clone https://github.com/lainx86/EcoScan.git
   cd EcoScan
   git lfs pull
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Tkinter is included with standard Python installations.*

3. Ensure the model file `coral_bleaching_resnet18.pt` (~43MB) is in the root directory.

### Usage

**Option 1: Pre-built Executable (Distribution)**
If you have the distributed folder:
1. Open the `EcoScan` folder.
2. Double-click `EcoScan.exe`.

**Option 2: Run from Source**
See "Development" below.

## Development

### Running with Python
```bash
python desktop_app.py
```
or use the launcher:
```bash
run_app.bat
```

### Building the Executable
To create a standalone `.exe` for distribution:

1. Install PyInstaller:
   ```bash
   pip install pyinstaller
   ```
2. Run the build script:
   ```bash
   build_exe.bat
   ```
3. The executable will be created in `dist/EcoScan/EcoScan.exe`.
   You can zip the `dist/EcoScan` folder and share it with others.

## Project Structure

```
EcoScan/
├── ml/                 # Shared Machine Learning logic (Model, GradCAM)
├── desktop_app.py      # Desktop Application entry point
├── run_app.bat         # Launcher for Desktop App
├── coral_bleaching_resnet18.pt # Trained model weights
├── requirements.txt    # Project dependencies
└── README.md           # Documentation
```

## Dataset

The model was trained using the **[Coral Reefs Images](https://www.kaggle.com/datasets/asfarhossainsitab/coral-reefs-images)** dataset by Asfar Hossain Sitab on Kaggle.

## License

[MIT](LICENSE)
