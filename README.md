# 🪸 EcoScan: Marine Forensic AI

**EcoScan** is a multimodal scientific analyzer designed to assess coral health. Using deep learning (ResNet18) and computer vision (Grad-CAM), it classifies coral images as **Healthy** or **Bleached** and provides visual explainability for its diagnoses.

## ✨ Features

- **AI-Powered Diagnosis**: Utilizing a fine-tuned ResNet18 model for accurate binary classification.
- **Explainability**: Integrated Grad-CAM (Gradient-Weighted Class Activation Mapping) to visualize which parts of the coral the model is focusing on.
- **Modern Web Interface**: A sleek, dark-themed web application built with FastAPI and Jinja2.
- **Real-time Inference**: Fast, local inference on CPU or GPU.

## 🛠️ Tech Stack

- **Backend**: Python, FastAPI, PyTorch, Torchvision
- **Frontend**: HTML5, CSS3, Jinja2
- **Utilities**: Pillow, Matplotlib, Numpy

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/lainx86/EcoScan.git
   cd EcoScan
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the model file `coral_bleaching_resnet18.pt` is in the root directory.

### Usage

1. Start the server:
   ```bash
   uvicorn app:app --reload
   ```

2. Open your browser and navigate to:
   ```
   http://127.0.0.1:8000
   ```

3. Upload a coral image to receive a health diagnosis and heat map visualization.

## 📂 Project Structure

```
EcoScan/
├── app.py              # Main FastAPI application
├── model.py            # Model loading and inference logic
├── gradcam.py          # Grad-CAM implementation
├── main.py             # (Legacy) API-only entry point
├── requirements.txt    # Project dependencies
├── coral_bleaching_resnet18.pt # Trained model weights
├── templates/
│   └── index.html      # Frontend interface
└── static/
    └── outputs/        # Generated Grad-CAM results
```

## 📊 Dataset

The model was trained using the **[Coral Reefs Images](https://www.kaggle.com/datasets/asfarhossainsitab/coral-reefs-images)** dataset by Asfar Hossain Sitab on Kaggle.

## 📄 License

[MIT](LICENSE)
