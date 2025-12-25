# 🪸 EcoScan: Marine Forensic AI

**EcoScan** is a multimodal scientific analyzer designed to assess coral health. Using deep learning (ResNet18) and computer vision (Grad-CAM), it classifies coral images as **Healthy** or **Bleached** and provides visual explainability for its diagnoses.

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm-dark.svg)](https://huggingface.co/spaces/feby404/EcoScan)

## Features

- **AI-Powered Diagnosis**: Utilizing a fine-tuned ResNet18 model for accurate binary classification.
- **Explainability**: Integrated Grad-CAM (Gradient-Weighted Class Activation Mapping) to visualize which parts of the coral the model is focusing on.
- **Modern Web Interface**: A sleek, dark-themed web application built with FastAPI and Jinja2.
- **Real-time Inference**: Fast, local inference on CPU or GPU.

## Tech Stack

- **Backend**: Python, FastAPI, PyTorch, Torchvision
- **Frontend**: HTML5, CSS3, Jinja2
- **Utilities**: Pillow, Matplotlib, Numpy


## Dataset

The model was trained using the **[Coral Reefs Images](https://www.kaggle.com/datasets/asfarhossainsitab/coral-reefs-images)** dataset by Asfar Hossain Sitab on Kaggle.

## License

[MIT](LICENSE)
