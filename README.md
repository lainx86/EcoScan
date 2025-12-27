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

### Installation (via Winget)

   ```bash
   winget install lainx86.EcoScan
   ```


## Dataset

The model was trained using the **[Coral Reefs Images](https://www.kaggle.com/datasets/asfarhossainsitab/coral-reefs-images)** dataset by Asfar Hossain Sitab on Kaggle.

## License

[MIT](LICENSE)
