# ASL Gesture Controller

Real-time American Sign Language (ASL) alphabet recognition with a Streamlit dashboard. The system detects hand gestures via webcam, classifies 24 ASL letters (A–Z, excluding J and Z), and supports two modes: **Text Generation** (spell words letter by letter) and **Web Actions** (open websites mapped to each sign).

Final exam project for **Advanced Computational Techniques for Big Imaging and Signal Data**.

## Overview

This project builds an end-to-end ASL recognition pipeline: custom CNN and transfer-learning models (MobileNetV2, EfficientNetB0) trained on hand-sign images, combined with real-time hand tracking (CVZone / MediaPipe) in an interactive Streamlit application. Users draw or show signs in a webcam feed; confirmed predictions can build text or trigger mapped web shortcuts.

## Features

- **Three classifier backends:** Basic CNN, MobileNetV2, and EfficientNetB0 (switchable in the sidebar)
- **Live webcam inference** with hand landmark visualization
- **Text Generation mode:** accumulate confirmed letters into a typed sentence
- **Web Actions mode:** map each ASL letter to a website shortcut (Gmail, YouTube, Google Search, etc.)
- **Stability filtering:** configurable confidence threshold and stable-frame count before accepting a sign
- **Action cooldown** to prevent duplicate triggers
- **Event history** and live FPS monitoring
- **Reference sign chart** (`signs.png`) for user guidance
- **Training notebooks** for all three model architectures

## Technologies

| Category | Stack |
|---|---|
| Language | Python 3.10+ |
| Deep Learning | TensorFlow / Keras |
| Computer Vision | OpenCV, CVZone Hand Tracking |
| Dashboard | Streamlit |
| Training | Jupyter Notebook |
| Data | Custom ASL alphabet image dataset (24 classes) |

## Project Structure

```text
ICT/
├── app.py                  # Streamlit dashboard (main demo)
├── classes_labels.json     # Class label mapping
├── signs.png               # ASL reference chart
├── requirements.txt        # Python dependencies
├── training/               # Model training notebooks
│   ├── basic_cnn.ipynb
│   ├── tranfer_learning_MobileNet_model.ipynb
│   └── transfer_learning_EfficientNet_model.ipynb
├── models/                 # Trained Keras weights
├── dataset/                # ASL training images (A–Y)
├── presentation/           # Project presentation PDFs
└── streamlit_fps_results.txt
```

## Installation

### Prerequisites

- Python 3.10 or newer
- Webcam (for live demo)
- Git

### Setup

```bash
git clone https://github.com/iabrarbajwa/asl-gesture-controller.git
cd asl-gesture-controller

python -m venv signEnv
```

**Windows:**

```bash
signEnv\Scripts\activate
pip install -r requirements.txt
```

**Linux / macOS:**

```bash
source signEnv/bin/activate
pip install -r requirements.txt
```

### Model Weights & Dataset

The repository includes the full `dataset/` folder, trained model weights in `models/`, and the MediaPipe `hand_landmarker.task` file. Clone the repo and install dependencies to run locally.

## Usage

### Run the Streamlit Dashboard

```bash
streamlit run app.py
```

1. Select a model (MobileNetV2 recommended).
2. Enable **Start webcam**.
3. Choose **Text Generation** or **Web Actions**.
4. Adjust confidence and stable-frame sliders as needed.
5. Show ASL signs to the camera; confirmed predictions appear in the main panel.

### Train a Model

Open the relevant notebook in `training/` and follow the cells. Images should be organized as:

```text
dataset/
├── A/
├── B/
├── ...
└── Y/
```

## Results

### Model Architectures

| Model | Type | Approx. Size |
|---|---|---:|
| Basic CNN | Custom convolutional network | ~255 MB |
| MobileNetV2 | Transfer learning (ImageNet) | ~13 MB |
| EfficientNetB0 | Transfer learning (ImageNet) | ~20 MB |

### Runtime Performance (Streamlit + Webcam)

Benchmarks recorded on local hardware (`streamlit_fps_results.txt`):

| Session Length | Average FPS |
|---:|---:|
| 30 s | 4.96 |
| 60 s | 10.19 |
| 120 s | 6.64 |

FPS varies with model choice, hand detection load, and system resources. MobileNetV2 and EfficientNetB0 generally offer a better speed/accuracy trade-off than the basic CNN.

## Future Improvements

- Add J and Z dynamic gesture recognition (currently excluded from static alphabet set)
- Export models to TensorFlow Lite / ONNX for edge deployment
- Replace full `pip freeze` requirements with a minimal dependency list
- Add unit tests for preprocessing and prediction pipelines
- Support sentence-level sign language (word sequences) beyond single letters
- Integrate text-to-speech for accessibility
- Publish model weights as GitHub Releases or Hugging Face artifacts
- Add Grad-CAM / saliency maps for interpretability

## Author

Course project — Advanced Computational Techniques for Big Imaging and Signal Data.
