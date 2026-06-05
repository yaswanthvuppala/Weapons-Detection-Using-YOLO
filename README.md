---
title: Weapon Detector
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: streamlit
app_file: app.py
pinned: false
---

# Weapons Detection Using YOLO11s

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/yaswanthvuppala/weapon-detector)
**[🚀 Try the Live Demo on Hugging Face!](https://huggingface.co/spaces/yaswanthvuppala/weapon-detector)**

A real-time deep learning computer vision system built to detect **people** and **weapons** (e.g., firearms, knives, and other weapons) from static images, recorded videos, and live webcam feeds. 

The project utilizes the state-of-the-art **YOLO11s (You Only Look Once)** architecture from Ultralytics, trained and evaluated on a custom dataset.

---

## 🛠️ Tools & Technologies Used
* **Deep Learning Framework**: `ultralytics` YOLO11
* **Language & Runtime**: Python 3.10
* **Compute Backend**: PyTorch 2.11.0 with CUDA 12.8 GPU acceleration
* **Hardware**: NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM)
* **GUI / Video Processing**: OpenCV (for webcam capture and real-time display)
* **Dataset Management**: Roboflow (for dataset hosting and annotations)

---

## 📂 Project Structure
```text
YOLO_Weapon_Detection/
├── .venv/                      # Python virtual environment
├── weapons/                    # Dataset directory
│   ├── train/                  # Training images & labels (annotation txt)
│   ├── valid/                  # Validation images & labels
│   ├── test/                   # Test split images & labels (797 images)
│   └── data.yaml               # Dataset config (paths, class names ['person', 'weapon'])
├── runs/
│   └── detect/
│       ├── weapon_detector-5/  # Core training run output directory
│       │   ├── weights/        # Trained weights: best.pt (used for evaluation) & last.pt
│       │   ├── results.csv     # Training loss and metrics logs per epoch
│       │   └── results.png     # Plot graphs for training loss and validation metrics
│       ├── val-2/              # Test split evaluation metrics (plots & confusion matrix)
│       └── predict-3/          # Prediction outputs on online test images
├── train.py                    # Script used to configure and trigger training
├── test_model.py               # Evaluates model on test data & runs prediction on online URLs
├── system_cam_test.py          # Script for real-time weapon detection via webcam
├── yolo_weapons.ipynb          # Jupyter notebook for interactive experimentation
└── README.md                   # This project documentation file
```

---

## 📈 Training Details
* **Pretrained Model**: YOLO11s (9.4 Million parameters)
* **Epochs Run**: **5 Epochs**
* **Batch Size**: 8
* **Image Size**: 640x640 pixels
* **Total Training Time**: 2155.2 seconds (~36 minutes) on the RTX 3050 GPU
* **Training Output**: Saved under `runs/detect/weapon_detector-5/`

---

## 📊 Evaluation Metrics (Test Split Results)
The model was tested on the unseen **test split** containing **797 images** (containing 1,836 labeled instances). Below is the classification report showing the performance metrics:

| Class | Precision (P) | Recall (R) | mAP50 | mAP50-95 |
| :--- | :---: | :---: | :---: | :---: |
| **person** | 75.00% | 65.23% | 0.6982 | 0.3982 |
| **weapon** | 72.75% | 45.08% | 0.5237 | 0.3345 |
| **All (Mean)** | **73.88%** | **55.16%** | **0.6110** | **0.3663** |

### Key takeaways:
* **Confusion Matrix**: Visualizes true vs. predicted classifications. Available at `runs/detect/val-2/confusion_matrix.png`.
* **Inference Speed**: Preprocess: 2.4ms, Inference: 13.7ms, Postprocess: 1.3ms per image (totaling ~17.4ms per frame, capable of **57 FPS** real-time speeds on your GPU).

---

## 📥 How to Download the Dataset

This dataset is hosted on **Roboflow Universe**. You can download it using one of two methods:

### Method 1: Download from Web Browser (Recommended)
1. Visit the dataset page: [Roboflow Universe Dataset](https://universe.roboflow.com/weapon-detection-qktol/weapon-detection-ipl7p/dataset/7)
2. Click the **Download Dataset** button in the upper right.
3. Select **YOLOv11** or **YOLOv8** format.
4. Choose **Download ZIP** and download it to your local machine.
5. Extract the ZIP file and rename the folder to `weapons`, placing it in the project root directory.

### Method 2: Download Programmatically using Python
Install the Roboflow library in your virtual environment:
```powershell
pip install roboflow
```

Create and run a simple python script:
```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_ROBOFLOW_PRIVATE_API_KEY") # Get API Key from Roboflow settings
project = rf.workspace("weapon-detection-qktol").project("weapon-detection-ipl7p")
version = project.version(7)
dataset = version.download("yolov11")
```

---

## 🚀 How to Run the Project

Ensure you activate your virtual environment in your terminal before running any commands:
```powershell
.venv\Scripts\activate
```

### 1. Training the Model
To retrain or fine-tune the model, run:
```powershell
python train.py
```

### 2. Testing on the Test Set & Random Online Images
To evaluate the model on the test split and test predictions on remote URL images (which download automatically to a temp folder):
```powershell
python test_model.py
```

### 3. Live Webcam Real-time Detection
To start the live weapon detector using your system's built-in webcam or camera:
```powershell
python system_cam_test.py
```
* *Note: Press **`q`** on your keyboard while focusing on the camera window to stop the feed.*
