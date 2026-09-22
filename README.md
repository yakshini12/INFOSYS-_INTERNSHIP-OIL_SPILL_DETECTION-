# 🛢️ Oil Spill Detection Using U-Net

## 📘 Overview

This project focuses on detecting and segmenting oil spill regions in satellite or aerial images using a **U-Net semantic segmentation model with a pretrained ResNet-50 encoder**.

Developed using PyTorch in Google Colab, the implementation covers data preprocessing, augmentation, model training, validation, checkpoint selection, inference, and visualization of segmentation metrics such as IoU and Dice Coefficient.

> **Architecture:** U-Net
> **Encoder / Backbone:** ResNet-50
> **Task:** Binary semantic segmentation

U-Net is the segmentation architecture, while ResNet-50 is used as its encoder/backbone. The encoder extracts image features, while the U-Net decoder reconstructs the pixel-level segmentation map.

---

## 🧩 Features

* Automated pixel-level segmentation of oil spill regions
* U-Net architecture with pretrained ResNet-50 encoder
* End-to-end training and validation pipeline
* Image and mask preprocessing with synchronized transformations
* Data augmentation for training
* Dice Loss + Binary Cross-Entropy loss
* AdamW optimizer with cosine annealing learning-rate scheduling
* Mixed-precision training
* Gradient accumulation
* Visualization of Loss, IoU, and Dice Score
* Best-model checkpoint selection
* Confidence heatmap generation during inference
* Google Colab compatible
* Streamlit prototype for interactive inference

---

## 📂 Project Structure

```text
oil_spill_detection/
│
├── oil_spill_detection.py       # Main PyTorch implementation
├── oil_spill_detection.ipynb    # Main Jupyter/Colab notebook
├── oil_spill_detection1.ipynb
├── oil_spill_detection (1).ipynb
├── oil_spill_detection (2).ipynb
├── oil_spill_detection (3).ipynb
├── README.md                    # Project documentation
```

The training script expects the dataset separately in Google Drive.

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yakshini12/INFOSYS-_INTERNSHIP-OIL_SPILL_DETECTION-.git
cd INFOSYS-_INTERNSHIP-OIL_SPILL_DETECTION-
```

### 2. Install Dependencies

Make sure Python 3.8+ is installed.

```bash
pip install torch torchvision torchaudio
pip install numpy matplotlib opencv-python pillow tqdm
pip install segmentation-models-pytorch albumentations
```

On Google Colab, several dependencies may already be available.

---

## 📁 Dataset

The training implementation expects the dataset at:

```text
/content/drive/MyDrive/oil_spill_dataset/
```

Expected structure:

```text
oil_spill_dataset/
├── train/
│   ├── images/
│   └── masks/
└── val/
    ├── images/
    └── masks/
```

Images and masks are paired by filename. Masks are converted to binary targets during dataset loading.

All images are resized to **256 × 256** pixels.

---

## 🧠 Model Architecture

The implementation uses **U-Net with a pretrained ResNet-50 encoder**:

```python
model = smp.Unet(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation=None
).to(DEVICE)
```

This means:

* **U-Net** is the main semantic segmentation architecture.
* **ResNet-50** is the encoder/backbone used to extract image features.
* The decoder reconstructs the pixel-level segmentation map.
* The output contains one channel for binary oil-spill segmentation.

---

## 🔄 End-to-End Workflow

```text
Satellite / Aerial Image
          │
          ▼
Resize + Normalize
          │
          ▼
Training Augmentation
          │
          ▼
U-Net
┌──────────────────────────┐
│ ResNet-50 Encoder        │
│        ↓                 │
│ Feature Extraction       │
│        ↓                 │
│ U-Net Decoder            │
└──────────────────────────┘
          │
          ▼
Binary Segmentation Logits
          │
          ▼
Sigmoid + Threshold
          │
          ▼
Predicted Oil-Spill Mask
          │
          ├──► IoU
          ├──► Dice
          ├──► Pixel Accuracy
          └──► Confidence Visualization
```

---

## 🧪 Data Augmentation

Training images use several transformations:

* Horizontal flip
* Vertical flip
* Random 90-degree rotation
* Shift, scale, and rotation
* Random brightness and contrast
* Random blur variants
* ImageNet normalization

Validation data uses resizing and normalization without the training-only random augmentations.

---

## 🏋️ Training Configuration

| Configuration         | Value             |
| --------------------- | ----------------- |
| Image Size            | 256 × 256         |
| Batch Size            | 8                 |
| Epochs                | 30                |
| Gradient Accumulation | 2 steps           |
| Optimizer             | AdamW             |
| Learning Rate         | 3e-4              |
| Weight Decay          | 1e-4              |
| Scheduler             | CosineAnnealingLR |
| Minimum Learning Rate | 1e-6              |
| Mixed Precision       | Enabled           |

### Loss Function

The training loss combines Dice Loss and Binary Cross-Entropy:

```text
Total Loss = 0.5 × Dice Loss + 0.5 × BCEWithLogits Loss
```

### Best Model Selection

The implementation selects the best checkpoint using a combined validation score:

```text
Validation Score = 0.7 × Validation Dice
                 + 0.3 × Validation IoU
```

The highest-scoring checkpoint is saved for subsequent inference.

---

## 📊 Evaluation Metrics

| Metric           | Description                                                              |
| ---------------- | ------------------------------------------------------------------------ |
| Accuracy         | Measures the proportion of correctly classified pixels                   |
| IoU              | Measures overlap between predicted and ground-truth segmentation regions |
| Dice Coefficient | Measures segmentation overlap between prediction and ground truth        |
| Loss             | Measures model prediction error during training and validation           |

The inference pipeline applies sigmoid activation and a threshold to convert model logits into the final binary mask.

---

## 📈 Results

The current project documentation reports the following validation results:

| Metric           | Score |
| ---------------- | ----: |
| Accuracy         |   91% |
| IoU Score        | ~0.87 |
| Dice Coefficient | ~0.89 |

These values are reported from the documented project run.

---

## 🔍 Inference & Visualization

After training, the best checkpoint is loaded for inference.

The implementation can generate:

* Original input image
* Predicted segmentation mask
* Confidence heatmap
* Estimated oil-spill pixel percentage

A Streamlit prototype is also included in the implementation for interactive image upload and prediction.

---

## 🚀 How to Run

### Google Colab

Mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Set the dataset path and run the notebook cells.

For the Python implementation, install the dependencies listed above and execute the training/inference workflow after configuring the dataset path.

---

## 🛠️ Tech Stack

**Languages & Frameworks**

* Python
* PyTorch

**Deep Learning**

* U-Net
* ResNet-50 encoder
* Segmentation Models PyTorch

**Data & Image Processing**

* NumPy
* OpenCV
* Pillow
* Albumentations

**Visualization**

* Matplotlib

**Deployment Prototype**

* Streamlit
* Google Colab
* Google Drive

---

## 🧭 Future Work

* Validate the model on more geographically diverse and unseen imagery
* Improve threshold calibration and model confidence estimation
* Add experiment tracking and reproducibility support
* Expose inference through a production API
* Integrate additional satellite data sources for broader monitoring

---

## 👩‍💻 Author

**Patila Yakshini**

B.Tech — Computer Science and Engineering (AI & ML)
CMR Engineering College, Telangana

🔗 [LinkedIn](https://www.linkedin.com/in/patila-yakshini/)
📧 [Email](mailto:patilayakshinipatila@gmail.com)

---

## 🪶 License

This project is licensed under the MIT License.

Feel free to fork, modify, and use it for research or educational purposes.
