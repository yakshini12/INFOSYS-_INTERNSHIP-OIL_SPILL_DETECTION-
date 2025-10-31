🛢️ Oil Spill Detection Using Deep Learning
📘 Overview

This project focuses on detecting oil spills in satellite or aerial images using a Convolutional Neural Network (CNN) model.
The goal is to segment and classify oil spill regions from ocean surfaces, contributing to environmental monitoring and marine safety.

Developed using PyTorch in Google Colab, this notebook covers data preprocessing, training, evaluation, and visualization of performance metrics such as IoU and Dice Coefficient.

🧩 Features

Automated detection of oil spill regions from images

End-to-end training and validation pipeline

Visualization of Loss, IoU, and Dice Score

Achieved 91% accuracy on validation data

Fully compatible with Google Colab

📂 Project Structure
oil_spill_detection/
│
├── train/images_256/        # Training images (preprocessed)
├── train/masks_256/         # Training masks
├── val/images_256/          # Validation images
├── val/masks_256/           # Validation masks
├── oil_spill_detection.ipynb # Main Jupyter notebook
└── README.md                 # Project documentation

⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/<your-username>/oil_spill_detection.git
cd oil_spill_detection

2️⃣ Install Dependencies

Make sure you have Python 3.8+ installed.
Install required libraries using pip:

pip install torch torchvision torchaudio
pip install numpy matplotlib opencv-python pillow tqdm


💡 On Google Colab, most dependencies come pre-installed.

📁 Dataset

Path: /content/drive/MyDrive/oil_spill_dataset/

Dataset includes satellite images labeled as oil spill and non-oil spill.

All images are resized to 256×256 pixels for efficient training.

🧠 Model Architecture

The model is a CNN-based architecture designed for image classification and segmentation.
It includes:

Convolutional layers with ReLU activations

Max-pooling for dimensionality reduction

Fully connected layers for classification output

🚀 How to Run
On Google Colab:

Mount Google Drive:

from google.colab import drive
drive.mount('/content/drive')


Set base_path to your dataset directory.

Run all cells in oil_spill_detection.ipynb.

Training progress and metric plots will be displayed automatically.

📊 Evaluation Metrics
Metric	Description
Accuracy	Measures correct predictions out of total samples
IoU (Intersection over Union)	Measures overlap between prediction and ground truth
Dice Coefficient	Another overlap metric emphasizing smaller regions
Loss	Quantifies prediction error during training
📈 Results
Metric	Score
Accuracy	91%
IoU Score	~0.87
Dice Coefficient	~0.89

📉 The plots below (generated in the notebook) show steady improvement in model performance across epochs:

Loss decreases consistently

IoU and Dice scores increase over time

🧭 Future Work

Implement U-Net for pixel-wise segmentation

Add data augmentation for improved generalization

Integrate with real-time satellite APIs for live oil spill monitoring

👩‍💻 Author

Patila Yakshini
B.Tech – Computer Science and Engineering (AIML)
CMR Engineering College, Telangana

🔗 LinkedIn (www.linkedin.com/in/patila-yakshini)

📧 patilayakshini@gmail.com

🪶 License

This project is licensed under the MIT License.
Feel free to fork, modify, and use it for research or educational purposes.
