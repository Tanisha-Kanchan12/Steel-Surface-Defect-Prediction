## 🌐 Live Demo

Here is the link of the Streamlit application: **https://steel-surface-defect-prediction.streamlit.app/**

Upload a steel surface image and get an instant defect prediction with a Grad-CAM heatmap — no installation needed.
---

# 🔩 Steel Surface Defect Classification

A deep learning project that classifies hot-rolled steel surface defects into 6 categories using a Convolutional Neural Network (CNN), built with PyTorch. Includes a Jupyter Notebook for training/analysis and a Streamlit web app for real-time predictions with Grad-CAM explainability.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CNN-red)
![Streamlit](https://img.shields.io/badge/Streamlit-App-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

Quality control in steel manufacturing traditionally relies on manual visual inspection, which is slow and error-prone. This project automates that process using a CNN trained on the **NEU Surface Defect Database**, achieving **98.5% test accuracy** across six defect types:

| Class | Description |
|---|---|
| **Crazing** | Fine, irregular network of surface cracks |
| **Inclusion** | Foreign non-metallic particles embedded in the surface |
| **Patches** | Irregular, blotchy regions of uneven texture |
| **Pitted Surface** | Small, localized pits or cavities |
| **Rolled-in Scale** | Oxide scale pressed into the surface during hot rolling |
| **Scratches** | Linear marks from mechanical abrasion |


## 📂 Project Structure

```
steel-defect-classification/
├── Steel_Surface_Defect_Classification.ipynb   # Full training & analysis notebook
├── steel_defect_cnn.pth                        # Trained model weights
├── app.py                                       # Streamlit web app
├── requirements.txt                             # Python dependencies
└── README.md
```

---

## 🧠 Model Architecture

A custom 4-block CNN:

Conv2d → BatchNorm → ReLU → MaxPool   (×4 blocks: 16→32→64→128 filters)
        ↓
Flatten → FC(256) → Dropout → FC(64) → Dropout → FC(6)

- **Input:** 128×128 grayscale images
- **Optimizer:** Adam (lr=1e-3) with ReduceLROnPlateau scheduler
- **Loss:** Cross-Entropy
- **Augmentation:** 90°- multiple rotation, horizontal/vertical flips, brightness jitter

## 📊 Results

| Metric | Score |
|---|---|
| Test Accuracy | **98.50%** |
| Macro F1-Score | **0.985** |

The notebook includes:
- 📈 Exploratory Data Analysis (class distribution, sample images, pixel intensity plots)
- 📉 Training/validation loss & accuracy curves
- 🔲 Confusion matrix heatmap (raw + normalized)
- 🔥 Grad-CAM heatmaps for model interpretability
- ❌ Misclassification analysis

## 🚀 Getting Started

### 1. Clone the repository
git clone https://github.com/<your-username>/steel-defect-classification.git
cd steel-defect-classification

### 2. Install dependencies
pip install -r requirements.txt

### 3a. Run the Jupyter Notebook
jupyter notebook Steel_Surface_Defect_Classification.ipynb

Place the NEU dataset zip/folder in the same directory  the notebook auto-detects and extracts it.

### 3b. Run the Streamlit App
streamlit run app.py

Then open **http://localhost:8501** in your browser, upload a steel surface image, and view:
- Predicted defect class + confidence scores
- Grad-CAM heatmap showing where the model focused

## 🗂️ Dataset

**NEU Surface Defect Database**
- 1,800 grayscale images (200×200 px), 300 samples per class
- Source: Northeastern University Surface Defect Database — http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html

## 🔮 Future Work

- Transfer learning with pretrained backbones (ResNet / VGG)
- Siamese network for one-shot recognition of new/rare defect types
- Defect localization/segmentation instead of whole-image classification
- Deployment as a REST API for integration with factory-line cameras

## 🙏 Acknowledgements

Inspired by:
Deshpande, A. M., Minai, A. A., & Kumar, M. (2020). One-Shot Recognition of Manufacturing Defects in Steel Surfaces. Procedia Manufacturing, NAMRC 48.
