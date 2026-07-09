"""
Steel Surface Defect Classifier — Streamlit App
------------------------------------------------
Upload a hot-rolled steel surface image and the app will:
  1. Predict which of the 6 NEU defect classes it belongs to
  2. Show a confidence bar chart for all classes
  3. Show a Grad-CAM heatmap explaining WHERE the model is looking

Run with:  streamlit run app.py
Make sure 'steel_defect_cnn.pth' is in the same folder as this file.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Steel Surface Defect Classifier",
    page_icon="🔩",
    layout="wide",
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "steel_defect_cnn.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_DESCRIPTIONS = {
    "crazing": "Fine, irregular network of surface cracks caused by thermal/mechanical stress.",
    "inclusion": "Foreign, non-metallic particles embedded in the steel surface during production.",
    "patches": "Irregular, blotchy regions of uneven surface texture or coating.",
    "pitted_surface": "Small, localized pits or cavities on the steel surface.",
    "rolled-in_scale": "Oxide scale pressed into the surface during hot rolling.",
    "scratches": "Linear surface marks caused by mechanical abrasion during handling.",
}


# ----------------------------------------------------------------------
# MODEL DEFINITION  (must match the architecture used during training)
# ----------------------------------------------------------------------
class DefectCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=False),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=False),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=False),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256), nn.ReLU(inplace=False), nn.Dropout(0.4),
            nn.Linear(256, 64), nn.ReLU(inplace=False), nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class GradCAM:
    """Minimal Grad-CAM hooked onto the last convolutional layer."""

    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor, class_idx):
        self.model.eval()
        output = self.model(input_tensor)
        self.model.zero_grad()
        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1)).squeeze()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy(), output


# ----------------------------------------------------------------------
# CACHED MODEL LOADING
# ----------------------------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None, None

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    class_names = checkpoint["class_names"]
    img_size = checkpoint["img_size"]

    model = DefectCNN(num_classes=len(class_names)).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    gradcam = GradCAM(model, model.features[-2])  # last Conv2d before final pool
    return model, gradcam, {"class_names": class_names, "img_size": img_size,
                             "test_accuracy": checkpoint.get("test_accuracy")}


def preprocess_image(pil_img, img_size):
    gray = pil_img.convert("L").resize((img_size, img_size))
    arr = np.array(gray, dtype=np.float32)
    norm = (arr / 127.5) - 1.0
    tensor = torch.tensor(norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor, arr


# ----------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🔩 About")
    st.write(
        "This app uses a Convolutional Neural Network trained on the "
        "**NEU Surface Defect Database** to classify hot-rolled steel "
        "surface images into 6 defect categories."
    )
    st.markdown("### Defect classes")
    for cname, desc in CLASS_DESCRIPTIONS.items():
        st.markdown(f"**{cname.replace('_', ' ').title()}**  \n{desc}")
    st.markdown("---")
    st.caption("Model architecture: 4-block CNN (Conv+BatchNorm+ReLU+MaxPool) trained with PyTorch.")


# ----------------------------------------------------------------------
# MAIN PAGE
# ----------------------------------------------------------------------
st.title("🔩 Steel Surface Defect Classifier")
st.write(
    "Upload one or more steel surface images below. The model will predict "
    "the defect type and show you a **Grad-CAM heatmap** highlighting the "
    "region it focused on to make that decision."
)

model, gradcam, meta = load_model()

if model is None:
    st.error(
        f"Could not find the trained model file at `{MODEL_PATH}`.\n\n"
        "Please make sure **steel_defect_cnn.pth** is placed in the same "
        "folder as this app.py file, then restart the app."
    )
    st.stop()

class_names = meta["class_names"]
img_size = meta["img_size"]

if meta.get("test_accuracy"):
    st.success(f"Model loaded ✅ — trained test accuracy: **{meta['test_accuracy']*100:.2f}%**")

uploaded_files = st.file_uploader(
    "Upload steel surface image(s)",
    type=["jpg", "jpeg", "png", "bmp"],
    accept_multiple_files=True,
)

confidence_threshold = st.slider(
    "Flag predictions below this confidence as 'uncertain'", 0, 100, 60
) / 100.0

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.markdown("---")
        pil_img = Image.open(uploaded_file)

        input_tensor, gray_arr = preprocess_image(pil_img, img_size)
        input_tensor = input_tensor.to(DEVICE)

        with torch.no_grad():
            logits = model(input_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        pred_class = class_names[pred_idx]
        pred_conf = probs[pred_idx]

        # Grad-CAM needs gradients, so run it separately (not under no_grad)
        cam, _ = gradcam.generate(input_tensor, pred_idx)
        cam_resized = cv2.resize(cam, (img_size, img_size))

        col1, col2, col3 = st.columns([1, 1, 1.2])

        with col1:
            st.subheader("📷 Uploaded Image")
            st.image(pil_img, use_container_width=True)

        with col2:
            st.subheader("🔥 Grad-CAM Heatmap")
            fig, ax = plt.subplots(figsize=(3.5, 3.5))
            ax.imshow(gray_arr, cmap="gray")
            ax.imshow(cam_resized, cmap="jet", alpha=0.45)
            ax.axis("off")
            st.pyplot(fig)
            plt.close(fig)

        with col3:
            st.subheader("📊 Prediction")
            if pred_conf < confidence_threshold:
                st.warning(
                    f"**{pred_class.replace('_', ' ').title()}** "
                    f"(confidence {pred_conf*100:.1f}% — below threshold, review manually)"
                )
            else:
                st.markdown(f"### ✅ {pred_class.replace('_', ' ').title()}")
                st.caption(CLASS_DESCRIPTIONS.get(pred_class, ""))

            prob_df = pd.DataFrame({
                "Defect class": [c.replace("_", " ").title() for c in class_names],
                "Confidence": probs,
            }).sort_values("Confidence", ascending=True)

            fig2, ax2 = plt.subplots(figsize=(4, 2.6))
            bars = ax2.barh(prob_df["Defect class"], prob_df["Confidence"], color="#2b6cb0")
            ax2.set_xlim(0, 1)
            ax2.set_xlabel("Confidence")
            for bar, val in zip(bars, prob_df["Confidence"]):
                ax2.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                          f"{val*100:.1f}%", va="center", fontsize=8)
            fig2.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

        with st.expander("Raw probability values"):
            st.dataframe(
                prob_df.sort_values("Confidence", ascending=False).reset_index(drop=True),
                use_container_width=True,
            )
else:
    st.info("👆 Upload at least one image to get started. You can also drag & drop, or upload multiple images at once.")
