import os, io
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
import streamlit as st

import config, utils
from model import get_model, MODEL_REGISTRY
from gradcam import generate_gradcam

st.set_page_config(
    page_title="Brain Tumor Detector — BRISC2025",
    layout="wide",
)

st.title(" Brain Tumor Detector")
st.caption("BRISC2025 Dataset · Custom CNN · ResNet-50 · EfficientNet-B3 · Grad-CAM")

device = utils.get_device()

st.header("Settings")
model_choice = st.selectbox(
    "Select Model",
    options=list(MODEL_REGISTRY.keys()),
    format_func=lambda x: {
        'custom_cnn':   ' Custom CNN (from scratch)',
        'resnet50':     ' ResNet-50 (Transfer Learning)',
        'efficientnet': ' EfficientNet-B3 (Transfer Learning)',
        'mobilenetv3':  ' MobileNet-V3 (Transfer Learning)',
    }.get(x, x)
)
st.markdown("---")
st.markdown("**Classes:**")
for c in config.CLASS_NAMES:
    st.markdown(f"- {c}")
st.markdown("---")
st.info(f"Running on: **{device}**")

summary_path = os.path.join(config.OUTPUTS_DIR, 'results_summary.csv')
if os.path.exists(summary_path):
    st.markdown("**Model Comparison (Test Set)**")
    df = pd.read_csv(summary_path).drop_duplicates(subset='model', keep='last')
    st.dataframe(df, hide_index=True, use_container_width=True)

st.markdown("---")

@st.cache_resource(show_spinner="Loading model weights…")
def load_model(name: str):
    path = config.model_save_path(name)
    if not os.path.exists(path):
        return None, None
    ckpt  = torch.load(path, map_location=device)
    model = get_model(name, num_classes=config.NUM_CLASSES)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    val_acc = ckpt.get('val_acc', None)
    return model, val_acc

model, val_acc = load_model(model_choice)

if model is None:
    st.error(
        f"No trained checkpoint found for **{model_choice}**.\n\n"
        f"Train it first:\n```bash\npython train.py --model {model_choice}\n```"
    )
    st.stop()

st.success(f"Loaded **{model_choice}** — Val Accuracy: {val_acc:.2f}%")

col_img, col_pred = st.columns([1, 1])

with col_img:
    st.subheader(" Input MRI Scan")
    uploaded = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])
    image = None
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded MRI", use_container_width=True)

with col_pred:
    st.subheader(" Prediction & Grad-CAM")

    if image is not None:
        if st.button("Analyse MRI →", type="primary", use_container_width=True):
            mean = config.IMAGENET_MEAN
            std  = config.IMAGENET_STD

            img_arr    = np.array(image.resize((config.IMAGE_SIZE, config.IMAGE_SIZE))) / 255.0
            img_tensor = torch.FloatTensor(
                (img_arr - mean) / std
            ).permute(2, 0, 1).unsqueeze(0).to(device)

            with st.spinner("Analysing…"):
                heatmap, overlay, logits = generate_gradcam(model, img_tensor)
                probs      = F.softmax(logits, dim=1).cpu().detach().numpy()[0]
                pred_idx   = int(probs.argmax())
                pred_label = config.CLASS_NAMES[pred_idx]
                pred_prob  = probs[pred_idx]

            # Result banner
            if pred_label == 'no_tumor':
                st.success(f"**No Tumor Detected** ({pred_prob:.1%} confidence)")
            else:
                st.warning(f"**{pred_label.title()} Detected** ({pred_prob:.1%} confidence)")

            # Probability bar chart
            prob_df = pd.DataFrame({
                'Tumor Type': config.CLASS_NAMES,
                'Probability': probs,
            }).set_index('Tumor Type')
            st.bar_chart(prob_df)

            # Grad-CAM overlay
            st.image(overlay, caption="Grad-CAM — highlighted regions drove the prediction",
                     use_container_width=True)

            # Download button
            buf = io.BytesIO()
            overlay.save(buf, format="PNG")
            st.download_button(
                "⬇ Download Grad-CAM Overlay",
                data=buf.getvalue(),
                file_name=f"gradcam_{model_choice}.png",
                mime="image/png",
            )
    else:
        st.info("Upload an MRI image on the left to get started.")

st.markdown("---")
st.markdown(
    "**About Grad-CAM** — Gradient-weighted Class Activation Mapping highlights "
    "which regions of the MRI most influenced the model's decision.  "
    "🔴 Red/yellow = high importance.  🔵 Blue = low importance."
)
