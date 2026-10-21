import streamlit as st
import torch
from PIL import Image
import numpy as np
import pandas as pd

from model_utils import (
    load_model,
    load_ablation_model,
    get_transform,
    predict,
    generate_gradcam_overlay
)

# Page Configuration
st.set_page_config(
    page_title="Fruit Ripeness Classifier",
    page_icon="🍎",
    layout="wide"
)

# Constants — adjust if your checkpoint is named differently
MODEL_PATH          = "./models/best_model_v2.pth"
ABLATION_AUG_PATH   = "./models/ablation_with_aug_v2.pth"
ABLATION_NOAUG_PATH = "./models/ablation_no_aug_v2.pth"
IMG_SIZE   = 224
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Cache model loading so it only runs once per session
@st.cache_resource
def get_model_and_classes():
    model, class_names, idx_to_class = load_model(MODEL_PATH, DEVICE)
    return model, class_names, idx_to_class


@st.cache_resource
def get_ablation_model(checkpoint_path, num_classes):
    """Loads an ablation checkpoint on demand (only when user selects it)."""
    model, used_augmentation = load_ablation_model(checkpoint_path, DEVICE, num_classes)
    return model, used_augmentation


# Header
st.title("🍎 Fruit Ripeness & Defect Grading")
st.markdown(
    "Upload an image of a fruit to classify its **type** and **condition** "
    "(fresh / rotten / unripe), with an explainability heatmap showing "
    "exactly where the model focused its attention."
)
st.divider()

# Load model (cached — runs once)
try:
    model, class_names, idx_to_class = get_model_and_classes()
    st.sidebar.success(f"✓ Model loaded ({len(class_names)} classes)")
    st.sidebar.write("**Classes:**")
    for c in class_names:
        st.sidebar.write(f"  • {c}")
except FileNotFoundError:
    st.error(
        f"❌ Could not find `{MODEL_PATH}`. "
        f"Make sure you've downloaded `best_model.pth` from Google Drive "
        f"and placed it in the same folder as this app."
    )
    st.stop()

# Sidebar Settings
st.sidebar.divider()
st.sidebar.subheader("⚙️ Settings")
show_gradcam   = st.sidebar.checkbox("Show Grad-CAM heatmap", value=True)
overlay_alpha  = st.sidebar.slider("Heatmap intensity", 0.1, 0.9, 0.45, 0.05)
st.sidebar.caption(f"Device: `{DEVICE}`")

# Sidebar: Optional Ablation Comparison (research toggle, off by default)
st.sidebar.divider()
st.sidebar.subheader("🔬 Ablation Comparison (optional)")
st.sidebar.caption(
    "Compares the main model against ablation checkpoints "
    "(trained with/without data augmentation) from Chapter 4. "
    "For research demonstration only — the main model above "
    "is always the primary prediction."
)
compare_ablation = st.sidebar.checkbox("Enable ablation comparison", value=False)

ablation_choice = None
if compare_ablation:
    ablation_choice = st.sidebar.radio(
        "Compare main model against:",
        options=["With Augmentation", "Without Augmentation"],
        index=1   # default to "Without Augmentation" — the more interesting contrast
    )

# Image Upload
uploaded_file = st.file_uploader(
    "Choose a fruit image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    transform = get_transform(IMG_SIZE)

    # Layout: original image | results
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📷 Uploaded Image")
        st.image(image, use_column_width=True)

    # Run inference
    with st.spinner("Running inference..."):
        predicted_class, confidence, probs = predict(
            model, image, transform, class_names, DEVICE
        )

    with col2:
        st.subheader("🎯 Prediction")

        # Big prediction display
        st.markdown(
            f"### {predicted_class.upper()}"
        )
        st.metric(label="Confidence", value=f"{confidence*100:.2f}%")
        st.progress(min(confidence, 1.0))

        # Full probability distribution
        st.subheader("📊 Full Probability Distribution")
        prob_df = pd.DataFrame({
            "Class": class_names,
            "Probability": probs
        }).sort_values("Probability", ascending=False)

        st.bar_chart(prob_df.set_index("Class"))

        with st.expander("View raw probabilities"):
            for _, row in prob_df.iterrows():
                st.write(f"**{row['Class']}**: {row['Probability']*100:.2f}%")

    st.divider()

    # Optional Ablation Comparison
    if compare_ablation:
        st.subheader("🔬 Ablation Comparison")
        st.caption(
            f"Comparing the **main model** (trained with full pipeline) against "
            f"the ablation checkpoint trained **{ablation_choice}** — "
            f"same architecture, same hyperparameters, only this one variable changed."
        )

        ablation_path = (
            ABLATION_AUG_PATH if ablation_choice == "With Augmentation"
            else ABLATION_NOAUG_PATH
        )

        try:
            ablation_model, used_aug = get_ablation_model(ablation_path, len(class_names))

            with st.spinner("Running ablation model inference..."):
                ablation_pred, ablation_conf, ablation_probs = predict(
                    ablation_model, image, transform, class_names, DEVICE
                )

            comp_col1, comp_col2 = st.columns(2)
            with comp_col1:
                st.markdown("**Main Model (best_model.pth)**")
                st.write(f"Prediction: **{predicted_class}**")
                st.write(f"Confidence: **{confidence*100:.2f}%**")
            with comp_col2:
                st.markdown(f"**Ablation: {ablation_choice}**")
                st.write(f"Prediction: **{ablation_pred}**")
                st.write(f"Confidence: **{ablation_conf*100:.2f}%**")

            if predicted_class != ablation_pred:
                st.warning(
                    "⚠️ The two models disagree on this image — "
                    "a good talking point for the overfitting/ablation analysis."
                )
            else:
                st.info(
                    "Both models agree on the predicted class, "
                    "though confidence levels may differ."
                )

        except FileNotFoundError:
            st.warning(
                f"⚠️ Could not find `{ablation_path}`. "
                f"Download it from Drive and place it in this folder to enable comparison."
            )

        st.divider()

    # Grad-CAM Section
    if show_gradcam:
        st.subheader("🔍 Grad-CAM Explainability")
        st.caption(
            "This heatmap shows which pixels the model focused on most "
            "when making its decision. Warmer colors (red/yellow) indicate "
            "higher importance."
        )

        with st.spinner("Generating Grad-CAM heatmap..."):
            # Target layer: the last conv block BEFORE the SE attention module
            # model.features[-1][0] = original MobileNetV3 conv block
            # model.features[-1][1] = our injected SEBlock
            target_layer = model.features[-1][0]

            overlay_img, used_class_idx = generate_gradcam_overlay(
                model=model,
                image_pil=image,
                transform=transform,
                target_layer=target_layer,
                device=DEVICE,
                class_idx=None,
                alpha=overlay_alpha
            )

        col3, col4 = st.columns([1, 1])
        with col3:
            st.image(
                image.resize((224, 224)),
                caption="Original (resized)",
                use_column_width=True
            )
        with col4:
            st.image(
                overlay_img,
                caption=f"Grad-CAM for: {class_names[used_class_idx]}",
                use_column_width=True
            )

else:
    st.info("👆 Upload a fruit image above to get started.")

    with st.expander("ℹ️ About this app"):
        st.markdown(
            """
            This application uses a **MobileNetV3-Small** backbone with a
            manually-implemented **Squeeze-and-Excitation (SE) attention module**,
            fine-tuned on a fruit ripeness dataset containing fresh, rotten,
            and unripe samples of apples, bananas, and oranges.

            **Pipeline:**
            1. Image is resized to 224×224 and normalized using ImageNet statistics
            2. Forward pass through the trained CNN produces class logits
            3. Softmax converts logits to a probability distribution
            4. Grad-CAM backpropagates the predicted class score to visualize
               which spatial regions most influenced the decision
            """
        )
