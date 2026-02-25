import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Rice Leaf Disease Detection",
    page_icon="🌾",
    layout="centered"
)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("rice_leaf_model.h5")

model = load_model()

# -----------------------------
# Class Names
# -----------------------------
class_names = ["Leaf smut", "Brown spot", "Bacterial leaf blight"]

# -----------------------------
# Title & Description
# -----------------------------
st.title("🌾 Rice Leaf Disease Detection")
st.write(
    "Upload a rice leaf image to classify the disease using a trained CNN + MobileNetV2 model."
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("📌 About Project")
st.sidebar.write("**Model:** MobileNetV2 (Transfer Learning)")
st.sidebar.write("**Classes:**")
st.sidebar.write("- Leaf smut")
st.sidebar.write("- Brown spot")
st.sidebar.write("- Bacterial leaf blight")
st.sidebar.write("**Dataset:** 120 rice leaf images")

# -----------------------------
# File Uploader
# -----------------------------
uploaded_file = st.file_uploader(
    "📤 Upload Rice Leaf Image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# Prediction Function
# -----------------------------
def predict_disease(image):

    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0]

    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    return predicted_class, confidence, prediction

# -----------------------------
# Main Logic
# -----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    st.write("🔍 **Analyzing Image...**")

    predicted_class, confidence, probs = predict_disease(image)

    label = class_names[predicted_class]

    # -----------------------------
    # Prediction Output
    # -----------------------------
    st.success(f"✅ **Prediction:** {label}")
    st.info(f"📊 **Confidence:** {confidence:.2f}")

    # -----------------------------
    # Low Confidence Warning
    # -----------------------------
    if confidence < 0.60:
        st.warning("⚠️ Low confidence prediction. Try a clearer image.")

    # -----------------------------
    # Probability Breakdown
    # -----------------------------
    st.write("### 📈 Class Probabilities")

    for i, prob in enumerate(probs):
        st.write(f"**{class_names[i]}:** {prob:.2f}")
        st.progress(float(prob))

else:
    st.write("📎 Please upload an image to begin classification.")