import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
import gdown
import os

# ---------------- DOWNLOAD FILES ----------------

MODEL_URL = "https://drive.google.com/uc?id=1pflxySlxBXUmLVOorwLU93MzJVHM16ub"
CLASS_URL = "https://drive.google.com/uc?id=1w5zDqF4D6cgdkQnkalcYXRQwdI2JZNDk"

# Download model if not exists
if not os.path.exists("best_model.keras"):
    gdown.download(MODEL_URL, "best_model.keras", quiet=False)

# Download class names if not exists
if not os.path.exists("class_names.txt"):
    gdown.download(CLASS_URL, "class_names.txt", quiet=False)

# ---------------- LOAD MODEL ----------------

model = tf.keras.models.load_model("best_model.keras")

# Load class names
with open("class_names.txt") as f:
    class_names = [line.strip() for line in f]

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Tomato Leaf Disease Detection",
    page_icon="🍅",
    layout="centered"
)

# ---------------- UI ----------------

st.markdown(
    """
    <h1 style='text-align:center; color:green;'>
    🍅 Tomato Leaf Disease Detection
    </h1>
    """,
    unsafe_allow_html=True
)

st.write("Upload a tomato leaf image to detect possible diseases using Deep Learning.")

# Upload image
uploaded_file = st.file_uploader(
    "Upload Tomato Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Resize image
    image = image.resize((224, 224))

    # Convert to array
    img_array = np.array(image)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocess
    img_array = preprocess_input(img_array)

    # Prediction
    preds = model.predict(img_array)[0]

    # Confidence smoothing
    preds = preds ** 0.7
    preds = preds / np.sum(preds)

    # Top prediction
    pred_index = np.argmax(preds)

    # Prediction label
    prediction = class_names[pred_index]

    # Confidence
    confidence = preds[pred_index] * 100

    st.success(f"Prediction: {prediction}")

    st.info(f"Confidence: {confidence:.2f}%")
