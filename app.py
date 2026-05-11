import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
import gdown
import os

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Tomato Leaf Disease Detection",
    page_icon="🍅",
    layout="centered"
)

# ---------------- DOWNLOAD FILES ----------------

MODEL_URL = "https://drive.google.com/uc?id=1pflxySlxBXUmLVOorwLU93MzJVHM16ub"
CLASS_URL = "https://drive.google.com/uc?id=1w5zDqF4D6cgdkQnkalcYXRQwdI2JZNDk"

# Download model
if not os.path.exists("best_model.keras"):
    gdown.download(MODEL_URL, "best_model.keras", quiet=False)

# Download class names
if not os.path.exists("class_names.txt"):
    gdown.download(CLASS_URL, "class_names.txt", quiet=False)

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_my_model():
    model = keras.models.load_model(
        "best_model.keras",
        compile=False
    )
    return model

model = load_my_model()

# ---------------- LOAD CLASS NAMES ----------------

with open("class_names.txt") as f:
    class_names = [line.strip() for line in f]

# ---------------- UI ----------------

st.markdown(
    """
    <h1 style='text-align:center; color:green;'>
    🍅 Tomato Leaf Disease Detection
    </h1>
    """,
    unsafe_allow_html=True
)

st.write("Upload a tomato leaf image to detect diseases using Deep Learning.")

# ---------------- IMAGE UPLOAD ----------------

uploaded_file = st.file_uploader(
    "Upload Tomato Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    try:

        # Open image
        image = Image.open(uploaded_file).convert("RGB")

        # Show image
        st.image(image, caption="Uploaded Image", width=300)

        # Resize image
        image = image.resize((224, 224))

        # Convert to array
        img_array = np.array(image)

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # Preprocess
        img_array = preprocess_input(img_array)

        # Prediction
        preds = model.predict(img_array)

        # Get highest prediction
        pred_index = np.argmax(preds[0])

        # Prediction label
        prediction = class_names[pred_index]

        # Confidence
        confidence = float(np.max(preds[0])) * 100

        # Result
        st.success(f"Prediction: {prediction}")
        st.info(f"Confidence: {confidence:.2f}%")

    except Exception as e:
        st.error(f"Error: {e}")
