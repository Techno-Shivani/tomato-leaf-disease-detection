import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

MODEL_PATH = "best_model.keras"

# Download model from Google Drive
if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=1pflxySlxBXUmLVOorwLU93MzJVHM16ub"
    gdown.download(url, MODEL_PATH, quiet=False)

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Class names
with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

st.title("🍅 Tomato Leaf Disease Detection")

uploaded_file = st.file_uploader(
    "Upload Tomato Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")
