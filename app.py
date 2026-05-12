import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="Tomato Leaf Disease Detection")

MODEL_PATH = "best_model.keras"

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# Load class names
with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Title
st.title("🍅 Tomato Leaf Disease Detection")

st.write("Upload a tomato leaf image to detect disease.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction)]

    confidence = np.max(prediction) * 100

    st.success(f"Prediction: {predicted_class}")

    st.info(f"Confidence: {confidence:.2f}%")
