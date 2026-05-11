import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Tomato Leaf Disease Detection",
    page_icon="🍅",
    layout="centered"
)

# ---------------- TITLE ----------------

st.title("🍅 Tomato Leaf Disease Detection")

st.write(
    "Upload a tomato leaf image to detect possible diseases using Deep Learning."
)

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "best_model.keras",
        compile=False
    )
    return model

model = load_model()

# ---------------- LOAD CLASS NAMES ----------------

with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# ---------------- IMAGE PREPROCESS ----------------

IMG_SIZE = 224

def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ---------------- FILE UPLOAD ----------------

uploaded_file = st.file_uploader(
    "Upload Tomato Leaf Image",
    type=["jpg", "jpeg", "png"]
)

# ---------------- PREDICTION ----------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image)

    predicted_class = class_names[np.argmax(prediction)]

    confidence = np.max(prediction) * 100

    st.success(f"Prediction: {predicted_class}")

    st.info(f"Confidence: {confidence:.2f}%")
