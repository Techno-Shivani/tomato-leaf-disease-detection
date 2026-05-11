import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("best_model.h5", compile=False)
    return model

model = load_model()

# ---------------- LOAD CLASS NAMES ----------------

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

st.write("Upload a tomato leaf image to detect diseases using Deep Learning.")

# ---------------- FILE UPLOAD ----------------

uploaded_file = st.file_uploader(
    "Upload Tomato Leaf Image",
    type=["jpg", "jpeg", "png"]
)

# ---------------- PREDICTION ----------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", width=300)

    image = image.resize((224, 224))

    img_array = np.array(image)

    img_array = np.expand_dims(img_array, axis=0)

    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)

    pred_index = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    result = class_names[pred_index]

    st.success(f"Prediction: {result}")

    st.info(f"Confidence: {confidence:.2f}%")
