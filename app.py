import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Tomato Leaf Disease Detection",
    page_icon="🍅",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "best_model.h5",
        compile=False
    )
    return model

model = load_model()

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

st.write("Upload a tomato leaf image to detect possible diseases using Deep Learning.")

# ---------------- FILE UPLOAD ----------------

uploaded_file = st.file_uploader(
    "Upload Tomato Leaf Image",
    type=["jpg", "jpeg", "png"]
)

# ---------------- PREDICTION ----------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Resize image
    image = image.resize((224, 224))

    # Convert to numpy array
    img_array = np.array(image)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocess image
    img_array = preprocess_input(img_array)

    # Predict
    preds = model.predict(img_array)[0]

    # Get top prediction
    pred_index = np.argmax(preds)

    prediction = class_names[pred_index]

    confidence = preds[pred_index] * 100

    # Show result
    st.success(f"Prediction: {prediction}")

    st.info(f"Confidence: {confidence:.2f}%")
