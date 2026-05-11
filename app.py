import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Tomato Leaf Disease Detection",
    layout="centered"
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

# ---------------- TITLE ----------------

st.markdown(
    """
    <h1 style='text-align: center;'>
    🍅 Tomato Leaf Disease Detection
    </h1>
    """,
    unsafe_allow_html=True
)

# ---------------- IMAGE UPLOAD ----------------

uploaded_file = st.file_uploader(
    "",
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

    # Resize image
    image = image.resize((224, 224))

    # Convert to array
    img_array = np.array(image)

    # Normalize
    img_array = img_array / 255.0

    # Expand dimensions
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction)]

    confidence = np.max(prediction) * 100

    # ---------------- RESULT ----------------

    st.markdown("## Prediction")

    st.success(f"{predicted_class}")

    st.info(f"Confidence: {confidence:.2f}%")
