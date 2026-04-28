import streamlit as st
import numpy as np
# import tensorflow as tf
from PIL import Image
import requests

# --------- Download model from Google Drive ---------
@st.cache_resource
def load_model():
    url = "https://drive.google.com/uc?id=1xfvLMQcReQBzmYg4KVptdeYnxpXcdbKe"
    model_path = "model.keras"

    if not tf.io.gfile.exists(model_path):
        r = requests.get(url)
        with open(model_path, "wb") as f:
            f.write(r.content)
            

   # return tf.keras.models.load_model(model_path)


# --------- Load class names ---------
def load_classes():
    url = "https://drive.google.com/uc?id=1tTb6mwl8B0WA701qRN0FZ4kzK-_F5eOZ"
    r = requests.get(url)
    classes = r.text.splitlines()
    return classes


# model = load_model()
class_names = load_classes()


# --------- UI ---------
st.title("🌿 Tomato Leaf Disease Detection")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]

    # Confidence smoothing
    preds = preds ** 0.7
    preds = preds / np.sum(preds)

    # Top 3
    top_indices = preds.argsort()[-3:][::-1]

    st.subheader("🔍 Prediction")

    for i in top_indices:
        st.write(f"{class_names[i]} : {preds[i]*100:.2f}%")
