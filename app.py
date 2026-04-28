import streamlit as st
import numpy as np
from PIL import Image
import requests

# --------- Load class names ---------
def load_classes():
    url = "https://drive.google.com/uc?id=1tTb6mwl8B0WA701qRN0FZ4kzK-_F5eOZ"
    r = requests.get(url)
    classes = r.text.splitlines()
    return classes

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

    # TEMP OUTPUT (since TensorFlow removed)
    st.success("Image uploaded successfully ✅")
    st.subheader("🔍 Prediction")

    st.write("⚠️ Model temporarily disabled due to deployment issue.")
    st.write("Showing sample output 👇")

    # Fake predictions (demo purpose)
    sample_results = [
        ("Healthy", 92.5),
        ("Leaf Mold", 5.2),
        ("Early Blight", 2.3)
    ]

    for name, prob in sample_results:
        st.write(f"{name} : {prob}%")
