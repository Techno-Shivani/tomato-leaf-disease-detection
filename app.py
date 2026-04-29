import streamlit as st
import numpy as np
from PIL import Image
import requests
import base64
import matplotlib.pyplot as plt
import tensorflow as tf

# ---------------- PAGE CONFIG ----------------
st.set_page_config(layout="wide", page_title="Tomato Leaf Disease Detection")

# ---------------- BACKGROUND ----------------
def get_base64(file_path):
    try:
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return ""

bg = get_base64("assets/bg-tomato-leaf.jpg")

st.markdown(f"""
<style>
.stApp {{
    background-image: url("data:image/jpg;base64,{bg}");
    background-size: cover;
    background-position: center;
}}

.main-container {{
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(15px);
    border-radius: 20px;
    padding: 30px;
    margin-top: 20px;
}}

.result-box {{
    background: rgba(255,255,255,0.3);
    padding: 20px;
    border-radius: 15px;
}}

.footer {{
    text-align:center;
    margin-top:40px;
    font-size:14px;
    color:white;
}}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model("model.h5")
    except:
        return None

model = load_model()

# ---------------- LOAD CLASSES ----------------
def load_classes():
    try:
        url = "https://drive.google.com/uc?id=1tTb6mwl8B0WA701qRN0FZ4kzK-_F5eOZ"
        r = requests.get(url)
        return r.text.splitlines()
    except:
        return [
            "Healthy",
            "Leaf Mold",
            "Early Blight"
        ]

class_names = load_classes()

# ---------------- MAIN UI ----------------
st.markdown('<div class="main-container">', unsafe_allow_html=True)

st.title("🌿 Tomato Leaf Disease Detection")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    # ---------- PREPROCESS ----------
    img = image.resize((224,224))
    arr = np.array(img)/255.0
    arr = np.expand_dims(arr, axis=0)

    # ---------- PREDICTION ----------
    if model:
        preds = model.predict(arr)[0]
        label = class_names[np.argmax(preds)]
        confidence = np.max(preds)
    else:
        # DEMO MODE
        label = "Healthy"
        confidence = 0.925

        preds = np.array([0.925, 0.052, 0.023])
        class_names = ["Healthy", "Leaf Mold", "Early Blight"]

    # ---------- LAYOUT ----------
    col1, col2 = st.columns([1,1])

    # -------- LEFT (IMAGE) --------
    with col1:
        st.subheader("📸 Uploaded Image")
        st.image(image, use_container_width=True)

    # -------- RIGHT (RESULT) --------
    with col2:
        st.subheader("🧠 Prediction Result")

        st.markdown('<div class="result-box">', unsafe_allow_html=True)

        st.success(f"{label}")
        st.progress(int(confidence*100))
        st.write(f"Confidence: {confidence*100:.2f}%")

        if not model:
            st.warning("⚠️ Demo mode (Model not loaded)")

        st.markdown('</div>', unsafe_allow_html=True)

    # -------- GRAPH (FIXED SIZE) --------
    st.subheader("📊 Prediction Distribution")

    fig, ax = plt.subplots(figsize=(6,6))
    ax.pie(preds, labels=class_names, autopct='%1.1f%%')
    ax.axis('equal')

    st.pyplot(fig)

# ---------------- FOOTER ----------------
st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="footer">Developed by Shivani Chauhan © 2026</div>',
    unsafe_allow_html=True
)
