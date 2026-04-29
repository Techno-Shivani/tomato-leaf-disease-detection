import streamlit as st
import numpy as np
from PIL import Image
import requests
import base64
import matplotlib.pyplot as plt

# ---------------- BACKGROUND + UI ----------------

def get_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

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

# ---------------- LOAD CLASSES ----------------

def load_classes():
    url = "https://drive.google.com/uc?id=1tTb6mwl8B0WA701qRN0FZ4kzK-_F5eOZ"
    r = requests.get(url)
    return r.text.splitlines()

class_names = load_classes()

# ---------------- MAIN UI ----------------

st.markdown('<div class="main-container">', unsafe_allow_html=True)

st.title("🌿 Tomato Leaf Disease Detection")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    # Layout
    col1, col2 = st.columns([1,1])

    # -------- LEFT (IMAGE) --------
    with col1:
        st.subheader("📸 Uploaded Image")
        st.image(image, use_column_width=True)

    # -------- RIGHT (RESULT) --------
    with col2:
        st.subheader("🧠 Prediction Result")

        st.markdown('<div class="result-box">', unsafe_allow_html=True)

        st.success("🌿 Healthy (Demo Result)")
        st.progress(0.92)

        st.write("⚠️ Model temporarily disabled (Demo mode)")

        sample_results = [
            ("Healthy", 92.5),
            ("Leaf Mold", 5.2),
            ("Early Blight", 2.3)
        ]

        labels = []
        values = []

        for name, prob in sample_results:
            st.write(f"{name} : {prob}%")
            labels.append(name)
            values.append(prob)

        st.markdown('</div>', unsafe_allow_html=True)

    # -------- GRAPH --------
    st.subheader("📊 Prediction Distribution")

    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, autopct='%1.1f%%')
    ax.axis('equal')

    st.pyplot(fig)

# ---------------- FOOTER ----------------

st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="footer">Developed by Shivani Chauhan © 2026</div>',
    unsafe_allow_html=True
)
