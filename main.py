import streamlit as st
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
from numpy.linalg import norm
import faiss
import pickle
import base64

class L2NormLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)

# Page Configuration
st.set_page_config(
    page_title="Fashion Recommendation System",
    page_icon="🧥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Theme Styling with Modern CSS
st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #1f1c2c, #928DAB);
            font-family: 'Segoe UI', sans-serif;
            color: #f0f0f0;
        }
        .main {
            background-color: rgba(255, 255, 255, 0.05);
            padding: 2rem;
            border-radius: 16px;
        }
        .stButton>button {
            background-color: #e91e63;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            font-size: 16px;
            padding: 10px 24px;
        }
        .stFileUploader, .stImage {
            background-color: #2c2c54;
            padding: 15px;
            border-radius: 16px;
            margin-bottom: 20px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
        }
        h1, h2, h3, h4, h5, h6, p {
            color: #ffffff;
            text-align: center;
        }
        .custom-header {
            background: rgba(255, 255, 255, 0.1);
            padding: 1.5rem;
            border-radius: 16px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
        }
        .css-1v0mbdj, .css-1v0mbdj p {
            color: #ffffff;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class='custom-header'>
        <h1>🧥 Fashion Recommendation System</h1>
        <p>Upload a fashion item and discover visually similar recommendations</p>
    </div>
""", unsafe_allow_html=True)

# Load model and data
@st.cache_resource(show_spinner=False)
def load_artifacts():
    features = np.array(pickle.load(open("features.pkl", "rb"))).astype(np.float32)
    files = pickle.load(open("filename.pkl", "rb"))
    model = tf.keras.models.load_model("model.keras", custom_objects={"L2NormLayer": L2NormLayer})
    return features, files, model

feature_list, filenames, model = load_artifacts()

# Sidebar for image upload
with st.sidebar:
    st.header("📤 Upload Fashion Image")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# Helper functions
def save_file(uploaded):
    os.makedirs("uploads", exist_ok=True)
    path = os.path.join("uploads", uploaded.name)
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())
    return path

# Extract features from uploaded image
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(300, 300))
    img_arr = image.img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    img_arr = preprocess_input(img_arr)
    result = model.predict(img_arr).flatten()
    return result / norm(result)

# get similar images
def search_similar(query_feat, feature_list, k=5):
    query_feat = np.array([query_feat], dtype=np.float32)
    feature_list = feature_list.astype(np.float32)
    index = faiss.IndexFlatL2(query_feat.shape[1])
    index.add(feature_list)
    distances, indices = index.search(query_feat, k)
    return distances[0], indices[0]

# Display recommendations
if uploaded_file:
    file_path = save_file(uploaded_file)

    # Encode the uploaded image to base64 string
    encoded_img = base64.b64encode(uploaded_file.getvalue()).decode()

    st.markdown(f"""
        <div style='display: flex; justify-content: center; align-items: center; flex-direction: column;'>
            <p style='font-size: 20px; font-weight: 600; margin-bottom: 10px;'>Your Uploaded Image</p>
            <img src='data:image/jpeg;base64,{encoded_img}' width='450' height='400' 
                 style='border-radius: 12px; box-shadow: 0 4px 8px rgba(0,0,0,0.3); margin-bottom: 20px;'>
        </div>
    """, unsafe_allow_html=True)

    with st.spinner("Searching for similar styles..."):
        query_feat = extract_features(file_path)
        distances, indices = search_similar(query_feat,feature_list)

    st.subheader("🎯 Recommended Styles")
    rec_cols = st.columns(5)
    for i, idx in enumerate(indices):
        with rec_cols[i]:
            st.image(filenames[idx], width=300)
            sim_score = 1 / (1 + distances[i])
            st.caption(f"Similarity: {sim_score*100:.2f}%")
else:
    st.info("Upload an image from the sidebar to get recommendations.")