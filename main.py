import streamlit as st
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import backend as K

import numpy as np
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import pickle
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
import faiss

class L2NormLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)

st.title("Fashion Recommendation System")

feature_list = np.array(pickle.load(open('features.pkl','rb')))
filenames = pickle.load(open('filename.pkl','rb'))
models = tf.keras.models.load_model('model.keras', custom_objects={'L2NormLayer': L2NormLayer})




## save the uploaded file
def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join("uploads", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0


## extract fearure from image we uopload
def feature_extraction(img_path,model):
    img = image.load_img(img_path,target_size=(300,300))
    img_array = image.img_to_array(img)
    expand_img_array = np.expand_dims(img_array,axis=0)
    preprocessed_input = preprocess_input(expand_img_array)
    result = model.predict(preprocessed_input).flatten()
    normalized_result = result/norm(result)
    return normalized_result
             
             
def recommendation(features,feature_list): 
    # Ensure correct data types
    feature_list = feature_list.astype(np.float32)
    features = np.array([features], dtype=np.float32)

    # Create FAISS index and perform search
    index = faiss.IndexFlatL2(features.shape[1])  # d = 1536 for EfficientNetB3
    index.add(feature_list)

    distances, indices = index.search(features, k=5)
    return distances,indices      

       
## upload file
uploaded_file = st.file_uploader("Choose a file", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        
        ## display the image with streamlit
        disp_img = Image.open(uploaded_file)
        st.image(uploaded_file)
        
        features = feature_extraction(os.path.join("uploads", uploaded_file.name),models)
        st.text(features.shape)
        
        distances,indices = recommendation(features,feature_list)
        col1,col2,col3,col4,col5 = st.columns(5)
        
        with col1:
            st.image(filenames[indices[0][0]])
            similarity = 1 / (1+distances[0][0])
            percentage = similarity*100
            st.caption(f"Distance: {distances[0][0]:.4f}")
            st.caption(percentage)
            # st.caption(f"Distance: {distances:.4f}\n\nSimilarity: {percentage:.2f}%")
        with col2:
            st.image(filenames[indices[0][1]])
            st.caption(f"Distance: {distances[0][1]:.4f}")
        with col3:
            st.image(filenames[indices[0][2]])
            st.caption(f"Distance: {distances[0][2]:.4f}")
        with col4:
            st.image(filenames[indices[0][3]])
            st.caption(f"Distance: {distances[0][3]:.4f}")
        with col5:
            st.image(filenames[indices[0][4]])
            st.caption(f"Distance: {distances[0][4]:.4f}")
         
    else:
        st.warning("Could not save file")