import os
import pickle

import numpy as np
import streamlit as st
import tensorflow as tf
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import GlobalMaxPooling2D
from keras.preprocessing import image
from numpy.linalg import norm
from PIL import Image
from sklearn.neighbors import NearestNeighbors

with open('embedding.pkl', 'rb') as embedding_file:
    feature_list = pickle.load(embedding_file)
with open('filename.pkl', 'rb') as filename_file:
    filenames = pickle.load(filename_file)
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])
st.title('Fashion Recommender System')
def save_uploaded_file(uploaded_file):
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.read())  
        return 1
    except Exception as e:
        st.write(f"Error saving the uploaded file: {e}")
        return 0
def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result
def recommend(features, feature_list, k=5):
    neighbors = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        display_image = Image.open(os.path.join("uploads", uploaded_file.name))
        st.image(display_image)
        features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)
        indices = recommend(features, feature_list)
        cols = st.columns(5)  
        for i, col in enumerate(cols):
            with col:
                st.image(Image.open(filenames[indices[0][i]]))
else:
    st.header("Please upload an image")
