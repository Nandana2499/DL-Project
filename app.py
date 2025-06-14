import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load trained model (using cache for efficiency)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("plant_leaf_disease_model.h5")
    return model

model = load_model()

# Class labels (replace with your actual labels from training)
CLASS_LABELS = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'PlantVillage', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']
# Prediction function (corrected preprocessing)
def preprocess_and_predict(image, model):
    # Convert PIL Image to OpenCV format
    img = np.array(image)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # Convert RGB to BGR
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_idx = np.argmax(prediction)

    return class_idx, prediction[0][class_idx]  # Return class index and confidence


# Streamlit UI
st.title("🌿 Plant Leaf Disease Detection")
st.markdown("Upload a plant leaf image to predict its disease.")

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')  # Ensure RGB
    st.image(image, caption="Uploaded Leaf", use_column_width=True)

    st.write("Predicting...")
    class_idx, confidence = preprocess_and_predict(image, model)
    predicted_class_name = CLASS_LABELS[class_idx]

    st.markdown(f"### Prediction: {predicted_class_name} (Confidence: {confidence:.4f})")