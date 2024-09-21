import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the saved model with error handling
try:
    model = tf.keras.models.load_model('emotion_classification_model.h5')
    st.write("Model loaded successfully.")
except Exception as e:
    st.write(f"Error loading model: {e}")

# Define emotion labels (in the same order as your model's output classes)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Streamlit app UI
st.title("Image Emotion Detection")

# File uploader to upload an image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Preprocess the image and make a prediction
    try:
        img_array = np.array(image.resize((48, 48)))  # Resize to (48x48) or whatever your model needs
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize image
        st.write(f"Image shape for model: {img_array.shape}")  # Check image shape
    except Exception as e:
        st.write(f"Error in preprocessing: {e}")
    
    # Make prediction
    try:
        prediction = model.predict(img_array)
        # Get the index of the highest probability
        predicted_index = np.argmax(prediction)
        # Get the corresponding emotion label
        predicted_emotion = emotion_labels[predicted_index]
        st.write(f"Image shape for model: {img_array.shape}")
        st.write(f"Prediction: {prediction}")
        st.write(f"Predicted index: {np.argmax(prediction)}")
         # Display the predicted emotion
        st.write(f"Predicted emotion: {predicted_emotion}")

    except Exception as e:
        st.write(f"Error during prediction: {e}")