import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load the saved model
model = load_model('CNN_model.h5')
print("Model loaded successfully!")

# Class names (same as your model's class labels)
class_names = ['Karacadag', 'Ipsala', 'Arborio', 'Basmati', 'Jasmine']

# Recreate and fit the encoder
encoder = LabelEncoder()
encoder.fit(class_names)

# Function to preprocess the image
def preprocess_image(img_path, target_size=(50, 50)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Rescale the image as done during training
    return img_array

# Function to make predictions
def predict_image(img_path):
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    predicted_class_idx = np.argmax(prediction, axis=1)[0]
    predicted_class = encoder.inverse_transform([predicted_class_idx])[0]
    return predicted_class, prediction[0][predicted_class_idx]

# Streamlit UI
st.title("Rice Variety Classifier")
st.write("Upload a rice image to predict its variety and confidence.")

# Image uploader widget
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption='Uploaded Image.', use_column_width=True)
    
    # Make prediction
    img_path = uploaded_image
    predicted_class, confidence = predict_image(img_path)

    # Display the result
    st.write(f"Predicted Class: **{predicted_class}**")
    st.write(f"Confidence: **{confidence * 100:.2f}%**")

# Run the app with the following command:
# streamlit run <your_script_name.py>
