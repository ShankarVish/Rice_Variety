# 🌾 Rice Variety Classifier

A deep learning-based image classifier that identifies the variety of rice grains from images using a Convolutional Neural Network (CNN). The project is deployed as an interactive web app using Streamlit.

🔗 **Live Demo**: [https://ricevar.streamlit.app](https://ricevar.streamlit.app)

---

## 📌 Overview

- **Objective**: Classify rice grain images into 5 categories: Karacadag, Ipsala, Arborio, Basmati, Jasmine.
- **Approach**: Trained a CNN model on a labeled image dataset using TensorFlow/Keras.
- **Deployment**: Real-time inference with Streamlit.
- **Accuracy Achieved**: ~97.11% on the test set.

---

## 📂 Dataset Structure

```
Rice_Image_Dataset/
├── Karacadag/
├── Ipsala/
├── Arborio/
├── Basmati/
└── Jasmine/
```

- Each subfolder contains thousands of labeled rice grain images.
- Images resized to 50x50 pixels for model input.

---

## 🧠 Model Architecture

- Input: 50x50 RGB images
- Conv2D (32 filters, 3x3) + ReLU → MaxPooling (2x2)
- Conv2D (64 filters, 3x3) + ReLU → MaxPooling (2x2)
- Flatten → Dense(128, ReLU) → Dense(5, Softmax)

### Training Configuration

- Loss: `categorical_crossentropy`
- Optimizer: `adam`
- Epochs: 5
- Final Accuracy: **97.11%**

---

## 🏋️‍♂️ Training Output Snapshot

```
Epoch 5/5
loss: 0.0866 - accuracy: 0.9689
val_loss: 0.0784 - val_accuracy: 0.9706
Test Accuracy: 97.11%
```

---

## 🔍 Prediction Logic

```python
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from sklearn.preprocessing import LabelEncoder

model = load_model('CNN_model.h5')
class_names = ['Karacadag', 'Ipsala', 'Arborio', 'Basmati', 'Jasmine']
encoder = LabelEncoder().fit(class_names)

def preprocess_image(img_path, target_size=(50, 50)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

def predict_image(img_path):
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    predicted_class_idx = np.argmax(prediction, axis=1)[0]
    predicted_class = encoder.inverse_transform([predicted_class_idx])[0]
    return predicted_class, prediction[0][predicted_class_idx]
```

---

## 🌐 Streamlit Web App - https://ricevar.streamlit.app/

### Features

- Upload a rice grain image.
- Displays predicted rice variety and confidence score.
- Runs in the browser using Streamlit.

### To Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Project Structure

```
rice-variety-classifier/
├── Rice_Image_Dataset/
├── CNN_model.h5
├── app.py
├── try1.py
├── requirements.txt
└── README.md
```

---

## 🧪 Sample Prediction Output

```
Predicted Class: Basmati
Confidence: 99.99%
```

---

## 🧾 Requirements

```text
tensorflow
keras
numpy
pandas
matplotlib
seaborn
scikit-learn
streamlit
```

---

## 🙌 Acknowledgements

- TensorFlow & Keras for model development
- Streamlit for deployment interface
- Rice Image Dataset authors
