import os
import cv2
import gdown
import numpy as np
import tensorflow as tf
import streamlit as st

from tensorflow.keras.applications.xception import preprocess_input

# ==========================================================
# GOOGLE DRIVE MODEL
# ==========================================================

FILE_ID = "1i8hPZ-WF9dbbcYjgylvim8SQMPgpw02X"

MODEL_DIR = "model"
MODEL_NAME = "Xception_AI_Human.keras"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

URL = f"https://drive.google.com/uc?id={FILE_ID}"


# ==========================================================
# DOWNLOAD MODEL
# ==========================================================

def download_model():

    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):

        st.info("Downloading AI model... (first time only)")

        gdown.download(
            URL,
            MODEL_PATH,
            quiet=False
        )

    return MODEL_PATH


# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_model():

    model_path = download_model()

    model = tf.keras.models.load_model(model_path)

    return model


# ==========================================================
# PREPROCESS
# ==========================================================

def preprocess_image(image):

    img = np.array(image)

    img = cv2.resize(img, (299,299))

    img = img.astype(np.float32)

    img = preprocess_input(img)

    img = np.expand_dims(img, axis=0)

    return img


# ==========================================================
# PREDICTION
# ==========================================================

def predict(image):

    model = load_model()

    x = preprocess_image(image)

    probability = float(model.predict(x, verbose=0)[0][0])

    if probability >= 0.5:

        label = "AI"

        confidence = probability * 100

    else:

        label = "Human"

        confidence = (1-probability) * 100

    return label, confidence, probability
