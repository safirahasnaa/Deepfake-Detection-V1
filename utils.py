import os
import cv2
import gdown
import numpy as np
import tensorflow as tf
import streamlit as st

from tensorflow.keras.applications.xception import preprocess_input

MODEL_DIR = "model"
MODEL_NAME = "Xception_AI_Human.keras"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

FILE_ID = "1i8hPZ-WF9dbbcYjgylvim8SQMPgpw02X"

URL = f"https://drive.google.com/uc?id={FILE_ID}"


def download_model():

    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):

        with st.spinner("Downloading model... (hanya sekali)"):

            gdown.download(
                URL,
                MODEL_PATH,
                quiet=False
            )

    return MODEL_PATH


@st.cache_resource
def load_model():

    model_path = download_model()

    model = tf.keras.models.load_model(model_path)

    return model


def preprocess_image(image):

    img = np.array(image)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2RGB)

    img = cv2.resize(img, (299,299))

    img = preprocess_input(img.astype(np.float32))

    img = np.expand_dims(img,0)

    return img


def predict(image):

    model = load_model()

    x = preprocess_image(image)

    prob = float(model.predict(x, verbose=0)[0][0])

    if prob >= 0.5:

        label = "AI Generated"

        confidence = prob * 100

    else:

        label = "Human"

        confidence = (1 - prob) * 100

    return label, confidence, prob
