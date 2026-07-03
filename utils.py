import os
import gdown
import streamlit as st
import tensorflow as tf

MODEL_PATH = "model/Xception_AI_Human.keras"

FILE_ID = "ISI_FILE_ID_GOOGLE_DRIVE"

URL = f"https://drive.google.com/uc?id={FILE_ID}"


def download_model():

    os.makedirs("model", exist_ok=True)

    if not os.path.exists(MODEL_PATH):

        print("Downloading model...")

        gdown.download(
            URL,
            MODEL_PATH,
            quiet=False
        )

    return MODEL_PATH


@st.cache_resource
def load_xception():

    model_path = download_model()

    model = tf.keras.models.load_model(model_path)

    return model
