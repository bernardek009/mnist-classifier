# streamlit_app/app.py

import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import requests
import os
import psycopg2
from datetime import datetime
import io

from inference import MNISTModel

# Initialize model
model = MNISTModel(model_path=os.getenv("MODEL_PATH", "model/model.pkl"))

# Setup DB connection (using environment variables)
db_host = os.getenv("POSTGRES_HOST", "localhost")
db_port = os.getenv("POSTGRES_PORT", "5432")
db_name = os.getenv("POSTGRES_DB", "mnist_db")
db_user = os.getenv("POSTGRES_USER", "mnist_user")
db_pass = os.getenv("POSTGRES_PASSWORD", "mnist_pass")

def get_db_connection():
    return psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_pass
    )

# Function to log prediction to DB
def log_prediction(predicted_digit, true_label):
    conn = get_db_connection()
    cursor = conn.cursor()
    insert_query = """
    INSERT INTO predictions (timestamp, predicted_digit, true_label)
    VALUES (%s, %s, %s)
    """
    cursor.execute(insert_query, (datetime.now(), predicted_digit, true_label))
    conn.commit()
    cursor.close()
    conn.close()

st.title("MNIST Digit Classifier")

st.write("Draw or upload an image of a digit (0–9).")


uploaded_file = st.file_uploader("Upload an image of a digit", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Digit", use_column_width=True)

    # Predict
    predicted_digit, confidence = model.predict(image)
    st.write(f"**Prediction:** {predicted_digit}")
    st.write(f"**Confidence:** {confidence:.4f}")

    # True label input
    true_label = st.text_input("True Label (if you know it):")

    if st.button("Submit True Label"):
        if true_label.isdigit():
            log_prediction(predicted_digit, true_label)
            st.success("Feedback logged to database!")
        else:
            st.error("Please enter a valid digit (0–9).")

