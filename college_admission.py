import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Load model
model = load_model("model.h5")

# App Title
st.title("Admission Prediction")

# User Input
gre = st.number_input("GRE Score", 0, 340)
toefl = st.number_input("TOEFL Score", 0, 120)
university_rating = st.slider("University Rating", 1, 5)
sop = st.slider("SOP Strength (1-5)", 1.0, 5.0, step=0.5)
lor = st.slider("LOR Strength (1-5)", 1.0, 5.0, step=0.5)
cgpa = st.number_input("CGPA (0-10)", 0.0, 10.0, step=0.1)
research = st.selectbox("Research Experience", ["No", "Yes"])

# Convert input
input_data = np.array([[gre, toefl, university_rating, sop, lor, cgpa, 1 if research == "Yes" else 0]])

# Predict
if st.button("Predict Admission Chance"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Chance of Admission: {prediction[0][0]*100:.2f}%")
