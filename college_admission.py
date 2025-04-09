import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open("trained_model.pkl", "rb") as file:
    model = pickle.load(file)

# Streamlit app UI
st.title("🎓 Admission Chance Predictor")

gre = st.number_input("GRE Score", min_value=0, max_value=340)
toefl = st.number_input("TOEFL Score", min_value=0, max_value=120)
univ_rating = st.slider("University Rating", 1, 5)
sop = st.slider("SOP Strength", 1.0, 5.0, step=0.5)
lor = st.slider("LOR Strength", 1.0, 5.0, step=0.5)
cgpa = st.number_input("CGPA (0-10)", min_value=0.0, max_value=10.0)
research = st.selectbox("Research Experience", ["No", "Yes"])

if st.button("🎯 Predict Admission Chance"):
    input_data = np.array([[gre, toefl, univ_rating, sop, lor, cgpa, 1 if research == "Yes" else 0]])
    prediction = model.predict(input_data)[0]

    st.success(f"📈 Predicted Chance of Admission: {prediction*100:.2f}%")
