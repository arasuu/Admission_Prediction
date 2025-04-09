import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open("trained_model.pkl", "rb") as file:
    model = pickle.load(file)

# Streamlit app UI
st.set_page_config(page_title="Admission Chance Predictor", page_icon="🎓")
st.title("🎓 Admission Chance Predictor")
st.markdown("Fill out the details below to predict your chance of admission.")

# Input fields
gre = st.number_input("GRE Score", min_value=0, max_value=340, value=300)
toefl = st.number_input("TOEFL Score", min_value=0, max_value=120, value=100)
univ_rating = st.slider("University Rating", 1, 5, value=3)
sop = st.slider("SOP Strength", 1.0, 5.0, step=0.5, value=3.0)
lor = st.slider("LOR Strength", 1.0, 5.0, step=0.5, value=3.0)
cgpa = st.number_input("CGPA (0-10)", min_value=0.0, max_value=10.0, value=8.0)
research = st.selectbox("Research Experience", ["No", "Yes"])

# Prediction
if st.button("🎯 Predict Admission Chance"):
    try:
        input_data = np.array([[gre, toefl, univ_rating, sop, lor, cgpa, 1 if research == "Yes" else 0]], dtype=float)
        prediction = model.predict(input_data)[0]
        st.success(f"📈 Predicted Chance of Admission: {prediction * 100:.2f}%")
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
