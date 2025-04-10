import streamlit as st
import pickle
import pandas as pd
import numpy as np

# ✅ Set page config FIRST
st.set_page_config(page_title="Admission Predictor", page_icon="🎓", layout="centered")

# Load model and scaler
model = pickle.load(open('trained_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))  # Saved after one-hot encoding during training

# Page title
st.title("🎓 Neural Network Admission Predictor")
st.markdown("Enter your academic profile to predict your admission chance!")

# Input fields
gre_score = st.number_input("GRE Score", min_value=260, max_value=340, value=320)
toefl_score = st.number_input("TOEFL Score", min_value=0, max_value=120, value=110)
sop = st.slider("SOP Strength (1-5)", 1.0, 5.0, 4.0)
lor = st.slider("LOR Strength (1-5)", 1.0, 5.0, 4.0)
cgpa = st.number_input("CGPA (out of 10)", min_value=6.0, max_value=10.0, value=8.5)
research = st.radio("Research Experience", ("No", "Yes"))
university_rating = st.selectbox("University Rating", [1, 2, 3, 4, 5])

# Binary encode research
research_binary = 1 if research == "Yes" else 0

# One-hot encode university rating
rating_encoding = [0, 0, 0, 0, 0]
rating_encoding[university_rating - 1] = 1

# Create input dataframe
input_df = pd.DataFrame({
    'GRE_Score': [gre_score],
    'TOEFL_Score': [toefl_score],
    'SOP': [sop],
    'LOR': [lor],
    'CGPA': [cgpa],
    'Research_0': [1 - research_binary],
    'Research_1': [research_binary],
    'University_Rating_1': [rating_encoding[0]],
    'University_Rating_2': [rating_encoding[1]],
    'University_Rating_3': [rating_encoding[2]],
    'University_Rating_4': [rating_encoding[3]],
    'University_Rating_5': [rating_encoding[4]],
})

# Button to trigger prediction
if st.button("Predict Admission"):
    try:
        # Scale the input
        input_scaled = scaler.transform(input_df)

        # Predict probability
        prob = model.predict_proba(input_scaled)[0][1]
        st.markdown(f"📊 **Predicted Admission Probability: {prob*100:.2f}%**")

        if prob >= 0.5:
            st.success("🎉 Congratulations! You are likely to be admitted!")
        else:
            st.warning("😞 Sorry, you may not be admitted.")

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
