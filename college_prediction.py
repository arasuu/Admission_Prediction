import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load model and scaler
model = pickle.load(open('trained_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))  # Make sure this was saved after one-hot encoding

# Page title
st.title("🎓 Neural Network Admission Predictor")
st.markdown("Fill in your academic profile to predict your admission chance!")

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

# Debug info (optional)
# st.write("🔍 Input Data:", input_df)

# Predict button
if st.button("Predict Admission"):
    try:
        # Scale the input
        input_scaled = scaler.transform(input_df)

        # Predict probabilities
        prob = model.predict_proba(input_scaled)[0][1]

        st.markdown(f"📊 **Predicted Probability of Admission: {prob*100:.2f}%**")

        # Set threshold for admission
        threshold = 0.5
        if prob >= threshold:
            st.success("🎉 Congratulations! You are likely to be admitted!")
        else:
            st.warning("😞 Sorry, you may not be admitted.")

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
