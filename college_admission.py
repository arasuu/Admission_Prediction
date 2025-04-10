import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load the trained model and scaler
model = pickle.load(open('trained_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))  # Ensure this was saved during training

# Function to scale the input features using the saved scaler
def scale_input(input_data):
    return scaler.transform(input_data.values)  # Bypass column name mismatch

# Streamlit UI
st.title("🎓 Neural Network Admission Predictor")
st.write("📋 Enter your academic profile:")

# User inputs for the features
gre_score = st.number_input("GRE Score", min_value=260, max_value=340, value=320)
toefl_score = st.number_input("TOEFL Score", min_value=0, max_value=120, value=110)
sop = st.slider("Statement of Purpose (SOP) Strength", 1.0, 5.0, 4.0)
lor = st.slider("Letter of Recommendation (LOR) Strength", 1.0, 5.0, 4.0)
cgpa = st.number_input("CGPA (out of 10)", min_value=6.0, max_value=10.0, value=8.5)
research = st.radio("Research Experience", ("No", "Yes"))

# Convert research to binary
research_binary = 1 if research == "Yes" else 0

# University Rating Selection
university_rating = st.selectbox("University Rating (1 to 5)", [1, 2, 3, 4, 5])

# One-hot encode university rating
univ_rating_encoded = [1 if i == university_rating else 0 for i in range(1, 6)]

# Combine all input features into a DataFrame
input_data = pd.DataFrame({
    'GRE_Score': [gre_score],
    'TOEFL_Score': [toefl_score],
    'SOP': [sop],
    'LOR': [lor],
    'CGPA': [cgpa],
    'Research_0': [1 - research_binary],
    'Research_1': [research_binary],
    'University_Rating_1': [univ_rating_encoded[0]],
    'University_Rating_2': [univ_rating_encoded[1]],
    'University_Rating_3': [univ_rating_encoded[2]],
    'University_Rating_4': [univ_rating_encoded[3]],
    'University_Rating_5': [univ_rating_encoded[4]],
})

# Add the "Predict" button
if st.button("Predict"):
    try:
        # Scale input
        scaled_input = scale_input(input_data)

        # Predict
        prediction = model.predict(scaled_input)

        # Output
        if prediction[0] == 1:
            st.success("🎉 Congratulations! You are likely to be admitted!")
        else:
            st.warning("😞 Sorry, you may not be admitted.")
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
