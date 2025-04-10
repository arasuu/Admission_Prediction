import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load the trained model and scaler
model = pickle.load(open('trained_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))  # Assuming you've saved your scaler during model training

# Function to scale the input features using the saved scaler
def scale_input(input_data):
    return scaler.transform(input_data)  # Use transform instead of fit_transform

# Streamlit UI
st.title("🎓 Neural Network Admission Predictor")
st.write("📋 Enter your academic profile:")

# User inputs for the features
gre_score = st.number_input("GRE Score", min_value=0, max_value=340, value=320)
toefl_score = st.number_input("TOEFL Score", min_value=0, max_value=120, value=110)
sop = st.slider("Statement of Purpose (SOP) Strength", 1.0, 5.0, 4.0)
lor = st.slider("Letter of Recommendation (LOR) Strength", 1.0, 5.0, 4.0)
cgpa = st.number_input("CGPA (out of 10)", min_value=0.0, max_value=10.0, value=8.5)
research = st.radio("Research Experience", ("No", "Yes"))

# Convert research to binary
research_binary = 1 if research == "Yes" else 0

# University Rating (One-hot encoded, example for University_Rating_1)
university_rating = [0, 0, 0, 1, 0]  # Assuming the user selects rating 4 out of 5

# Combine all input features into a DataFrame
input_data = pd.DataFrame({
    'GRE_Score': [gre_score],
    'TOEFL_Score': [toefl_score],
    'SOP': [sop],
    'LOR': [lor],
    'CGPA': [cgpa],
    'Research_0': [1 - research_binary],
    'Research_1': [research_binary],
    'University_Rating_1': [university_rating[0]],
    'University_Rating_2': [university_rating[1]],
    'University_Rating_3': [university_rating[2]],
    'University_Rating_4': [university_rating[3]],
    'University_Rating_5': [university_rating[4]],
})

# Display input data (for debugging)
st.write("Input Data:", input_data)

# Add the "Predict" button
if st.button("Predict"):
    # Ensure input_data is correctly shaped and scaled
    scaled_input = scale_input(input_data)

    # Check the scaled input values before prediction
    st.write("Scaled Input:", scaled_input)

    # Predict the admission chance
    try:
        prediction = model.predict(scaled_input)
        st.write("Prediction:", prediction)

        if prediction[0] == 1:
            st.write("🎉 Congratulations! You are likely to be admitted!")
        else:
            st.write("😞 Sorry, you may not be admitted.")
    except ValueError as e:
        st.error(f"Prediction Error: {str(e)}")

