import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained model and scaler
model = pickle.load(open('trained_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))  # Ensure this is the scaler you used during training

# Streamlit app title
st.title("🎓 Neural Network Admission Predictor")
st.write("📋 Enter your academic profile:")

# Input fields for each feature
gre_score = st.number_input("GRE Score", min_value=260, max_value=340, value=320)
toefl_score = st.number_input("TOEFL Score", min_value=0, max_value=120, value=110)
sop = st.slider("Statement of Purpose (SOP) Strength", 1.0, 5.0, 4.0)
lor = st.slider("Letter of Recommendation (LOR) Strength", 1.0, 5.0, 4.0)
cgpa = st.number_input("CGPA (out of 10)", min_value=6.0, max_value=10.0, value=8.5)
research = st.radio("Research Experience", ("No", "Yes"))
university_rating = st.selectbox("University Rating (1-5)", [1, 2, 3, 4, 5])

# Encode 'Research' and 'University Rating'
research_0 = 1 if research == "No" else 0
research_1 = 1 if research == "Yes" else 0

# One-hot encoding for university rating (index 1 to 5)
univ_rating = [0] * 5
univ_rating[university_rating - 1] = 1

# Ensure features are in the correct order as used in training the model
feature_names = [
    'GRE_Score', 'TOEFL_Score', 'SOP', 'LOR', 'CGPA',
    'Research_0', 'Research_1',
    'University_Rating_1', 'University_Rating_2', 'University_Rating_3',
    'University_Rating_4', 'University_Rating_5'
]

# Combine features into a list, matching the feature order
feature_values = [
    gre_score, toefl_score, sop, lor, cgpa,
    research_0, research_1,
    *univ_rating
]

# Create a DataFrame with the correct order of features
input_data = pd.DataFrame([dict(zip(feature_names, feature_values))])

# Display the input data for debugging
st.subheader("🔍 Input Data")
st.write(input_data)

# Button to make prediction
if st.button("🎯 Predict Admission"):
    try:
        # Scale the input features using the previously fitted scaler
        scaled_input = scaler.transform(input_data)

        # Make prediction using the trained model
        prediction = model.predict(scaled_input)

        # Show prediction result
        if prediction[0] == 1:
            st.success("🎉 Congratulations! You are likely to be admitted!")
        else:
            st.error("😞 Sorry, you may not be admitted.")
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
