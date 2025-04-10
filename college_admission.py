import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
model = pickle.load(open('trained_model.pkl', 'rb'))

# Function to scale the input features using MinMaxScaler
def scale_input(input_data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(input_data)

# Streamlit UI
st.title("Admission Chance Prediction")

# User inputs for the features
gre_score = st.number_input("GRE Score", min_value=0, max_value=340, value=320)
toefl_score = st.number_input("TOEFL Score", min_value=0, max_value=120, value=110)
sop = st.slider("Statement of Purpose (SOP) Strength", 1.0, 5.0, 4.0)
lor = st.slider("Letter of Recommendation (LOR) Strength", 1.0, 5.0, 4.0)
cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=8.5)
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

# Scale the input data
scaled_input = scale_input(input_data)

# Predict the admission chance
prediction = model.predict(scaled_input)

# Show prediction result
if prediction[0] == 1:
    st.write("Congratulations! You are likely to be admitted!")
else:
    st.write("Sorry, you may not be admitted.")
