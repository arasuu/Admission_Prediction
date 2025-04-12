import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

st.title("🎓 UCLA Admission Chance Predictor")
st.write("Enter your profile details below to predict your chances of admission into UCLA.")

# Input fields
GRE = st.number_input("GRE Score", min_value=260, max_value=340, value=300)
TOEFL = st.number_input("TOEFL Score", min_value=0, max_value=120, value=100)
UniversityRating = st.slider("University Rating", min_value=1, max_value=5, value=3)
SOP = st.slider("Statement of Purpose (SOP) Strength", min_value=1.0, max_value=5.0, value=3.0, step=0.5)
LOR = st.slider("Letter of Recommendation (LOR) Strength", min_value=1.0, max_value=5.0, value=3.0, step=0.5)
CGPA = st.number_input("CGPA (out of 10)", min_value=0.0, max_value=10.0, value=8.5)
Research = st.radio("Research Experience", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# Input data
input_data = np.array([[GRE, TOEFL, UniversityRating, SOP, LOR, CGPA, Research]])

# Dummy scaler and model (replace this with actual trained model in production)
scaler = StandardScaler()
X_dummy = np.random.rand(100, 7)
scaler.fit(X_dummy)

model = Sequential()
from keras.layers import Input

model.add(Input(shape=(7,)))
model.add(Dense(10, activation='relu'))

model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_dummy, np.random.rand(100, 1), epochs=1, verbose=0)  # Dummy training

# Prediction
if st.button("Predict Chance of Admission"):
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)
    st.success(f"Your predicted chance of admission is: {prediction[0][0]*100:.2f}%")
