import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.set_page_config(page_title="Admission Predictor", page_icon="🎓", layout="centered")

# Load model and scaler
model = pickle.load(open('trained_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("🎓 Neural Network Admission Predictor")
st.markdown("Enter your academic profile to predict your admission chance!")

# Input
gre_score = st.number_input("GRE Score", min_value=260, max_value=340, value=320)
toefl_score = st.number_input("TOEFL Score", min_value=0, max_value=120, value=110)
sop = st.slider("SOP Strength (1-5)", 1.0, 5.0, 4.0)
lor = st.slider("LOR Strength (1-5)", 1.0, 5.0, 4.0)
cgpa = st.number_input("CGPA (out of 10)", min_value=6.0, max_value=10.0, value=8.5)
research = st.radio("Research Experience", ("No", "Yes"))
university_rating = st.selectbox("University Rating", [1, 2, 3, 4, 5])

# Convert binary + one-hot
research_0 = 1 if research == "No" else 0
research_1 = 1 if research == "Yes" else 0

univ_ratings = [0, 0, 0, 0, 0]
univ_ratings[university_rating - 1] = 1

# Construct input in fixed order
columns = [
    'GRE_Score', 'TOEFL_Score', 'SOP', 'LOR', 'CGPA',
    'Research_0', 'Research_1',
    'University_Rating_1', 'University_Rating_2', 'University_Rating_3', 'University_Rating_4', 'University_Rating_5'
]

input_df = pd.DataFrame([[
    gre_score, toefl_score, sop, lor, cgpa,
    research_0, research_1,
    univ_ratings[0], univ_ratings[1], univ_ratings[2], univ_ratings[3], univ_ratings[4]
]], columns=columns)

st.write("🧾 Input DataFrame:")
st.dataframe(input_df)

# 🧪 Debug: show column mismatch
try:
    st.write("✅ Scaler expects columns:")
    st.code(scaler.feature_names_in_.tolist())
    st.write("🔍 Your input columns:")
    st.code(input_df.columns.tolist())
except Exception as e:
    st.warning("Scaler does not have `feature_names_in_`. Was it fit with a DataFrame?")

# Prediction
if st.button("Predict Admission"):
    try:
        input_scaled = scaler.transform(input_df)
        prob = model.predict_proba(input_scaled)[0][1]

        st.markdown(f"📊 **Predicted Admission Probability: {prob*100:.2f}%**")
        if prob >= 0.5:
            st.success("🎉 Congratulations! You are likely to be admitted!")
        else:
            st.warning("😞 Sorry, you may not be admitted.")
    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
