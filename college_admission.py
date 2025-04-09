import streamlit as st
import numpy as np
import pickle

# Load model
with open("trained_model.pkl", "rb") as file:
    model = pickle.load(file)

# Inputs
gre = st.number_input("GRE Score", min_value=260, max_value=340, value=320)
toefl = st.number_input("TOEFL Score", min_value=0, max_value=120, value=100)
sop = st.slider("SOP Strength", 1.0, 5.0, step=0.5, value=3.0)
lor = st.slider("LOR Strength", 1.0, 5.0, step=0.5, value=3.0)
cgpa = st.number_input("CGPA (0-10)", min_value=0.0, max_value=10.0, value=8.0)
univ_rating = st.selectbox("University Rating", [1, 2, 3, 4, 5], index=2)
research = st.radio("Research Experience", ["No", "Yes"], index=1)

if st.button("Predict"):
    try:
        # One-hot encoding
        univ_ratings = [1 if i == univ_rating else 0 for i in range(1, 6)]
        research_encoded = [1 if research == "No" else 0, 1 if research == "Yes" else 0]
        
        # Features in correct order
        features = [gre, toefl, sop, lor, cgpa, *univ_ratings, *research_encoded]
        
        # Debug: Test worst-case inputs
        bad_student = [260, 80, 1.0, 1.0, 6.0, 0, 0, 0, 0, 1, 1, 0]
        
        # Get prediction
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([features])[0][1]
            bad_proba = model.predict_proba([bad_student])[0][1]
        else:
            proba = model.predict([features])[0]
            bad_proba = model.predict([bad_student])[0]
        
        # Display
        st.success(f"Predicted Chance: {proba * 100:.1f}%")
        st.warning(f"Worst-case test prediction: {bad_proba * 100:.1f}%")
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
