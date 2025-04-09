import streamlit as st
import numpy as np
import pickle

# Load model
with open("trained_model.pkl", "rb") as file:
    model = pickle.load(file)

# Input fields
gre = st.number_input("GRE Score", min_value=260, max_value=340, value=320)
toefl = st.number_input("TOEFL Score", min_value=0, max_value=120, value=100)
sop = st.slider("SOP Strength", 1.0, 5.0, step=0.5, value=3.0)
lor = st.slider("LOR Strength", 1.0, 5.0, step=0.5, value=3.0)
cgpa = st.number_input("CGPA (0-10)", min_value=0.0, max_value=10.0, value=8.0)
univ_rating = st.selectbox("University Rating", [1, 2, 3, 4, 5], index=2)
research = st.radio("Research Experience", ["No", "Yes"], index=1)

if st.button("Predict Admission Chance"):
    try:
        # One-hot encode university rating (5 columns)
        univ_ratings = [1 if i == univ_rating else 0 for i in range(1, 6)]
        
        # One-hot encode research (2 columns)
        research_encoded = [1 if research == "No" else 0, 1 if research == "Yes" else 0]
        
        # ⚠️⚠️⚠️ MUST MATCH MODEL'S EXPECTED ORDER! ⚠️⚠️⚠️
        features = [
            gre, toefl, sop, lor, cgpa,  # First 5 features (numerical)
            *univ_ratings,  # Univ ratings (one-hot)
            *research_encoded  # Research (one-hot)
        ]
        
        input_data = np.array(features).reshape(1, -1)
        
        # Debug: Show exact feature order
        st.write("Features sent to model:", features)
        
        # Get prediction (use predict_proba if possible)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_data)[0][1]  # Probability of class "1" (Admitted)
            st.success(f"Predicted Chance: {proba * 100:.1f}%")
        else:
            pred = model.predict(input_data)[0]
            st.success(f"Prediction: {'Admitted' if pred == 1 else 'Rejected'}")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
