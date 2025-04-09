import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open("trained_model.pkl", "rb") as file:
    model = pickle.load(file)

# Streamlit UI
st.title("🎓 Admission Predictor (Yes/No)")
st.markdown("Will you get admitted? Fill in your details to find out!")

# Input fields
col1, col2 = st.columns(2)
with col1:
    gre = st.number_input("GRE Score", min_value=260, max_value=340, value=320)
    toefl = st.number_input("TOEFL Score", min_value=0, max_value=120, value=100)
    cgpa = st.number_input("CGPA (0-10)", min_value=0.0, max_value=10.0, value=8.0)
with col2:
    sop = st.slider("SOP Strength", 1.0, 5.0, step=0.5, value=3.0)
    lor = st.slider("LOR Strength", 1.0, 5.0, step=0.5, value=3.0)
    univ_rating = st.selectbox("University Rating", [1, 2, 3, 4, 5])
    research = st.radio("Research Experience", ["No", "Yes"])

if st.button("🚀 Predict Admission"):
    try:
        # One-hot encoding
        univ_ratings = [1 if i == univ_rating else 0 for i in range(1, 6)]
        research_encoded = [1 if research == "No" else 0, 1 if research == "Yes" else 0]
        
        # Feature order: GRE, TOEFL, SOP, LOR, CGPA, Univ_Ratings, Research
        features = [gre, toefl, sop, lor, cgpa, *univ_ratings, *research_encoded]
        
        # Predict (0 or 1)
        prediction = model.predict([features])[0]
        
        # Display "Yes" or "No"
        result = "✅ Yes" if prediction == 1 else "❌ No"
        st.success(f"**Prediction:** {result}")
        
        # Debug (optional)
        with st.expander("Debug Details"):
            st.write("Features sent:", features)
            if hasattr(model, "feature_names_in_"):
                st.write("Model expects:", model.feature_names_in_)
            
    except Exception as e:
        st.error(f"Error: {str(e)}")


