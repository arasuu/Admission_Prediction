import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open("trained_model.pkl", "rb") as file:
    model = pickle.load(file)

# Streamlit app UI
st.set_page_config(page_title="Admission Chance Predictor", page_icon="🎓")

st.title("🎓 Admission Chance Predictor")
st.markdown("Fill out the details below to predict your chance of admission.")

# Input fields
gre = st.number_input("GRE Score", min_value=0, max_value=340, value=300)
toefl = st.number_input("TOEFL Score", min_value=0, max_value=120, value=100)
univ_rating = st.slider("University Rating", 1, 5, value=3)
sop = st.slider("SOP Strength", 1.0, 5.0, step=0.5, value=3.0)
lor = st.slider("LOR Strength", 1.0, 5.0, step=0.5, value=3.0)
cgpa = st.number_input("CGPA (0-10)", min_value=0.0, max_value=10.0, value=8.0)
research = st.selectbox("Research Experience", ["No", "Yes"])

# Prediction
if st.button("🎯 Predict Admission Chance"):
    try:
        # Create one-hot encoding for university rating
        univ_rating_1 = 1 if univ_rating == 1 else 0
        univ_rating_2 = 1 if univ_rating == 2 else 0
        univ_rating_3 = 1 if univ_rating == 3 else 0
        univ_rating_4 = 1 if univ_rating == 4 else 0
        univ_rating_5 = 1 if univ_rating == 5 else 0
        
        # Create one-hot encoding for research
        research_0 = 1 if research == "No" else 0
        research_1 = 1 if research == "Yes" else 0
        
        # Prepare input array with all 12 features in the correct order
        input_data = np.array([
            [gre, toefl, sop, lor, cgpa,  # First 5 features
             univ_rating_1, univ_rating_2, univ_rating_3, univ_rating_4, univ_rating_5,  # University rating one-hot
             research_0, research_1]  # Research one-hot
        ], dtype=float)
        
        prediction = model.predict(input_data)[0]
        st.success(f"📈 Predicted Chance of Admission: {prediction * 100:.2f}%")
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
