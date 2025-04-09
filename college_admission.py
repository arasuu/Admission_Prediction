import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler (if you used one)
with open("trained_model.pkl", "rb") as file:
    model = pickle.load(file)

# Streamlit app UI
st.set_page_config(page_title="Admission Chance Predictor", page_icon="🎓")

st.title("🎓 Admission Chance Predictor")
st.markdown("Fill out the details below to predict your chance of admission.")

# Input fields
gre = st.number_input("GRE Score", min_value=260, max_value=340, value=320)
toefl = st.number_input("TOEFL Score", min_value=0, max_value=120, value=100)
univ_rating = st.slider("University Rating", 1, 5, value=3)
sop = st.slider("SOP Strength", 1.0, 5.0, step=0.5, value=3.0)
lor = st.slider("LOR Strength", 1.0, 5.0, step=0.5, value=3.0)
cgpa = st.number_input("CGPA (0-10)", min_value=0.0, max_value=10.0, value=8.0)
research = st.selectbox("Research Experience", ["No", "Yes"])

# Prediction
if st.button("🎯 Predict Admission Chance"):
    try:
        # Create one-hot encoding
        univ_ratings = [1 if i == univ_rating else 0 for i in range(1, 6)]
        research_encoded = [1 if research == "No" else 0, 1 if research == "Yes" else 0]
        
        # Prepare features in correct order (match your training data)
        features = [
            gre, toefl, sop, lor, cgpa,  # First 5 features
            *univ_ratings,  # University ratings (one-hot)
            *research_encoded  # Research (one-hot)
        ]
        
        # Convert to numpy array and reshape
        input_data = np.array(features).reshape(1, -1)
        
        # If your model expects probabilities
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_data)[0][1]  # Get probability of class 1
            st.success(f"📈 Predicted Chance of Admission: {proba * 100:.2f}%")
        else:
            prediction = model.predict(input_data)[0]
            st.success(f"📈 Prediction: {'Admitted' if prediction == 1 else 'Not Admitted'}")
            
        # Debug output (you can remove this later)
        with st.expander("Debug Info"):
            st.write("Input features:", features)
            if hasattr(model, "feature_names_in_"):
                st.write("Model expects features in this order:", model.feature_names_in_)
            
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
