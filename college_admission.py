import streamlit as st
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix

# Load model and scaler (if used during training)
with open("trained_model.pkl", "rb") as f:
    model = pickle.load(f)

# App UI
st.set_page_config(page_title="Admission Predictor", layout="wide")
st.title("🎓 University Admission Predictor")
st.markdown("Predict if you'll be admitted based on your academic profile")

# Input Section
with st.sidebar:
    st.header("Your Profile")
    gre = st.slider("GRE Score", 260, 340, 320)
    toefl = st.slider("TOEFL Score", 0, 120, 100)
    cgpa = st.slider("CGPA", 0.0, 10.0, 8.0, step=0.1)
    sop = st.slider("Statement of Purpose (1-5)", 1.0, 5.0, 3.0, step=0.5)
    lor = st.slider("Letter of Rec (1-5)", 1.0, 5.0, 3.0, step=0.5)
    research = st.radio("Research Experience", ["No", "Yes"], index=1)
    univ_rating = st.selectbox("University Rating", [1, 2, 3, 4, 5], index=2)
    threshold = st.slider("Prediction Threshold", 0.1, 0.9, 0.5, 0.05,
                         help="Higher values make predictions more conservative")

# Prediction Logic
def predict_admission():
    # One-hot encoding
    univ_ratings = [1 if i == univ_rating else 0 for i in range(1, 6)]
    research_encoded = [1 if research == "No" else 0, 1 if research == "Yes" else 0]
    
    # Feature order MUST match training data
    features = [
        gre, toefl, sop, lor, cgpa,
        *univ_ratings,
        *research_encoded
    ]
    
    # Convert to numpy array
    input_data = np.array(features).reshape(1, -1)
    
    # Get prediction
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_data)[0][1]
        prediction = 1 if proba >= threshold else 0
    else:
        prediction = model.predict(input_data)[0]
        proba = None
    
    return prediction, proba, features

# Run Prediction
if st.button("Predict Admission"):
    prediction, proba, features = predict_admission()
    
    # Display Results
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Result")
        if prediction == 1:
            st.success("🎉 **Admitted!**", icon="✅")
        else:
            st.error("❌ Not Admitted", icon="🚫")
        
        if proba is not None:
            st.metric("Confidence", f"{proba*100:.1f}%")
    
    with col2:
        with st.expander("Debug Details"):
            st.write("**Features sent to model:**", features)
            if hasattr(model, "feature_names_in_"):
                st.write("**Model expects:**", list(model.feature_names_in_))
            
            # Sample test cases
            st.write("\n**Test Cases:**")
            test_cases = {
                "Strong Candidate": [340, 120, 5.0, 5.0, 9.5, 1,0,0,0,0, 0,1],
                "Weak Candidate": [260, 80, 1.0, 1.0, 6.0, 0,0,0,0,1, 1,0]
            }
            
            for name, case in test_cases.items():
                if hasattr(model, "predict_proba"):
                    case_proba = model.predict_proba([case])[0][1]
                    case_pred = 1 if case_proba >= threshold else 0
                else:
                    case_pred = model.predict([case])[0]
                    case_proba = None
                
                st.write(f"{name}: {'✅' if case_pred else '❌'} "
                        f"(Confidence: {case_proba*100:.1f}%)" if case_proba else "")

# Model Metrics (from your screenshot)
st.divider()
st.subheader("Model Performance Metrics")
col1, col2 = st.columns(2)
with col1:
    st.metric("Accuracy", "90%")
with col2:
    st.write("**Confusion Matrix:**")
    st.table([[63, 6], [4, 27]])

# Footer
st.caption("Note: Model thresholds can be adjusted for more conservative predictions")
