# 🎓 Neural Network Admission Predictor

This project provides a Machine Learning model that predicts whether a student will be admitted to a university based on their academic profile.

## Features
- Predict admission chances based on:
  - GRE Score
  - TOEFL Score
  - Statement of Purpose (SOP) Strength
  - Letter of Recommendation (LOR) Strength
  - CGPA (out of 10)
  - Research Experience
  - University Rating

How it Works
Input Data: The user inputs their academic details, including GRE Score, TOEFL Score, SOP Strength, LOR Strength, CGPA, Research Experience, and University Rating.

Preprocessing: The input data is processed and scaled using the same scaler (scaler.pkl) used during model training.

Prediction: The scaled input data is fed into the trained model (trained_model.pkl) to predict the admission chances.

Output: The result will indicate if the student is likely to be admitted or not.

Model Details
The model is a Neural Network (MLPClassifier from Scikit-learn).

The model was trained using features such as GRE Score, TOEFL Score, CGPA, SOP, LOR, University Rating, and Research Experience to predict the admission chance.

🤝 Acknowledgements Thanks to all contributors and dataset providers. This is a learning project for educational and demonstrative purposes.

Streamlit App - [Click Here](https://admissionprediction-ngetxvkzuawxdjtx7ikktv.streamlit.app/)

🔗 Author: Arasu Ragupathi 📧 Contact: arasuragu23@gmail.com 🌟 GitHub: https://github.com/arasuu

