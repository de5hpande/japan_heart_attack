import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the preprocessor and model
with open("final_model/preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

with open("final_model/model.pkl", "rb") as f:
    model = pickle.load(f)

# Set page title and icon
st.set_page_config(
    page_title="❤️ Heart Attack Prediction App",
    page_icon="❤️",
    layout="wide"
)

# Add a title with emoji
st.title("❤️ Heart Attack Prediction App")
st.markdown("""
    <style>
    .big-font {
        font-size: 20px !important;
        color: #FF4B4B;
    }
    </style>
    <div class="big-font">
    Predict your risk of heart attack using this app! 🏥
    </div>
    """, unsafe_allow_html=True)

# Add a sidebar for user input
st.sidebar.header("📝 User Input Features")

# Function to get user input
def get_user_input():
    # Numerical features
    age = st.sidebar.slider("👤 Age", 18, 100, 30)
    cholesterol_level = st.sidebar.slider("🩸 Cholesterol Level", 100, 400, 200)
    bmi = st.sidebar.slider("⚖️ BMI", 10, 50, 25)
    heart_rate = st.sidebar.slider("💓 Heart Rate", 50, 120, 72)
    systolic_bp = st.sidebar.slider("🩺 Systolic BP", 90, 200, 120)
    diastolic_bp = st.sidebar.slider("🩺 Diastolic BP", 60, 120, 80)
    stress_levels = st.sidebar.slider("😫 Stress Levels", 1, 10, 5)

    # Categorical features
    gender = st.sidebar.selectbox("🚻 Gender", ["Male", "Female"])
    smoking_history = st.sidebar.selectbox("🚬 Smoking History", ["No", "Yes"])
    diabetes_history = st.sidebar.selectbox("🩸 Diabetes History", ["No", "Yes"])
    hypertension_history = st.sidebar.selectbox("🩺 Hypertension History", ["No", "Yes"])
    alcohol_consumption = st.sidebar.selectbox("🍷 Alcohol Consumption", ["Low", "Moderate", "High"])
    family_history = st.sidebar.selectbox("👨‍👩‍👧‍👦 Family History", ["No", "Yes"])
    physical_activity = st.sidebar.selectbox("🏃‍♂️ Physical Activity", ["Low", "Moderate", "High"])

    # Create a dictionary for user input
    user_data = {
        "Age": age,
        "Cholesterol_Level": cholesterol_level,
        "BMI": bmi,
        "Heart_Rate": heart_rate,
        "Systolic_BP": systolic_bp,
        "Diastolic_BP": diastolic_bp,
        "Stress_Levels": stress_levels,
        "Gender": gender,
        "Smoking_History": smoking_history,
        "Diabetes_History": diabetes_history,
        "Hypertension_History": hypertension_history,
        "Alcohol_Consumption": alcohol_consumption,
        "Family_History": family_history,
        "Physical_Activity": physical_activity
    }

    # Convert to DataFrame
    features = pd.DataFrame(user_data, index=[0])
    return features

# Get user input
user_input = get_user_input()

# Display user input
st.subheader("📊 User Input Features")
st.write(user_input)

# Add a submit button
if st.button("🚀 Submit"):
    # Preprocess user input
    user_input_preprocessed = preprocessor.transform(user_input)

    # Make prediction
    prediction = model.predict(user_input_preprocessed)
    prediction_proba = model.predict_proba(user_input_preprocessed)

    # Display prediction
    st.subheader("🔮 Prediction")
    heart_attack_risk = "High Risk ❌" if prediction[0] == 1 else "Low Risk ✅"
    st.markdown(f"""
        
        </style>
        <div class="prediction-font">
        {heart_attack_risk}
        </div>
        """, unsafe_allow_html=True)

    # Display prediction probability
    st.subheader("📈 Prediction Probability")
    st.write(f"Probability of Heart Attack: {prediction_proba[0][1] * 100:.2f}%")

# Add a footer
st.markdown("---")
st.markdown("""
    <style>
    .footer {
        font-size: 16px !important;
        color: #808080;
        text-align: center;
    }
    </style>
    <div class="footer">
    Made with ❤️ by Your Name
    </div>
    """, unsafe_allow_html=True)