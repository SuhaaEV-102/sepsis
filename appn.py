import streamlit as st
import pickle
import numpy as np

# Load the trained model and scaler
@st.cache_resource

with open("sepsis_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Navigation Menu
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "About", "Signup", "Login", "Prediction"])

if page == "Home":
    st.title("Sepsis Foresee - Home")
    st.image("in.jpg", use_column_width=True)
    st.markdown("### Join us in the fight against sepsis")
    if st.button("Get Started"):
        st.switch_page("Signup")

elif page == "About":
    st.title("Why Sepsis Foresee?")
    st.write("Sepsis Foresee is an AI-based platform designed to predict sepsis at an early stage, providing life-saving insights.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Early Detection**\nQuick and accurate predictions to help doctors take immediate action.")
    with col2:
        st.markdown("**AI-Powered Analysis**\nAdvanced machine learning algorithms ensure precision and reliability.")
    with col3:
        st.markdown("**Easy to Use**\nUser-friendly interface with a seamless experience.")

elif page == "Signup":
    st.title("Signup Page")
    full_name = st.text_input("Full Name")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    if st.button("Sign Up"):
        st.success("Account created successfully!")

elif page == "Login":
    st.title("Login Page")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        st.success("Logged in successfully!")

elif page == "Prediction":
    st.title("Sepsis Prediction")
    st.write("Enter patient details below for sepsis prediction.")

    # Input fields
    feature_names = ['PRG', 'PL', 'PR', 'SK', 'TS', 'M11', 'BD2', 'Age', 'Insurance']
    user_input = []
    for feature in feature_names:
        user_input.append(st.number_input(f"Enter {feature}", value=0.0))

    # Predict button
    if st.button("Predict"):
        input_array = np.array(user_input).reshape(1, -1)
        scaled_input = scaler.transform(input_array)
        prediction = model.predict(scaled_input)
        result = "Positive Sepsis" if prediction[0] == 1 else "Negative Sepsis"
        st.write(f"### Prediction: {result}")
