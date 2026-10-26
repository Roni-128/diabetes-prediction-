import streamlit as st
import pandas as pd
import joblib

# Load trained model
saved = joblib.load("diabetes_model.joblib")
model = saved["model"]
features = saved["features"]

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺")

st.title("🩺 Diabetes Prediction App")
st.write("Enter the health measurements below.")

# Input fields for the Pima Indians dataset
inputs = {
    "preg": st.number_input("Pregnancies", 0, 20, 1),
    "plas": st.number_input("Glucose level", 0, 250, 120),
    "pres": st.number_input("Blood pressure", 0, 150, 70),
    "skin": st.number_input("Skin thickness", 0, 100, 20),
    "insu": st.number_input("Insulin level", 0, 900, 80),
    "mass": st.number_input("BMI", 0.0, 80.0, 25.0),
    "pedi": st.number_input("Diabetes pedigree function", 0.0, 3.0, 0.5),
    "age": st.number_input("Age", 1, 120, 30)
}

if st.button("Predict"):
    # Arrange inputs in the model's original feature order
    row = {
        "preg": inputs["preg"],
        "plas": inputs["plas"],
        "pres": inputs["pres"],
        "skin": inputs["skin"],
        "insu": inputs["insu"],
        "mass": inputs["mass"],
        "pedi": inputs["pedi"],
        "age": inputs["age"]
    }

    input_df = pd.DataFrame([row])[features]

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.warning("The model flags this input as higher risk.")
    else:
        st.success("The model does not flag this input as higher risk.")

    st.write(f"Model-estimated positive probability: {probability:.1%}")
    st.caption(
        "This result is not a diagnosis. Consult a qualified healthcare "
        "professional for testing and interpretation."
    )
