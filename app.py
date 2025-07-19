import streamlit as st # S_S1, S_S2, S_S3, S_S4, S_S5, S_S6, S_S7, S_S8
import pandas as pd
import numpy as np
import joblib

# --- Load the trained model, label encoder, and categorical options ---
@st.cache_resource # Cache the model loading for efficiency
def load_resources():
    try:
        model = joblib.load('salary_prediction_model.pkl')
        le = joblib.load('label_encoder.pkl')
        categorical_options = joblib.load('categorical_options.pkl')
        return model, le, categorical_options
    except FileNotFoundError:
        st.error("Required model files not found. Please run 'train_and_save_model.py' first.")
        st.stop()

model_pipeline, label_encoder, categorical_options = load_resources()

# Define the order of columns as used during model training
# This is crucial for the ColumnTransformer to work correctly on new data
feature_columns = ['age', 'workclass', 'education', 'educational-num',
                   'marital-status', 'occupation', 'relationship', 'race',
                   'gender', 'capital-gain', 'capital-loss', 'hours-per-week',
                   'native-country']

# --- Streamlit App Interface ---
st.set_page_config(page_title="Employee Salary Predictor", layout="centered")

st.title("📊 Employee Salary Prediction App")
st.markdown("""
    Enter the details of an employee below to predict if their annual income is
    **<=50K** or **>50K**.
""")

# --- Input Widgets ---
st.header("Employee Details")

# Create columns for better layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=17, max_value=90, value=30, step=1)
    workclass = st.selectbox("Workclass", options=categorical_options['workclass'])
    education = st.selectbox("Education", options=categorical_options['education'])
    marital_status = st.selectbox("Marital Status", options=categorical_options['marital-status'])
    occupation = st.selectbox("Occupation", options=categorical_options['occupation'])
    relationship = st.selectbox("Relationship", options=categorical_options['relationship'])

with col2:
    race = st.selectbox("Race", options=categorical_options['race'])
    gender = st.selectbox("Gender", options=categorical_options['gender'])
    capital_gain = st.number_input("Capital Gain", min_value=0, max_value=100000, value=0, step=100)
    capital_loss = st.number_input("Capital Loss", min_value=0, max_value=5000, value=0, step=10)
    hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=99, value=40, step=1)
    native_country = st.selectbox("Native Country", options=categorical_options['native-country'])
    educational_num = st.number_input("Educational Num (Years of Education)", min_value=1, max_value=16, value=9, step=1)


# --- Prediction Button ---
st.markdown("---")
if st.button("Predict Salary"):
    # Create a DataFrame from user inputs
    input_data = pd.DataFrame([[
        age, workclass, education, educational_num, marital_status,
        occupation, relationship, race, gender, capital_gain,
        capital_loss, hours_per_week, native_country
    ]], columns=feature_columns)

    # Make prediction
    prediction_encoded = model_pipeline.predict(input_data)
    predicted_salary_bracket = label_encoder.inverse_transform(prediction_encoded)

    # Display prediction
    st.subheader("Prediction Result:")
    if predicted_salary_bracket == ">50K":
        st.success(f"The predicted annual income is: **{predicted_salary_bracket}** 🎉")
    else:
        st.info(f"The predicted annual income is: **{predicted_salary_bracket}**")

    st.markdown("---")
    st.caption("Note: This prediction is based on a machine learning model trained on the provided dataset.")
