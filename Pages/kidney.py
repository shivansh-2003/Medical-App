import numpy as np
import streamlit as st
import joblib
from xgboost import XGBClassifier

loaded_model = joblib.load('/Users/shivanshmahajan/Desktop/DataScinece/project/Medical App/combined/Pages/kidney_disease_model.joblib')


def Kidney_prediction(input_data):
    input_data = np.asarray(input_data, dtype=np.float32)  # Convert input data to float
    input_data_reshaped = input_data.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)

    if prediction == 0:
        return 'The person is not having kidney prone.'
    else:
        return 'The person is having Kidney Disease.'


def main():
    st.title("Kidney Disease Prediction Web App")
   # image_path = '/Users/shivanshmahajan/Desktop/DataScinece/project/Medical App/combined/Images/istockphoto-465015220-612x612.jpg'
  #  st.image(image_path, use_column_width=True)

    # Using sliders for input values
    Serum_creatinine = st.slider('Serum Creatinine', min_value=0.0, max_value=17.0, step=0.1, value=1.0)
    Specific_gravity = st.slider('Specific Gravity', min_value=1.0, max_value=2.0, step=0.01, value=1.02)
    Albumin = st.slider('Albumin', min_value=0, max_value=5, step=1, value=0)

    # Using radio buttons for Diabetes_mellitus and Hypertension
    Diabetes_mellitus = st.radio('Diabetes Mellitus', ('No', 'Yes'))
    Hypertension = st.radio('Hypertension', ('No', 'Yes'))

    Haemoglobin = st.slider('Haemoglobin', min_value=0.0, max_value=20.0, step=0.1, value=13.0)
    Blood_pressure = st.slider('Blood Pressure', min_value=50, max_value=200, step=1, value=120)

    # Creating a button for Prediction
    if st.button('Kidney Failure Test Result'):
        diagnosis = Kidney_prediction(
            [Serum_creatinine, Specific_gravity, Albumin, Diabetes_mellitus == 'Yes', Haemoglobin,
             Hypertension == 'Yes', Blood_pressure])
        st.success(diagnosis)


if __name__ == '__main__':
    main()
