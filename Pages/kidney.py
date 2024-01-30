import numpy as np
import streamlit as st
import joblib
from xgboost import XGBClassifier


loaded_model = joblib.load('/Users/shivanshmahajan/Desktop/Kidney/kidney_disease_model.joblib')

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
    image_path = '/Users/shivanshmahajan/Desktop/Medical App/combined/Images/istockphoto-465015220-612x612.jpg'
    st.image(image_path, use_column_width=True)
    Serum_creatinine = st.text_input('serum_creatinine')
    Specific_gravity = st.text_input('specific_gravity')
    Albumin = st.text_input('albumin')
    Diabetes_mellitus = st.text_input('diabetes_mellitus')
    Haemoglobin = st.text_input('haemoglobin')
    Hypertension = st.text_input('hypertension')
    Blood_pressure = st.text_input('blood_pressure')

    

    # creating a button for Prediction
    if st.button('Kidney Failure Test Result'):
        diagnosis = Heart_prediction([Serum_creatinine,Specific_gravity,Albumin,Diabetes_mellitus, Haemoglobin,Hypertension,Blood_pressure])
        st.success(diagnosis)

if __name__ == '__main__':
    main()
