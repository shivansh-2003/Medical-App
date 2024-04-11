import numpy as np
import streamlit as st
import joblib
from xgboost import XGBClassifier

loaded_model = joblib.load('/Users/shivanshmahajan/Desktop/DataScinece/project/Medical App/combined/Pages/heart_disease_model.joblib')

def Heart_prediction(input_data):
    input_data = np.asarray(input_data, dtype=np.float32)  # Convert input data to float
    input_data_reshaped = input_data.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)

    if prediction == 0:
        return 'The person is not heart-prone.'
    else:
        return 'The person is heart-prone.'

def main():
    st.title("Heart Failure Prediction Web App")
    image_path ='/Users/shivanshmahajan/Desktop/DataScinece/project/Medical App/combined/Images/Anatomy-of-the-heart.jpeg'
    st.image(image_path, use_column_width=True)
    Age = st.text_input('Age')
    Sex = st.text_input('Sex')
    ChestPain = st.text_input('ChestPainType')
    BP = st.text_input('RestingBP')
    Cholestrol = st.text_input('Cholestrol')
    BS = st.text_input('FastingBS')
    ECG = st.text_input('RestingECG')
    Angeina = st.text_input('ExerciseAngina')
    HR = st.text_input('MaxHR')
    Oldpeak = st.text_input('Oldpeak')
    St_slope = st.text_input('ST_Slope')

    # creating a button for Prediction
    if st.button('Heart Failure Test Result'):
        diagnosis = Heart_prediction([Age, Sex, ChestPain, BP, Cholestrol, BS, ECG, Angeina, HR, Oldpeak, St_slope])
        st.success(diagnosis)

if __name__ == '__main__':
    main()
