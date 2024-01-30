import numpy as np
import streamlit as st
import joblib
from xgboost import XGBClassifier

loaded_model = joblib.load('/Users/shivanshmahajan/Desktop/Lung/Lung_Cancer_model.joblib')


def Lung_prediction(input_data):
    input_data = np.asarray(input_data, dtype=np.float32)  # Convert input data to float
    input_data_reshaped = input_data.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)

    if prediction == 0:
        return 'The person is not having Lung Cancer.'
    else:
        return 'The person is having Lung Cancer.'


def main():
    st.title("Lung Cancer Prediction Web App")
    image_path = '/Users/shivanshmahajan/Desktop/Medical App/combined/Images/images.jpeg'
    st.image(image_path, use_column_width=True)
    Gender = st.text_input('GENDER')
    Age = st.text_input('AGE')
    Smoke = st.text_input('SMOKING')
    Yellow_Fingers = st.text_input('YELLOW_FINGERS')
    Anxiety = st.text_input('ANXIETY')
    Peer_Pressure = st.text_input('PEER_PRESSURE')
    Chronic_Disease = st.text_input('CHRONIC DISEASE')
    Fatigue = st.text_input('FATIGUE')
    Allergy = st.text_input('ALLERGY')
    Wheezling = st.text_input('WHEEZING')
    Alcohol_Consumption = st.text_input('ALCOHOL CONSUMING')
    Coughing = st.text_input('COUGHING')
    Shortness_Of_Breath = st.text_input('SHORTNESS OF BREATH')
    Swallowing_difficulty = st.text_input('SWALLOWING DIFFICULTY')
    Chest_Pain = st.text_input('CHEST PAIN')


    # creating a button for Prediction
    if st.button('Lung Cancer Test Result'):
        diagnosis = Lung_prediction([Gender,Age,Smoke,Yellow_Fingers,Anxiety,Peer_Pressure,Chronic_Disease,Fatigue,Allergy,Wheezling,Alcohol_Consumption,Coughing,Shortness_Of_Breath,Swallowing_difficulty,Chest_Pain])
        st.success(diagnosis)


if __name__ == '__main__':
    main()
