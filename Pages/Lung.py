import numpy as np
import streamlit as st
import joblib
from xgboost import XGBClassifier

loaded_model = joblib.load('/Users/shivanshmahajan/Desktop/DataScinece/project/Medical App/combined/Pages/Lung_Cancer_model.joblib')


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
    image_path = '/Users/shivanshmahajan/Desktop/DataScinece/project/Medical App/combined/Images/images.jpeg'
    st.image(image_path, use_column_width=True)
    Gender = st.radio('GENDER', ['Male', 'Female'])
    Age = st.text_input('AGE')
    Smoke = st.radio('SMOKING', ['Yes', 'No'])
    Yellow_Fingers = st.radio('YELLOW_FINGERS', ['Yes', 'No'])
    Anxiety = st.radio('ANXIETY', ['Yes', 'No'])
    Peer_Pressure = st.radio('PEER_PRESSURE', ['Yes', 'No'])
    Chronic_Disease = st.radio('CHRONIC DISEASE', ['Yes', 'No'])
    Fatigue = st.radio('FATIGUE', ['Yes', 'No'])
    Allergy = st.radio('ALLERGY', ['Yes', 'No'])
    Wheezling = st.radio('WHEEZING', ['Yes', 'No'])
    Alcohol_Consumption = st.radio('ALCOHOL CONSUMING', ['Yes', 'No'])
    Coughing = st.radio('COUGHING', ['Yes', 'No'])
    Shortness_Of_Breath = st.radio('SHORTNESS OF BREATH', ['Yes', 'No'])
    Swallowing_difficulty = st.radio('SWALLOWING DIFFICULTY', ['Yes', 'No'])
    Chest_Pain = st.radio('CHEST PAIN', ['Yes', 'No'])

    # creating a button for Prediction
    if st.button('Lung Cancer Test Result'):
        diagnosis = Lung_prediction([Gender, Age, Smoke, Yellow_Fingers, Anxiety, Peer_Pressure, Chronic_Disease,
                                     Fatigue, Allergy, Wheezling, Alcohol_Consumption, Coughing,
                                     Shortness_Of_Breath, Swallowing_difficulty, Chest_Pain])
        st.success(diagnosis)


if __name__ == '__main__':
    main()
