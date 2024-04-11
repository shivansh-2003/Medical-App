import numpy as np
import streamlit as st
import joblib
from xgboost import XGBClassifier

loaded_model = joblib.load('/Users/shivanshmahajan/Desktop/DataScinece/project/Medical App/combined/Pages/obesity.joblib')


def obese_prediction(input_data):
    input_data = np.asarray(input_data, dtype=np.float32)  # Convert input data to float
    input_data_reshaped = input_data.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)

    if (prediction == 0):
        return('normal weight')
    elif (prediction == 1):
        return('obese')
    elif (prediction == 2):
        return('overweight')
    else:
        return("underweight")

def main():
    st.title("Obese Prediction Web App")
    image_path = '/Users/shivanshmahajan/Desktop/DataScinece/project/Medical App/combined/Images/download.png'
    st.image(image_path, use_column_width=True)

    Age = st.text_input('Age')
    Gender=st.text_input('Gender')
    Height = st.text_input('Height')
    Weight = st.text_input('Weight')
    BMI=st.text_input('BMI')


    # creating a button for Prediction
    if st.button('BMI Test Result'):
        diagnosis = obese_prediction([Age,Gender,Height,Weight,BMI])
        st.success(diagnosis)


if __name__ == '__main__':
    main()
