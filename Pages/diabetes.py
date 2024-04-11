import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(
    open('/Users/shivanshmahajan/Desktop/DataScinece/project/Medical App/combined/Pages/trained_model.sav', 'rb'))


def diabetic_prediction(input_data):
    input_data = np.asarray(input_data)
    input_data_reshaped = input_data.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)

    print(prediction)
    if prediction == 0:
        return 'the person is not diabetic'
    else:
        return 'the person is diabetic'


def main():
    st.title("Diabetes Prediction Web app ")
   # image_path = '/Users/shivanshmahajan/Desktop/DataScinece/project/Medical App/combined/Images/Biru_Elemen___Mockup_Isometrik_Teknologi_dalam_Pendidikan_Presentasi_Teknologi__1_.jpg'
   # st.image(image_path, use_column_width=True)
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.slider('Glucose Level', 0, 200)
    BloodPressure = st.slider('Blood Pressure value', 0, 200)
    SkinThickness = st.slider('Skin Thickness value', 0, 100)
    Insulin = st.slider('Insulin Level', 0, 400)
    BMI = st.slider('BMI value', 0.0, 50.0)
    DiabetesPedigreeFunction = st.slider('Diabetes Pedigree Function value', 0.0, 2.5)
    Age = st.text_input('Age of the Person', 0)

    diagnosis = ''

    # creating a button for Prediction

    if st.button('Diabetes Test Result'):
        diagnosis = diabetic_prediction(
            [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])

    st.success(diagnosis)


if __name__ == '__main__':
    main()
