# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 08:59:51 2024

@author: mukaram
"""

import numpy as np
import pickle
import streamlit as st
from streamlit_option_menu import option_menu


# loading the already tained models from their binary dump file
diabetes_model = pickle.load(open("saved_diabetes_model.sav", 'rb'))
heart_model = pickle.load(open("heart_disease_diagnosis.sav", 'rb'))


# creating sidebar menue for selecting the relevant prediction model
def diabetes_prediction(input):
    #buidling predictive model
    input_data=(5,166,72,19,175,25.8,0.587,51)
    input_data_as_numpy_array = np.asarray(input_data)
    reshaped_input_data = input_data_as_numpy_array.reshape(1, -1)
    #standardized_input_data = scaler.transform(reshaped_input_data)
    #print(standardized_input_data)
    prediction = diabetes_model.predict(reshaped_input_data)
    #print(prediction)
    if prediction[0] == 0:
      return 'not diabetic'
    else:
      return 'she is diabetic'
      
def heart_prediction(input):
    #buidling predictive model
    test_data = (58,0,0,100,248,0,0,122,0,1,1,0,2)
    test_data_np = np.asarray(test_data)
    test_data_reshaped = test_data_np.reshape(1,-1)
    prediction = heart_model.predict(test_data_reshaped)
    #print(prediction)
    if prediction[0] == 0:
      return 'You are perfect'
    else:
      return 'You have developed a heart disease'
    
    
    
with st.sidebar:
    selected = option_menu('Select the Prediction Option', 
                           ['Diabetes Prediction',
                            'Heart Prediction'],
                           icons=['activity',
                                  'heart'],
                           default_index=0)
# opening the relevant page based on the selection
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction Using ML')
    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.text_input('Pregnancies of Patient')
    with col2:
        Glucose = st.text_input('Glucose of Patient')
    with col3:
        BloodPressure = st.text_input('BloodPressure of Patient')
    with col1:
        SkinThickness = st.text_input('SkinThickness of Patient')
    with col2:
        Insulin = st.text_input('Insulin of Patient')
    with col3:
        BMI = st.text_input('BMI of Patient')
    with col1:
        DiabetesPedigreeFunction = st.text_input('DiabetesPedigreeFunction of Patient')
    with col2:
        Age = st.text_input('Age of Patient')
    
    if st.button('Predict'):
        prediction = diabetes_prediction([Pregnancies, Glucose, BloodPressure,	SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        st.success(prediction)
if selected == 'Heart Prediction':
    st.title('Heart Prediction Using ML')
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.text_input('Age of Patient')
    with col2:
        sex = st.text_input('sex of Patient')
    with col3:
        cp = st.text_input('cp of Patient')
    with col1:
        trestbps = st.text_input('trestbps of Patient')
    with col2:
        chol = st.text_input('chol of Patient')
    with col3:
        fbs = st.text_input('fbs of Patient')
    with col1:
        restecg = st.text_input('restecg of Patient')
    with col2:
        thalach = st.text_input('thalach of Patient')
    with col3:
        exang = st.text_input('exang of Patient')
    with col1:
        oldpeak = st.text_input('oldpeak of Patient')
    with col2:
        slope = st.text_input('slope of Patient')
    with col3:
        ca = st.text_input('ca of Patient')
    with col1:
        thal = st.text_input('thal of Patient')
    if st.button('Predict'):
        prediction = heart_prediction([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])
        st.success(prediction)
    
    
    
    