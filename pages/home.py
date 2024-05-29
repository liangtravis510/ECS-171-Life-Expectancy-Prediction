import streamlit as st 
import numpy as np
import pandas as pd


def show_home():
    # Title and description
    st.title('LIFE EXPECTANCY ESTIMATOR TOOL')
    st.markdown('---')
    st.write('''
         This app uses a Linear Regression model to estimate life expectancy based on features and data obtained from WHO Dataset.
         Please fill in the attributes below and adjust the model's parameters to the desired values.
         Once ready, please hit the 'ESTIMATE LIFE EXPECTANCY' button to get the prediction and the model's performance. 
         ''')
    st.markdown('---')

    # Input Attributes
    st.header('Input Attributes')
    att_adult_mortality = st.slider('Adult Mortality %', min_value=0, max_value=100, value=0, step=1)
    att_Hepatitis_B = st.slider('Hepatitis B %', min_value=0, max_value=100, value=0, step=1)
    att_Measles = st.slider('Measles %', min_value=0, max_value=100, value=0, step=1)
    att_Diphtheria = st.slider('Diphtheria %', min_value=0, max_value=100, value=0, step=1)
    att_HIV_AIDS = st.slider('HIV/AIDS %', min_value=0, max_value=100, value=0, step=1)
    att_Schooling = st.slider('Schooling %', min_value=0, max_value=100, value=0, step=1)
    att_gdp = st.number_input('GDP', min_value=0.0, max_value=1e6, value=584.26)
    user_input = np.array([att_adult_mortality, att_Hepatitis_B, att_Measles, att_Diphtheria, att_HIV_AIDS, att_gdp, att_Schooling]).reshape(1, -1)

    # Sidebar - Set parameters
    with st.sidebar.header('Set Parameters'):
        split_size = st.sidebar.slider('Data split ratio (percentage for Training Set)', min_value=10, max_value=90, value=20, step=10)

