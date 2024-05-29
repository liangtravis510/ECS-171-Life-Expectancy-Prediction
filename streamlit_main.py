# import streamlit as st
# from streamlit_navigation_bar import st_navbar
# from pages import home as pg

# st.set_page_config(page_title='Life Expectancy Estimator Tool', layout='wide')

# pages = ['Home', 'Developed Countries', 'Developing Countries']
# page = st_navbar(pages)

# function = {'Home': pg.show_home}

# go_to_page = function.get(page)
# if go_to_page:
#     go_to_page()

# ------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Theme settings
st.set_page_config(page_title='Life Expectancy Estimator Tool', layout='wide')

# Title and description
st.title('LIFE EXPECTANCY ESTIMATOR TOOL')
st.markdown('---')
st.write('''
         This app uses a Linear Regression model to estimate life expectancy based on features and data obtained from World Bank Open Data.
         
Please fill in the attributes below and adjust the model's parameters to the desired values.

Once ready, please hit the 'ESTIMATE LIFE EXPECTANCY' button to get the prediction and the model's performance. 
''')
st.markdown('---')

# Input Attributes
st.header('Input Attributes')
att_adult_mortality = st.slider('Adult Mortality', min_value=0, max_value=700, value=263, step=10)
att_infant_deaths = st.slider('Infant Deaths', min_value=0, max_value=200, value=62, step=10)
att_alcohol = st.slider('Alcohol Consumption', min_value=0.0, max_value=20.0, value=0.01, step=0.1)
att_percentage_expenditure = st.slider('Percentage Expenditure', min_value=0, max_value=500, value=71, step=10)
att_hepatitis_b = st.slider('Hepatitis B Immunization (%)', min_value=0, max_value=100, value=65, step=5)
att_measles = st.slider('Measles Immunization', min_value=0, max_value=10000, value=1154, step=100)
att_bmi = st.slider('BMI', min_value=0.0, max_value=100.0, value=19.0, step=0.1)
att_under_five_deaths = st.slider('Under-five Deaths', min_value=0, max_value=200, value=83, step=10)
att_polio = st.slider('Polio Immunization (%)', min_value=0, max_value=100, value=65, step=5)
att_total_expenditure = st.slider('Total Expenditure (% of GDP)', min_value=0.0, max_value=20.0, value=8.16, step=0.1)
att_diphtheria = st.slider('Diphtheria Immunization (%)', min_value=0, max_value=100, value=65, step=5)
att_hiv_aids = st.slider('HIV/AIDS', min_value=0.0, max_value=10.0, value=0.1, step=0.1)
att_gdp = st.number_input('GDP', min_value=0.0, max_value=1e6, value=584.26)
att_population = st.number_input('Population', min_value=0.0, max_value=1e9, value=33736494.0)
att_thinness_1_19_years = st.slider('Thinness 1-19 years (%)', min_value=0.0, max_value=50.0, value=17.2, step=0.1)
att_thinness_5_9_years = st.slider('Thinness 5-9 years (%)', min_value=0.0, max_value=50.0, value=17.3, step=0.1)
att_income_composition = st.slider('Income Composition of Resources', min_value=0.0, max_value=1.0, value=0.479, step=0.01)
att_schooling = st.slider('Schooling (years)', min_value=0, max_value=20, value=10, step=1)

user_input = np.array([att_adult_mortality, att_infant_deaths, att_alcohol, att_percentage_expenditure, att_hepatitis_b, 
                       att_measles, att_bmi, att_under_five_deaths, att_polio, att_total_expenditure, 
                       att_diphtheria, att_hiv_aids, att_gdp, att_population, att_thinness_1_19_years, 
                       att_thinness_5_9_years, att_income_composition, att_schooling]).reshape(1, -1)

# Sidebar - Set Parameters
with st.sidebar.header('Set Parameters'):
    split_size = st.sidebar.slider('Data split ratio (percentage for Training Set)', min_value=10, max_value=90, value=20, step=10)

# Load dataset
@st.cache
def get_dataset():
    data = pd.read_csv("E:\ECS-171-Project\Life Expectancy Data.csv")
    return data

life_df = get_dataset()
df = life_df.copy()

# Data preprocessing
