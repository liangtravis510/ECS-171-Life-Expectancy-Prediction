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
from sklearn.impute import SimpleImputer
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
att_hepatitis_b = st.slider('Hepatitis B Immunization (%)', min_value=0, max_value=100, value=65, step=5)
att_measles = st.slider('Measles Immunization', min_value=0, max_value=10000, value=1154, step=100)
att_bmi = st.slider('BMI', min_value=0.0, max_value=100.0, value=19.0, step=0.1)
att_polio = st.slider('Polio Immunization (%)', min_value=0, max_value=100, value=65, step=5)
att_total_expenditure = st.slider('Total Expenditure (% of GDP)', min_value=0.0, max_value=20.0, value=8.16, step=0.1)
att_diphtheria = st.slider('Diphtheria Immunization (%)', min_value=0, max_value=100, value=65, step=5)
att_hiv_aids = st.slider('HIV/AIDS', min_value=0.0, max_value=10.0, value=0.1, step=0.1)
att_gdp = st.number_input('GDP', min_value=0.0, max_value=1e6, value=584.26)
att_thinness_1_19_years = st.slider('Thinness 1-19 years (%)', min_value=0.0, max_value=50.0, value=17.2, step=0.1)
att_schooling = st.slider('Schooling (years)', min_value=0, max_value=20, value=10, step=1)

user_input = np.array([att_adult_mortality, att_infant_deaths, att_alcohol, att_hepatitis_b, 
                       att_measles, att_bmi, att_polio, att_total_expenditure, 
                       att_diphtheria, att_hiv_aids, att_gdp, att_thinness_1_19_years, 
                       att_schooling])

# Sidebar - Set Parameters
with st.sidebar.header('Set Parameters'):
    split_size = st.sidebar.slider('Data split ratio (percentage for Training Set)', min_value=10, max_value=90, value=20, step=10)

# Load dataset
def get_dataset():
    data = pd.read_csv("E:/ECS-171-Project/Life Expectancy Data.csv")
    return data

life_df = get_dataset()
df = life_df.copy()

# Data preprocessing
df.columns = df.columns.str.strip()
# Define specific columns to keep capitalized
specific_columns = ['GDP', 'HIV/AIDS', 'BMI']

# Capitalize the first letter of each feature name and keep specific columns capitalized
df.columns = [col if col in specific_columns else col.title() for col in df.columns]
df.rename(columns={'Gdp': 'GDP', 'Hiv/Aids': 'HIV/AIDS', 'Bmi': 'BMI'}, inplace=True)

# Replace missing values with mean
numeric_data = df.select_dtypes(include=[np.number])
imputer = SimpleImputer(strategy='mean')
numeric_data = imputer.fit_transform(numeric_data)
df.update(pd.DataFrame(numeric_data, columns=df.select_dtypes(include=[np.number]).columns))

# One-hot encoding for categorical variables
df = pd.get_dummies(df, columns=['Status'], drop_first=True)

# target variable
y = df['Life Expectancy']
# Features
X = df.drop(['Life Expectancy', 'Country', 'Year'], axis=1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(split_size/100), random_state=42)

if st.button('Estimate Life Expectancy'):
    
    # Model training
    model = LinearRegression()
    model.fit(X_train, y_train) 
    
    # Align user_input with X_train columns
    user_input_df = pd.DataFrame([user_input], columns=[
        'Adult Mortality', 'Infant Deaths', 'Alcohol', 'Hepatitis B', 
        'Measles', 'BMI', 'Polio', 'Total Expenditure', 
        'Diphtheria', 'HIV/AIDS', 'GDP', 'Thinness 1-19 Years', 
        'Schooling'
    ])
    # Add missing dummy columns for 'Status'
    for col in X_train.columns:
        if col not in user_input_df.columns:
            user_input_df[col] = 0
    
    # Reorder user_input_df columns to match X_train
    user_input_df = user_input_df[X_train.columns]

    # Make predictions
    predictions = model.predict(user_input_df)
    model_score = model.score(X_test, y_test)
    mse = mean_squared_error(y_test, model.predict(X_test))
    rmse = np.sqrt(mse)

    mse_train = mean_squared_error(y_train, model.predict(X_train))
    rmse_train = np.sqrt(mse_train)
    
    # Display results
    st.markdown('**Result - Prediction!**')
    st.write('Based on the user input, the estimated Life Expectancy is:')
    st.info(predictions[0])
    st.write('Model Performance:')
    st.write(f'R^2 score: {model_score}')
    st.write('Error (MSE) for testing:')
    st.info(mse)
    st.write(f"Root Mean Squared Error (RMSE) on test set: {rmse:.4f}")
    st.write('Error (MSE) for training:')

    st.markdown('---')
    show_data = st.checkbox("See the raw data?")
    if show_data:
        st.write(df)
