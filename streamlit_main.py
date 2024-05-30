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
         This app uses a Linear Regression model to estimate life expectancy based on features and data obtained the World Health Organization (WHO) on Kaggle
         
Please fill in the attributes below and adjust the model's parameters to the desired values.

Once ready, please hit the 'ESTIMATE LIFE EXPECTANCY' button to get the prediction and the model's performance. 
''')
st.markdown('---')

# Input Attributes
st.header('Input Attributes')
att_adult_mortality = st.slider('Adult Mortality', min_value=0, max_value=24, value=10, step=1)
att_infant_deaths = st.slider('Infant Deaths', min_value=0, max_value=55, value=10, step=1)
att_alcohol = st.slider('Alcohol Consumption', min_value=0.0, max_value=4.3, value=0.01, step=0.1)
att_hepatitis_b = st.slider('Hepatitis B Immunization (%)', min_value=0.0, max_value=5.0, value=3.1, step=0.1)
att_measles = st.slider('Measles Immunization', min_value=0.0, max_value=11.7, value=5.0, step=0.1)
att_bmi = st.slider('BMI', min_value=0.0, max_value=5.0, value=1.0, step=0.1)
att_polio = st.slider('Polio Immunization (%)', min_value=0.0, max_value=5.0, value=1.0, step=0.1)
att_total_expenditure = st.slider('Total Expenditure (% of GDP)', min_value=0.0, max_value=13.0, value=8.16, step=0.1)
att_diphtheria = st.slider('Diphtheria Immunization (%)', min_value=0.0, max_value=5.0, value=3.2, step=0.01)
att_hiv_aids = st.slider('HIV/AIDS deaths (per 1000 lives)', min_value=0.0, max_value=3.0, value=0.1, step=0.1)
att_gdp = st.slider('GDP (10000)', min_value=0.0, max_value=12.0, value=2.1, step=0.1)
att_thinness_1_19_years = st.slider('Thinness 1-19 years (%)', min_value=0.0, max_value=3.0, value=1.1, step=0.1)
att_schooling = st.slider('Schooling (years)', min_value=0, max_value=18, value=10, step=1)

user_input = np.array([att_adult_mortality, att_infant_deaths, att_alcohol, att_hepatitis_b, 
                       att_measles, att_bmi, att_polio, att_total_expenditure, 
                       att_diphtheria, att_hiv_aids, att_gdp, att_thinness_1_19_years, 
                       att_schooling])

# Sidebar - Set Parameters
with st.sidebar.header('Set Parameters'):
    split_size = st.sidebar.slider('Data split ratio (percentage for Training Set)', min_value=10, max_value=90, value=20, step=10)

# Load dataset
def get_dataset():
    data = pd.read_csv("Life Expectancy Data.csv")
    return data

life_df = get_dataset()
df = life_df.copy()

selected_cols = ["Alcohol", "Adult Mortality", "Hepatitis B", "Measles ", " BMI ",
                 "Polio", "Total expenditure", "Diphtheria ", " HIV/AIDS",
                 "GDP", " thinness  1-19 years", "Schooling", "infant deaths", "Life expectancy ", "Status"]
selected_data = df[selected_cols]
selected_data.columns = ["Alcohol", "Adult Mortality", "Hepatitis B", "Measles", "BMI",
                         "Polio", "Total Expenditure", "Diphtheria", "HIV/AIDS",
                         "GDP", "Thinness 1-19 Years", "Schooling", "Infant Deaths", "Life Expectancy", "Status"]

# Transform selected columns
transformed_data = selected_data.apply({"Adult Mortality": np.sqrt,
                                     "Life Expectancy": lambda x: x,
                                     "Alcohol": np.sqrt,
                                     "Hepatitis B": lambda x: np.log(100 - x),
                                     "Measles": lambda x: np.log(x + 0.1),
                                     "BMI": lambda x: np.log(100 - x),
                                     "Polio": lambda x: np.log(100 - x),
                                     "Total Expenditure": lambda x: x,
                                     "Diphtheria": lambda x: np.log(100 - x),
                                     "HIV/AIDS": np.log,
                                     "GDP": np.log,
                                     "Thinness 1-19 Years": np.log,
                                     "Schooling": lambda x: x,
                                     "Status": lambda x: x,
                                     "Infant Deaths": lambda x: x})

# Remove rows with outliers
def find_outliers(data):
  q1 = np.nanpercentile(data, 25)
  q3 = np.nanpercentile(data, 75)
  iqr = q3 - q1
  min_threshold = q1 - 1.5*iqr
  max_threshold = q3 + 1.5*iqr
  return list( np.where((data < min_threshold) | (data > max_threshold))[0] )

outliers = []
for i, col in transformed_data.items():
  if i != "Status":
      outliers += find_outliers(col)
  else:
      outliers += list( np.where(df["Status"] == "Developed")[0] )


data_no_outliers = transformed_data.drop(index=outliers).reset_index(drop=True).drop('Status', axis=1)

# Replace missing values with mean
imputer = SimpleImputer(strategy="mean").set_output(transform="pandas")
new_data = imputer.fit_transform(data_no_outliers)

# Perform 80/20 train/test split
train_data, test_data = train_test_split(new_data, train_size=split_size/100, random_state=1)
train_data, val_data = train_test_split(train_data, train_size=(split_size/100), random_state=1)

# Features
vars = ["Alcohol", "Adult Mortality", "Hepatitis B", "Measles", "BMI",
                         "Polio", "Total Expenditure", "Diphtheria", "HIV/AIDS",
                         "GDP", "Thinness 1-19 Years", "Schooling", "Infant Deaths"]

X_test = test_data[vars]
y_test = test_data["Life Expectancy"]

# Prepare training and validation sets
X_train = train_data[vars]
X_val = val_data[vars]
y_train = train_data["Life Expectancy"]
y_val = val_data["Life Expectancy"]

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
    st.write('Error (MSE) for testing:')
    st.info(mse)
    st.markdown('---')
    