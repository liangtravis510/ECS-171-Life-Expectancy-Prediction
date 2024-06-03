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
st.set_page_config(page_title='Life Expectancy for Developed Countries', layout='wide')

# Title and description
st.title('LIFE EXPECTANCY FOR DEVELOPED COUNTRIES')
st.markdown('---')

# Input Attributes
st.header('Input Attributes')
att_adult_mortality = st.slider('Adult Mortality (per 1000 people)', min_value=0, max_value=24, value=10, step=1)
att_GDP = st.slider('GDP (per 10000$ USD)', min_value=0.0, max_value=12.0, value=2.1, step=0.1)
att_thinness_1_19_years = st.slider('Thinness 1-19 years (%)', min_value=0.0, max_value=3.0, value=1.1, step=0.1)



user_input = np.array([att_adult_mortality, att_GDP, att_thinness_1_19_years])

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

def get_filtered_data(remove_status):
    outliers = []
    for i, col in transformed_data.items():
      if i != "Status":
          outliers += find_outliers(col)
      else:
          outliers += list( np.where(df["Status"] == remove_status)[0] )
    
    
    data_no_outliers = transformed_data.drop(index=outliers).reset_index(drop=True).drop('Status', axis=1)
    # Replace missing values with mean
    imputer = SimpleImputer(strategy="mean")
    new_data = pd.DataFrame(imputer.fit_transform(data_no_outliers), columns=data_no_outliers.columns)

    train_data, test_data = train_test_split(new_data, train_size=0.8, random_state=1)
    train_data, val_data = train_test_split(train_data, train_size=0.8, random_state=1)

    return train_data, val_data, test_data

train_data, val_data, test_data = get_filtered_data(remove_status="Developing")


vars = ["Adult Mortality", "GDP", "Thinness 1-19 Years"]
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
        'Adult Mortality', 'GDP', 'Thinness 1-19 Years'
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

    # Display results
    st.markdown('**Result - Prediction!**')
    st.write('Based on the user input, the estimated Life Expectancy is:')
    st.info(predictions[0])
    st.write('Model Performance:')
    st.write('Error (MSE) for testing:')
    st.info(mse)
    st.markdown('---')