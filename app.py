import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
import joblib
import os


#INTRO
st.title("House Pricing Prediction")

#LOAD IN TRAINED MODEL FROM SAV FILE
loaded_random_forest_file = 'sav_files/ran_for_model.sav'
loaded_random_forest_model = None

#CHECK IF THE MODEL FILE EXISTS OTHERWISE CREATE AN EMPTY FILE
os.makedirs(os.path.dirname(loaded_random_forest_file), exist_ok=True)

if not os.path.exists(loaded_random_forest_file):
    with open(loaded_random_forest_file, 'w') as file:
        file.write('')
    st.warning('Model file was not found. An empty file has been created at ' + loaded_random_forest_file + '. Please wait while I create a new model for you.')
elif os.path.getsize(loaded_random_forest_file) == 0:
    st.warning('Model file at ' + loaded_random_forest_file + ' is empty. Please wait while I create a new model for you.')
else:
    try:
        loaded_random_forest_model = joblib.load(loaded_random_forest_file)
        st.success('Model loaded successfully!')
    except EOFError:
        st.warning('The model file at' + loaded_random_forest_file + 'is corrupted or incomplete.')
    except Exception as e:
        st.warning('An error occurred while loading the model:' + e)

#LOAD IN CSV FILE
csvFile = pd.read_csv('housing.csv')

#SEPERATE OCEAN PROXIMITY INTO ORDINAL CATEGORIES
unique_categories = csvFile['ocean_proximity'].unique()
#st.write(unique_categories)
category_order = [['INLAND', '<1H OCEAN', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND']]
encoder = OrdinalEncoder(categories=category_order)
csvFile['ocean_proximity_score'] = encoder.fit_transform(csvFile[['ocean_proximity']])

#DROP UNNECCESSARY COLUMNS
csvFile.drop('ocean_proximity', axis=1, inplace=True)
csvFile.drop('latitude', axis=1, inplace=True)
csvFile.drop('longitude', axis=1, inplace=True)

#DECLARE X AND Y
X = csvFile.drop('median_house_value', axis=1)
y = csvFile['median_house_value']

# SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#FILL MISSING VALUES WITH MEAN
#Due to the data having NAN values, we have to Impute data to be able to use predict method.
#We do this by using the SimpleImputer with the mean strategy.
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

#INPUT
housing_median_age = st.number_input('Housing Median Age', value=None, step=1,  placeholder='Type a number...')
total_rooms = st.number_input('Total Rooms', value=None, step=1, placeholder='Type a number...')
total_bedrooms = st.number_input('Total Bedrooms', value=None, step=1, placeholder='Type a number...')
population = st.number_input('Population', value=None, step=1, placeholder='Type a number...')
households = st.number_input('Households', value=None, step=1, placeholder='Type a number...')
median_income = st.number_input('Median Income', value=None, step=1, placeholder='Type a number...')
options = {
     'INLAND': 0,
     '<1H OCEAN': 1,
     'NEAR OCEAN': 2,
     'NEAR BAY': 3,
     'ISLAND': 4
}
ocean_proximity = st.selectbox('Ocean Proximity', list(options.keys()))
ocean_proximity_score = options[ocean_proximity]


calculate_button = st.button("Get House Price!")
if calculate_button:
    data = []
    data.append( {'housing_median_age' : housing_median_age, 
                  'total_rooms' : total_rooms, 
                  'total_bedrooms' : total_bedrooms, 
                  'population' : population, 
                  'households' : households, 
                  'median_income' : median_income,
                  'ocean_proximity_score' : ocean_proximity_score
                  })
    
    df = pd.DataFrame(data)
    pred_price = int(loaded_random_forest_model.predict(df))
    st.write("The predicted house price is: $", str(pred_price))