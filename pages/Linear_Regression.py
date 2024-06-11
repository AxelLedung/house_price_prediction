import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
import joblib
import os


#INTRO
st.title("Linear Regression")

#LOAD IN TRAINED MODEL FROM SAV FILE
loaded_lin_reg_file = 'sav_files/lin_reg_model.sav'
loaded_lin_reg_model = None

#CHECK IF THE MODEL FILE EXISTS OTHERWISE CREATE AN EMPTY FILE
os.makedirs(os.path.dirname(loaded_lin_reg_file), exist_ok=True)

if not os.path.exists(loaded_lin_reg_file):
    with open(loaded_lin_reg_file, 'w') as file:
        file.write('')
    st.warning('Model file was not found. An empty file has been created at ' + loaded_lin_reg_file + '. Please wait while I create a new model for you.')
elif os.path.getsize(loaded_lin_reg_file) == 0:
    st.warning('Model file at ' + loaded_lin_reg_file + ' is empty. Please wait while I create a new model for you.')
else:
    try:
        loaded_lin_reg_model = joblib.load(loaded_lin_reg_file)
        st.success('Model loaded successfully!')
    except EOFError:
        st.warning('The model file at' + loaded_lin_reg_file + 'is corrupted or incomplete.')
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

#LOAD IN MODEL FROM FILE AND GET SCORE
if loaded_lin_reg_model is not None:
    st.write("Score: ", loaded_lin_reg_model.score(X_test, y_test))

#LINEAR REGRESSION
lin_reg = LinearRegression()

scores_lr = cross_validate(lin_reg, X_train, y_train, scoring = 'neg_mean_squared_error')["test_score"]
st.write('RMSE for each iteration:', np.sqrt(-scores_lr))
st.write('RMSE:', np.sqrt(np.mean(-scores_lr)))

lin_reg.fit(X_train, y_train)

#DUMP MODEL INTO FILE
joblib.dump(lin_reg, loaded_lin_reg_file)