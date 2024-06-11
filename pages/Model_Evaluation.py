import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
import joblib

#INTRO
st.title("Model Evaluation")

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

#LOAD IN LINEAR REGRESSION MODEL FROM FILE
lin_reg_file = 'sav_files/lin_reg_model.sav'
lin_reg_model = joblib.load(lin_reg_file)

st.write("Linear Regression Score: ", lin_reg_model.score(X_test, y_test))


#LOAD IN LASSO REGRESSION MODEL FROM FILE
lasso_reg_file = 'sav_files/las_reg_model.sav'
lasso_reg_model = joblib.load(lasso_reg_file)

st.write("Lasso Regression Score: ", lasso_reg_model.score(X_test, y_test))

#LOAD IN RANDOM FOREST MODEL FROM FILE
random_forest_file = 'sav_files/ran_for_model.sav'
random_forest_model = joblib.load(random_forest_file)

st.write("Random Forest Score: ", random_forest_model.score(X_test, y_test))