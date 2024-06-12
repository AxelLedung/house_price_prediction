import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
import joblib
import os

#INTRO
st.title("Linear Regression without Ocean Proximity")

#LOAD IN CSV FILE
csvFile = pd.read_csv('housing.csv')

#DROP UNNECCESSARY COLUMNS
csvFile.drop('ocean_proximity', axis=1, inplace=True)
csvFile.drop('latitude', axis=1, inplace=True)
csvFile.drop('longitude', axis=1, inplace=True)

X = csvFile.drop('median_house_value', axis=1)
y = csvFile['median_house_value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#FILL MISSING VALUES WITH MEAN
#Due to the data having NAN values, we have to Impute data to be able to use predict method.
#We do this by using the SimpleImputer with the mean strategy.
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

#RANDOM FOREST REGRESSION
forest = RandomForestRegressor()
param_grid = { 
    'n_estimators': [10, 100, 120],
    'max_depth' : [None, 10],
}
forest_reg = GridSearchCV(forest, param_grid=param_grid, cv= 5)
# Fit the gridsearch to use the best hyperparameter in our cross validation
forest_reg.fit(X_train, y_train)

st.write(forest_reg.best_params_)
scores_forest = cross_validate(forest_reg, X_train, y_train, cv=5, scoring = 'neg_mean_squared_error')["test_score"]
st.write('RMSE for each iteration:', np.sqrt(-scores_forest))
st.write('RMSE:', np.sqrt(np.mean(-scores_forest)))
