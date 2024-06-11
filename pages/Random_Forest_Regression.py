import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer


import joblib

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


#FIT FOREST REGRESSION
forest.fit(X_train, y_train)
y_test.plot.box()


#FILL MISSING VALUES WITH MEAN
#Due to the data having NAN values, we have to Impute data to be able to use predict method.
#We do this by using the SimpleImputer with the mean strategy.
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

#PREDICT FOREST REGRESSION
y_test_pred_forest = forest.predict(X_test)
RMSE_test_data = mean_squared_error(y_test, y_test_pred_forest, squared = False)
st.write("RMSE Test data: ", RMSE_test_data)
st.write("RMSE Test data / y mean: ",(RMSE_test_data)/(np.mean(y_test)))


#DUMP MODEL INTO FILE
random_forest_file = 'sav_files/ran_for_model.sav'
joblib.dump(forest, random_forest_file)#

#LOAD IN MODEL FROM FILE
random_forest_model = joblib.load(random_forest_file)
st.write("Score: ", random_forest_model.score(X_test, y_test))