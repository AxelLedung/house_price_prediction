import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
import joblib
import os

#INTRO
st.title("Lasso Regression")

#LOAD IN TRAINED MODEL FROM SAV FILE
loaded_lasso_reg_file = 'sav_files/las_reg_model.sav'
loaded_lasso_reg_model = None

#CHECK IF THE MODEL FILE EXISTS OTHERWISE CREATE AN EMPTY FILE
os.makedirs(os.path.dirname(loaded_lasso_reg_file), exist_ok=True)

if not os.path.exists(loaded_lasso_reg_file):
    with open(loaded_lasso_reg_file, 'w') as file:
        file.write('')
    st.warning('Model file was not found. An empty file has been created at ' + loaded_lasso_reg_file + '. Please wait while I create a new model for you.')
elif os.path.getsize(loaded_lasso_reg_file) == 0:
    st.warning('Model file at ' + loaded_lasso_reg_file + ' is empty. Please wait while I create a new model for you.')
else:
    try:
        loaded_lasso_reg_model = joblib.load(loaded_lasso_reg_file)
        st.success('Model loaded successfully!')
    except EOFError:
        st.warning('The model file at' + loaded_lasso_reg_file + 'is corrupted or incomplete.')
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
if loaded_lasso_reg_model is not None:
    st.write("Score: ", loaded_lasso_reg_model.score(X_test, y_test))

#LASSO REGRESSION
lasso = Lasso()
hyper_param_lasso = {'alpha':(0.01, 1, 2, 5, 10)}
lasso_reg = GridSearchCV(lasso, hyper_param_lasso, cv = 5)
# Fit the gridsearch to use the best hyperparameter in our cross validation
lasso_reg.fit(X_train, y_train)

st.write(lasso_reg.best_params_)

scores_lasso = cross_validate(lasso_reg, X_train, y_train, cv=5, scoring = 'neg_mean_squared_error')["test_score"]
st.write('RMSE for each iteration:', np.sqrt(-scores_lasso))
st.write('RMSE:', np.sqrt(np.mean(-scores_lasso)))

#DUMP MODEL INTO FILE
joblib.dump(lasso_reg, loaded_lasso_reg_file)