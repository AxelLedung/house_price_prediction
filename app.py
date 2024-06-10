import streamlit as st
import numpy as np
from numpy import asarray
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer


#INTRO
st.title("House Pricing Prediction")

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
st.write('X (Entries WITHOUT Median House Value)', X)
st.write('Y (Median House Value)', y)

# SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# CORRELATION MATRIX
st.write("Correlation Matrix")
df = X.copy()
df['target'] = y

fig, ax = plt.subplots()
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, ax=ax, annot=True)
st.write(fig)

#DESCRIBE Y
st.write("Y Description")
st.write(y_train.describe())

#LINEAR REGRESSION
st.write("Linear Regression")
lin_reg = LinearRegression()

scores_lr = cross_validate(lin_reg, X_train, y_train, scoring = 'neg_mean_squared_error')["test_score"]
st.write('RMSE for each iteration:', np.sqrt(-scores_lr))
st.write('RMSE:', np.sqrt(np.mean(-scores_lr)))

#LASSO REGRESSION
st.write("Lasso Regression")
lasso = Lasso()
hyper_param_lasso = {'alpha':(0.01, 1, 2, 5, 10)}
lasso_reg = GridSearchCV(lasso, hyper_param_lasso, cv = 5)
# Fit the gridsearch to use the best hyperparameter in our cross validation
lasso_reg.fit(X_train, y_train)

st.write(lasso_reg.best_params_)

scores_lasso = cross_validate(lasso_reg, X_train, y_train, cv=5, scoring = 'neg_mean_squared_error')["test_score"]
st.write('RMSE for each iteration:', np.sqrt(-scores_lasso))
st.write('RMSE:', np.sqrt(np.mean(-scores_lasso)))

#RANDOM FOREST REGRESSION 
st.write("Random Forest Regression")
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
print(np.mean(y_test))
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