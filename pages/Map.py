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

#INTRO
st.title('Map over California')

#LOAD IN CSV FILE
csvFile = pd.read_csv('housing.csv')

st.write('Under construction!')
#st.write(csvFile)

#X = csvFile['latitude']
#y = csvFile['longitude']
#st.write(X, y)
#data = []