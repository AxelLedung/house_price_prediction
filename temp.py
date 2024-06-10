X = csvFile['ocean_proximity'].astype(str)
y = data[:, :-1].astype(str)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

ordinal_encoder = OrdinalEncoder()
ordinal_encoder.fit(X_train)
X_train = ordinal_encoder.transform(X_train)
X_test = ordinal_encoder.transform(X_test)

label_encoder = LabelEncoder()
label_encoder.fit(y_train)
y_train = label_encoder.transform(y_train)
y_test = label_encoder.transform(y_test)

model = LinearRegression()

model.fit(X_train, y_train)

yhat = model.predict(X_test)

accurary = accuracy_score(y_test, yhat)
st.write(accurary)
#x = np.array(csvFile["ocean_proximity"])
#y = np.array(csvFile["median _house_value"])

st.write(csvFile)

data_ordinal = asarray([['INLAND'], ['<1H OCEAN'], ['NEAR BAY'], ['NEAR OCEAN'], ['ISLAND']])
#data_ordinal = np.asarray(csvFile["ocean_proximity"])
st.write(data_ordinal)

ordinal_encoder = OrdinalEncoder()
# Code below can be ran if you want to specify how the categories should be ran. Test the code. 
# ordinal_encoder = OrdinalEncoder(categories = [['red', 'green', 'blue']])  
result_ordinal = ordinal_encoder.fit_transform(data_ordinal)
st.write(result_ordinal)

st.write(ordinal_encoder.categories_)


#x = np.array(csvFile["ocean_proximity"])
#y = np.array(csvFile["median _house_value"])

#Plotting the data
#fig_data, ax_data = plt.subplots(figsize=(8, 4))
#ax_data.set_title('Data')
#ax_data.scatter(x, y)
#ax_data.set_xlabel('x')
#ax_data.set_ylabel('y')

#st.write(fig_data)
