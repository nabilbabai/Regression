# Import Dependencies
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
# Import Dataset 
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# Encoding Features
labelencoder = LabelEncoder()
X[:,3] = labelencoder.fit_transform(X[:,3])
onehotencoder = OneHotEncoder()
column = ColumnTransformer([("onehotencoder", onehotencoder,[3])],remainder = 'passthrough')
X = np.array(column.fit_transform(X), dtype = int)
X = X[:,1:]

# Create Train and Test sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

# Fiting Linear Regressor to the train set
regressor = LinearRegression()
regressor.fit(X_train,y_train)

# Prediction
y_pred = regressor.predict(X_test)