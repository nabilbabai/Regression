# Import Dependencies 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Import Dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# Create Train and Test set
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=1/3, random_state=0)

# Fitting Simple Linear Regressor to train set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting Results
y_pred = regressor.predict(X_test)

# Visualising Results for Training Set
plt.figure(1)
plt.scatter(X_train,y_train,c='red')
plt.plot(X_train, regressor.predict(X_train), c='blue')
plt.title('Experience Vs Salarly (Training Set)')
plt.xlabel('Experience')
plt.ylabel('Salary')


# Visualising Results for Test set
plt.figure(2)
plt.scatter(X_test,y_test,c='red')
plt.plot(X_train, regressor.predict(X_train), c='blue')
plt.title('Experience Vs Salarly (Test Set)')
plt.xlabel('Experience')
plt.ylabel('Salary')