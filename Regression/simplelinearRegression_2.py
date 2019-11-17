#SIMPLE LINEAR REGRESSION
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#DATA PREPROCESSING
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1]
Y = dataset.iloc[:,1]

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

#FEATURE SCALING
from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
#had to scale features due to anaconda version on ubuntu 16.04

#FITTING SCALAR LINEAR REGRESSION TO TRAINING SET
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#PREDICTING THE TEST SET RESULTS
y_pred = regressor.predict(X_test)

#VISUALIZING THE TRAINING SET RESULTS/PLOTTING
plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title("Salary vs Experience (Training Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

#VISUALIZING TEST SET RESULTS
plt.scatter(X_test, Y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')     #unique line
plt.title("Salary vs Experience (Test Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()



