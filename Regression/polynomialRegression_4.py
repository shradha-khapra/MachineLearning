#POLYNOMIAL REGRESSION

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

#SPLITTING DATA SET
"""from sklearn.cross_validation import train_test_split
X_tran, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)"""

#FEATURE SCALING(function already does it for us)
"""from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

#FITTING LINEAR REGRESSION MODEL
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, Y)


#FITTING POLYNOMIAL REGRESSION MODEL
from sklearn.preprocessing import PolynomialFeatures #creates features raised to some power aka new features
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X) #oth column created for x0 feature-bias
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, Y)

#VISUALIZING LINEAR REGRESSION
plt.scatter(X, Y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#VISUALIZING POLYNOMIAL REGRESSION
plt.scatter(X, Y, color = 'red')
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#predicting a new result with linear regression
lin_reg.predict(6.5)

#predicting a new result with polynomial regression
lin_reg2.predict(poly_reg.fit_transform(6.5))




