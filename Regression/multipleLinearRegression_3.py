#MULTIPLE LINEAR REGRESSION

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:,4].values


#ENCODING CATEGORICAL DATA
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#AVOIDING DUMMY VARIABLE TRAP
#some columns representing indices may be treated as features for model
X = X[:, 1:]

#SPLITTING
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#FITTING MULTIPLE REGRESSION TO TRAINING SET
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#can't plot our predictions because it'll be 5-dimensional(5 features)

#PREDICTING TEST SET RESULTS
Y_pred = regressor.predict(X_test)

#BACKWARD ELIMINATIONH FOR OPTIMAL MODEL
#import statsmodels.formula.api as sm
#X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
#X_opt = X[:, [0,1,2,3,4,5]]











