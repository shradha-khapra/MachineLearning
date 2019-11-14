#Data Preprocessing - getting data ready to train our model

#IMPORTING LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#IMPORTING THE DATASET
dataset = pd.read_csv('Data.csv')
#data.iloc[<row-selection>, <col-selection>]
X = dataset.iloc[:, :-1] #'-1' represents last column
Y = dataset.iloc[:,3]

#SPLITTING THE DATASET INTO TRAINING & TEST SET (80-20 split)
#using the test data for cross-validation
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
#random_state=some-no allows same result to be reproduced 

#FEATURE SCALING
"""from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""
#Here the fit method, when applied to the training dataset,learns the model parameters 
#(for example, mean and standard deviation).
#We apply fit on the training dataset and use the transform method on both 
#- the training dataset and the test dataset. Thus the training as well as the test dataset
# are then transformed(scaled) using the model parameters that were learnt on applying the fit method 
#the training dataset.




