"""
A simple logistic regression (with gradient descent) model working on a dataset of 4298563 instances.
The model gave an accuracy of 63.5% - training and 68.37% - testing with iterations = 500 and learning rate = 0.05.
Similar results obtained for all possible combinations of iterations = 100 and learning rate = 0.1.
The model proves to be underfitting for our dataset due to being too simple for the data
(inference from similar values for training and test accuracies, Jtrain & jtest - similar)
   
 """

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#preprocessing
data = pd.read_csv('data.csv')
print(data.shape)

"""
VECTOR SHAPES
X = n x m
Y = 1 x m
w = n x 1
Z = 1 x m
A = 1 x m
m: the number of datasets
n: the number of features

"""

#preprocessing-2
train_set_X = data.iloc[:4000000, :-1].values.transpose()
train_set_Y = data.iloc[:4000000, 61].values.reshape(1,train_set_X.shape[1])
test_set_X = data.iloc[4000000: , :-1].values.transpose()
test_set_Y = data.iloc[4000000:, 61].values.reshape(1,test_set_X.shape[1])
print(test_set_X.shape)
print(test_set_Y.shape)

#helper function: sigmoid
def sigmoid(z):
  return 1.0 / (1 + np.exp(-z))

#helper function: parameter initialization
def initialize(dim):
    """ intializes vector w of shape(dim,1) and b as 0 """
    w = np.zeros((dim,1))
    b = 0
    return w, b  

#helper function: forward & back propagation
def propagate(w, b, X, Y):
    m = X.shape[1]
   
    #FORWARD
    Z = np.dot(w.transpose(), X) + b
    #print(Z.shape)
    A = sigmoid(Z)
    #print(A.shape)
    cost = -(np.dot(Y, np.log(A).transpose())+ np.dot((1-Y), np.log(1-A).transpose()))
   
    #BACKWARD
    dZ = A - Y
    dw = np.dot(X, dZ.transpose())/m
    db = np.sum(dZ)/m
   
    gradients = {"dw" : dw,
                 "db" : db}
    return gradients, cost

"""w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])
grads, cost = propagate(w, b, X, Y)
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print ("cost = " + str(cost))"""

#helper function: optimize via Gradient Descent
def gradientDescent(w, b ,X, Y, iterations, alpha, print_cost):
    costs  = []
    for i in range(iterations):
       
        #forward & back propagation
        grads, cost = propagate(w, b ,X, Y)
        dw = grads["dw"]
        db = grads["db"]
       
        #weight updation
        w = w - alpha*dw
        b = b - alpha*db
       
        if (i%100) == 0:
            costs.append(cost)
           
        if print_cost and (i%100==0):
            print("the cost after "+ str(i) +" iterations is: "+ str(cost))
           
    params = {"w" : w,
              "b" : b}
    grads = {"dw" : dw,
             "db" : db}
    return params, grads, costs

#helper function: to predict new values
def predict(w, b, X):
    """ w is a (n x 1) matrix, X is a (m x n) matrix """
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)  #no idea why this exist?
   
    Z = np.dot(w.transpose(), X)+b
    A = sigmoid(Z)
   
    for i in range(A.shape[1]):
        if A[0][i]<=0.5:
            Y_prediction[0][i] = 0
        else:
            Y_prediction[0][i] = 1
   
    return Y_prediction  

#model
def model(X_train, Y_train, X_test, Y_test, iterations, alpha, print_cost):
   
    #initialize
    w, b = initialize(X_train.shape[0])
   
    #fw & bp
    params, grads, costs = gradientDescent(w, b, X_train, Y_train, iterations, alpha, print_cost)
   
    #retreive parameters
    w = params["w"]
    b = params["b"]
   
    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)
   
    #print accuracies
   # print(Y_prediction_train.shape)
    #print(Y_train.shape)
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
   
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train" : Y_prediction_train,
         "w" : w,
         "b" : b,
         "learning_rate" : alpha,
         "iterations": iterations}
   
    return d

d = model(train_set_X, train_set_Y, test_set_X, test_set_Y, iterations = 500, alpha = 0.05, print_cost = True)


"""
OUTPUT :    the cost after 0 iterations is: [[2772588.72223999]]
            the cost after 100 iterations is: [[2636107.19196553]]
            the cost after 200 iterations is: [[2624253.75377114]]
            the cost after 300 iterations is: [[2623146.61367674]]
            the cost after 400 iterations is: [[2623040.22673616]]
            train accuracy: 63.58705 %
            test accuracy: 68.30719144703127 %
"""

