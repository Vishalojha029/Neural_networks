# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv("Social_Network_Ads.csv")
X = dataset.iloc[:, [2, 3]].values
Y = dataset.iloc[:, 4].values
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train=X_train.T
X_test=X_test.T
y_train=y_train.reshape(300,1)
y_test=y_test.reshape(100,1)
y_train=y_train.T
y_test=y_test.T


n_x=(X_train).shape[0]
n_h=int(input("enter the no of units you want in 1 layer Neural network :"))
n_y=(y_train).shape[0]
        
def Parametrers_of_nn(n_x,n_h,n_y):
    
    W1=np.random.randn(n_h,n_x)*0.001 
    b1=np.zeros((1,1))
    W2=np.random.randn(n_y,n_h)*0.001
    b2=np.zeros((n_y,1))
     
    parameters={"W1":W1,"b1":b1,"W2":W2,"b2":b2}
    return parameters
parameters=Parametrers_of_nn(n_x,n_h,n_y)
def sigmoid(x):
    x=1/(1+np.exp(-x))
    return x

def forward_propagation(X_train,y_train,parameters):
    
    W1=parameters["W1"]
    W2=parameters["W2"]
    b1=parameters["b1"]
    b2=parameters["b2"]
    
    Z1=np.dot(W1,X_train)+b1
    A1 = np.tanh(Z1)
    Z2=np.dot(W2,A1)+b2
    A2=sigmoid(Z2)
    results={"Z1":Z1,"Z2":Z2,"A1":A1,"A2":A2}
    
    return results
results=forward_propagation(X_train,y_train,parameters)

def backward_propagation(X_train,y_train,results,parameters):
    m=y_train.shape[1]
    A1=results["A1"]
    A2=results["A2"]
    Z1=results["Z1"]
    Z2=results["Z2"]
    W1=parameters["W1"]
    W2=parameters["W2"]
    b1=parameters["b1"]
    b2=parameters["b2"]
    dZ2=A2-y_train
    dW2=(np.dot(dZ2,A1.T))/m
    db2=(np.sum(dZ2,axis=1,keepdims=True))/m
    dZ1=np.dot(W2.T,dZ2)*(1 - np.power(A1, 2))
    dW1 = np.dot(dZ1,X_train.T)/m
    db1 = (1/m)*np.sum(dZ1,axis=1,keepdims=True)
    
    diff_param={"dW2":dW2,"dW1":dW1,"dZ1":dZ1,"dZ2":dZ2,"db1":db1,"db2":db2}
    
    return diff_param

diff_param=backward_propagation(X_train,y_train,results,parameters)

def updateParametrers_of_nn(X_train,y_train,parameters,diff_param):
    learning_rate=0.01
    W1=parameters["W1"]
    W2=parameters["W2"]
    b1=parameters["b1"]
    b2=parameters["b2"]
    
    dW1=diff_param["dW1"]
    dW2=diff_param["dW2"]
    db1=diff_param["db1"]
    db2=diff_param["db2"]
    
    W1=W1-learning_rate*dW1
    W2=W2-learning_rate*dW2
    b1=b1-learning_rate*db1
    b2=b2-learning_rate*db2
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    
    return parameters
    
def Cost_function(X_train,y_train,results):
    A1=results["A1"]
    A2=results["A2"]
    Z1=results["Z1"]
    Z2=results["Z2"]
    
    m=y_train.shape[1]
    cost1=(np.multiply(np.log(A2),y_train) +np.multiply(np.log(1-A2),(1-y_train)))/m
    cost=-1*np.sum(cost1)
    cost=np.squeeze(cost)
    
    return cost
        
def Artificial_Neural_network(X_train,y_train,parameters):
    for i in range(10000):
        results=forward_propagation(X_train,y_train,parameters)
        cost=Cost_function(X_train,y_train,results)
        diff_param=backward_propagation(X_train,y_train,results,parameters)
        parameters=updateParametrers_of_nn(X_train,y_train,parameters,diff_param)
        if(i%10000==0):         
            print ("Cost after iteration %i: %f" %(i, cost))
    return parameters
    
parameters=Artificial_Neural_network(X_train,y_train,parameters)
results=forward_propagation(X_train,y_train,parameters)
def classifier(parameters,results):
    y_pred=results["A2"]
    for i in range(300):
        if(y_pred[0][i]>=0.5):
            y_pred[0][i]=1
        else:
            y_pred[0][i]=0
            
    for i in range(300):
        print(y_pred[0][i])
            
    y_pred=y_pred.reshape(300,1)
    return y_pred
y_pred=classifier(parameters,results)
y_train=y_train.reshape(300,1)        
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train, y_pred)
