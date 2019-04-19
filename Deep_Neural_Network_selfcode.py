# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 15:15:14 2018

@author: vishal
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv("diabetes.csv")
X = dataset.iloc[:, 0: 8].values
Y = dataset.iloc[:, 8].values
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_train=X_train.T
X_test=X_test.T
y_train=y_train.reshape(y_train.size,1)
y_test=y_test.reshape(y_test.size,1)
no_of_test_set=y_test.shape[0]
no_of_training_set=y_train.shape[0]
y_train=y_train.T
y_test=y_test.T

no_of_layers=int(input("No of layers You want in neural network : "))
No_of_units=[None]*(no_of_layers+1)
No_of_units[0]=X_train.shape[0]
for i in range(1,no_of_layers+1):
    No_of_units[i]=int(input("no of units in layer {} : ".format(i)))
    
def parameters_NN(no_of_layers,No_of_units):
    W=[None]*(no_of_layers+1)
    b=[None]*(no_of_layers+1)
    for i in range(1,no_of_layers+1):
        W[i]=np.random.randn(No_of_units[i],No_of_units[i-1])*0.1
        b[i]=np.zeros((No_of_units[i],1))
        
    parameters={"W":W,"b":b}
    return parameters
parameters=parameters_NN(no_of_layers,No_of_units)

def sigmoid(x):
    x=1/(1+np.exp(-x))
    return x

def forward_propagation(X_train,parameters,no_of_layers):
    
    W=parameters["W"]
    b=parameters["b"]
    Z=[None]*(no_of_layers+1)
    A=[None]*(no_of_layers+1)
    A[0]=X_train
    for i in range(1,no_of_layers+1):
        if(i<no_of_layers):
            Z[i]=np.dot(W[i],A[i-1])+b[i]
            A[i]=np.tanh(Z[i])
        else:
            Z[i]=np.dot(W[i],A[i-1])+b[i]
            A[i]=sigmoid(Z[i])
            results={"Z":Z,"A":A}

    return results

results=forward_propagation(X_train,parameters,no_of_layers)

def backward_propagation(y_train,results,parameters,no_of_layers):
    m=y_train.size
    A=results["A"]
    W=parameters["W"]
    dZ=[None]*(no_of_layers+1)
    dW=[None]*(no_of_layers+1)
    db=[None]*(no_of_layers+1)
    dZ[no_of_layers]=A[no_of_layers]-y_train
    dW[no_of_layers]=(np.dot(dZ[no_of_layers],A[no_of_layers-1].T))/m
    db[no_of_layers]=(np.sum(dZ[no_of_layers],axis=1,keepdims=True))/m
    for i in range(no_of_layers-1,0,-1):
        
        
        dZ[i]=np.dot(W[i+1].T,dZ[i+1])*(1 - np.power(A[i], 2))
        dW[i] = np.dot(dZ[i],A[i-1].T)/m
        db[i] = (1/m)*np.sum(dZ[i],axis=1,keepdims=True)
    
    diff_param={"dW":dW,"dZ":dZ,"db":db}
    
    return diff_param

diff_param=backward_propagation(y_train,results,parameters,no_of_layers)

def updateParametrers_of_nn(parameters,diff_param,no_of_layers):
    learning_rate=0.1
    W=parameters["W"]
    b=parameters["b"]
    
    dW=diff_param["dW"]
    db=diff_param["db"]
    for i in range(1,no_of_layers+1):
        W[i]=W[i]-learning_rate*dW[i]
        b[i]=b[i]-learning_rate*db[i]
        
        parameters = {"W": W,
                      "b": b}
    
    
    return parameters
    
def Cost_function(y_train,results):
    A=results["A"]  
    m=y_train.size
    cost1=(np.multiply(np.log(A[no_of_layers]),y_train) +np.multiply(np.log(1-A[no_of_layers]),(1-y_train)))/m
    cost=-1*np.sum(cost1)
    cost=np.squeeze(cost)
    
    return cost
        
        
def deep_Neural_network(X_train,y_train,parameters):
    
    
    for i in range(10000):
        
        results=forward_propagation(X_train,parameters,no_of_layers)
        cost=Cost_function(y_train,results)
        diff_param=backward_propagation(y_train,results,parameters,no_of_layers)
        parameters=updateParametrers_of_nn(parameters,diff_param,no_of_layers)
        if(i%10000==0):
          
            print ("Cost after iteration %i: %f" %(i, cost))
           
    print ("Cost after iteration %i: %f" %(i, cost))
   
            
    
   
    #print("W1:"+W1+" W2:"+W2+" b1:"+b1+" b2:"+b2)
    return parameters
    
parameters=deep_Neural_network(X_train,y_train,parameters)
results=forward_propagation(X_test,parameters,no_of_layers)

def classifier(parameters,results):
    A=results["A"]
    y_pred=A[no_of_layers]
    for i in range(no_of_test_set):
        if(y_pred[0][i]>=0.5):
            y_pred[0][i]=1
        else:
            y_pred[0][i]=0
            
    for i in range(no_of_test_set):
        print(y_pred[0][i])
            
    y_pred=y_pred.reshape(no_of_test_set,1)
    return y_pred
y_pred=classifier(parameters,results)
y_train=y_train.reshape(no_of_training_set,1) 
y_test=y_test.reshape(no_of_test_set,1)       
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
