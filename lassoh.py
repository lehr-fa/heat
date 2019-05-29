import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
from sklearn import linear_model
from sklearn import datasets

import heat as ht
import time


plt.style.use('fivethirtyeight')

#Load the diabetes dataset. In this case we will not be using a constant intercept feature
diabetes = datasets.load_diabetes()
X = diabetes.data
X = X / (np.linalg.norm(X,axis = 0)) #normalizing X in case it was not done before
y = diabetes.target.reshape(-1,1)

X = ht.array(X,dtype=float, split=0)
y = ht.array(y,dtype=float, split=0)



def soft_threshold(rho,lamda):
    '''Soft threshold function used for normalized data and lasso regression'''
    if rho < - lamda:
        return (rho + lamda)
    elif rho >  lamda:
        return (rho - lamda)
    else: 
        return 0.
    

def coordinate_descent_lasso(theta,X,y,lam = .01, num_iters=100):
    '''Coordinate gradient descent for lasso regression - for normalized data. 
    The intercept parameter allows to specify whether or not we regularize theta_0'''
    
    #Initialisation of useful values 
    #m,n = X.shape
    # normalization

    #Looping until max number of iterations
    for i in range(num_iters): 
        #Looping through each coordinate
        for j in range(X.shape[1]):
            #Vectorized implementation
            X_j = ht.expand_dims(X[:,j], axis=1)
            y_pred = X @ theta
            #tmp = (y - y_pred  + theta[j]*X_j)
            tmp = (y_pred)
            #print(X_j.T.shape, tmp.shape)
            rho = X_j.T @ (y - y_pred  + theta[j]*X_j)
            print(y.comm.rank, theta[j])
            #print(rho.resplit(None)) 
            theta[j] =  soft_threshold(rho.resplit(None), lam) 
            
    return None #theta # .flatten()


# Initialize variables
m,n = X.shape
initial_theta = ht.ones((n,1))
theta_list = list()
lamda = np.logspace(0,4,10)/10 #Range of lambda values

#Run lasso regression for each lambda
#print(y[:5,0])



for l in lamda:
    theta = coordinate_descent_lasso(initial_theta,X,y,lam = l, num_iters=100)
    # theta_list.append(theta.numpy().flatten())

"""
#Stack into numpy array
theta_lasso = np.stack(theta_list).T

#Plot results
n,_ = theta_lasso.shape
plt.figure(figsize = (12,8))

for i in range(n):
    plt.plot(lamda, theta_lasso[i], label = diabetes.feature_names[i])

plt.xscale('log')
plt.xlabel('Log($\\lambda$)')
plt.ylabel('Coefficients')
plt.title('Lasso Paths - Numpy implementation')
plt.legend()
plt.axis('tight')


plt.show()
"""
