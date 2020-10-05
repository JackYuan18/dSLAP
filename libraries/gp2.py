#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 11:23:53 2019

@author: zqy5086@AD.PSU.EDU
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import time
#mu = 0
#x = np.linspace(-5,5,50)
#
#C = np.zeros((len(x),len(x)))
#mu = mu*np.ones(len(x))
#for ind,i in enumerate(x):
#    for jnd,j in enumerate(x):
#        C[ind,jnd] = np.exp(-0.5*(i-j)**2)
#
#f = np.random.multivariate_normal(mu,C)    
#
#plt.plot(x,f)
#plt.plot(x,[2]*len(x))
#plt.plot(x,[-2]*len(x))
#plt.ylim(-3,3)


sigma_f = 1
l = 0.1
#def kernel(x,y):
#    return np.exp(-l*(x-y)**2)*sigma_f

#def kernel(x,y):
#    return np.exp(-l*np.linalg.norm(x-y))*sigma_f






#def k(x,y):
#        C = np.zeros((len(x),len(y)))
#        for ind,i in enumerate(x):
#            for jnd,j in enumerate(y):
#                C[ind,jnd] = kernel(i,j)   
#        return C



#def gp(kernel,test,train, train_f):
#    
#    def k(x,y):
#        C = np.zeros((len(x),len(y)))
#        for ind,i in enumerate(x):
#            for jnd,j in enumerate(y):
#                C[ind,jnd] = kernel(i,j)   
#        return C
#
#    mu = np.matmul(np.matmul(k(test,train),np.linalg.inv(k(train,train))),train_f)
#    cov = k(test,test)-np.matmul(np.matmul(k(test,train),np.linalg.inv(k(train,train)+np.eye(num)*sigma**2)),k(train,test))
#    return mu,cov
def k(kernel,x,y):
            C = np.zeros((len(x),len(y)))
            for ind,i in enumerate(x):
                for jnd,j in enumerate(y):
                    C[ind,jnd] = kernel(i,j)   
            return C
class gp:
    def __init__(self, kernel, mean_fun,sigma):
        self.kernel=kernel
        self.mean_fun=mean_fun
        self.sigma=sigma
        self.mu_prep=None
        self.K_inv=None
        self.mu_prep=None
        self.K_inv=None
    def gp_prep(self,train, train_f):
        K_inv=np.linalg.inv(k(self.kernel,train,train)+np.eye(len(train))*self.sigma**2)
#        print('K_inv: ',np.diag(K_inv))
        if self.mean_fun is None:
            mu_prep = np.matmul(K_inv,train_f)
        else:
            mu_prep = np.matmul(K_inv,train_f-self.mean_fun(train))
        self.mu_prep=mu_prep
        self.K_inv=K_inv
        return mu_prep,K_inv
    
    
    
    def predict(self,test,train):     
#        self.gp_prep(train, train_f)
        k_star=k(self.kernel,test,train)
        if self.mean_fun is None:
            mu = np.matmul(k_star,self.mu_prep)
        else:
            mu = np.matmul(k_star,self.mu_prep)+self.mean_fun(test)
        cov = k(self.kernel,test,test)-np.matmul(np.matmul(k_star,self.K_inv),k_star.T)
        return mu,cov+self.sigma**2*np.eye(len(test))
    
#    def predict_again(self,test,train):
#        k_star=k(self.kernel,test,train)
#        if self.mean_fun is None:
#            mu = np.matmul(k_star,self.mu_prep)
#        else:
#            mu = np.matmul(k_star,self.mu_prep)+self.mean_fun(test)
#        cov = k(test,test)-np.matmul(np.matmul(k_star,self.K_inv),k_star.T)
#        return mu,cov
    

        
    def gp_(self,mu_prep,cov_prep,test,train,kernel):
        def k(x,y):
            C = np.zeros((len(x),len(y)))
            for ind,i in enumerate(x):
                for jnd,j in enumerate(y):
                    C[ind,jnd] = kernel(i,j)   
            return C
       
        k_star=k(test,train)
        mu = np.matmul(k_star,mu_prep)
        cov = k(test,test)-np.matmul(np.matmul(k_star,cov_prep),k_star.T)
        return mu,cov
    
def square3(x):
    return x[0]**2+x[1]**2
if __name__=='__main__':
    
    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d')
    num = 5
    t=20
    train  = np.linspace(0,5,t)
    sigma = 0.1
    
    #train[1] = train[1]-1
    #train  = np. random.uniform(-5,5,5)
#    test = np.stack((np.linspace(0,5,20),np.linspace(0,5,20))).T
    test=np.linspace(0,5,t*2)
    test2=np.linspace(0,5,t*4)
#    vsquare=np.vectorize(square3)
    train_f = np.array([np.sin(x)for x in train])
    #train_f = np.random.uniform(-2,2,num)

    
    sigma_f = 1
    l = 0.1
    kernel=lambda x,y:np.exp(-np.linalg.norm(x-y))*sigma_f
    
    gpr=gp(kernel,None,sigma)
#    mu_prep,cov_prep=gp_prep(kernel,train, train_f,sigma)
    
    
    plt.figure()
    gpr.gp_prep(train,train_f)
    mu,cov=gpr.predict(test,train)
    var_=sigma**2-sigma**4/((1+sigma**2)*t)
#    print(var_,np.diag(cov))
#    mu,cov = gp(kernel,test,train,train_f)

#    K_inv = np.linalg.inv(k(train,train)+np.eye(num)*sigma**2)
#    K = k(train,train)+np.eye(num)*sigma**2
#    d = np.zeros(len(test))
#    for ind,t in enumerate(test):
#        distances = [abs(t-x) for x in train]
#        d[ind]=max(distances)
    
#    var_est = 2*(sigma_f**2 - sigma_f**2*np.exp(-l*(d))**2)
    
    #for i in range(3):
    #    f = np.random.multivariate_normal(mu,cov)    
    #    plt.plot(test,f)
#    plt.ylim(-10,10)
#    plt.scatter(train,train_f)
    plt.plot(test,mu)
    plt.scatter(train,train_f)
    
    std2 = np.sqrt(np.diag(cov))*2
    
    plt.plot(test,mu+std2,'--b')
    plt.plot(test,mu-std2,'--b')
    
    
  

