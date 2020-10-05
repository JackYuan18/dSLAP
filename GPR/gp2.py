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
#from sklearn.neighbors import KDTree
from scipy import spatial
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
def k1(kernel,x,y):
    return kernel(x,y)
#            C = np.zeros((len(x),len(y)))
#            for ind,i in enumerate(x):
#                for jnd,j in enumerate(y):
#                    C[ind,jnd] = kernel(i,j)   
#            return C
class gp:
    def __init__(self, kernel, mean_fun,sigma):
        self.kernel=kernel
        self.mean_fun=mean_fun
        self.sigma=sigma
        self.mu_prep=None
        self.K_inv=None
        self.mu_prep=None
        self.K_inv=None
        self.kdt=None
        self.data=dict()
        self.data_keys=[]
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
    
    def gp_prep_local(self,train,train_f):
        y=iter(train_f)
        self.data_keys=[]
        for x in train:
            self.data[str(x)]=next(y)
            self.data_keys.append(x)
        try:
            self.kdt=spatial.KDTree(np.array(self.data_keys))
        except:
            self.kdt=spatial.KDTree(np.array([self.data_keys]).T)
       
    
    def predict(self,test,train):     
#        self.gp_prep(train, train_f)
        k_star=k(self.kernel,test,train)
        if self.mean_fun is None:
            mu = np.matmul(k_star,self.mu_prep)
        else:
            mu = np.matmul(k_star,self.mu_prep)+self.mean_fun(test)
        cov = k(self.kernel,test,test)-np.matmul(np.matmul(k_star,self.K_inv),k_star.T)
        return mu,cov+self.sigma**2*np.eye(len(test))

    def predict_local(self,test):     
#        self.gp_prep(train, train_f)
        ind=self.kdt.query(test,p=2)
        z_train=np.array([self.kdt.data[i] for i in ind[1]])
        try:
            y=np.array([self.data[str(z)] for z in z_train])
        except:          
            y=np.array([self.data[str(float(z))] for z in z_train])
        try:
            k_star=np.array([[k1(self.kernel,test[i],z_train[i])] for i in range(len(test))])
        except:
            k_star=np.array([k1(self.kernel,test[i],z_train[i]) for i in range(len(test))]).T
        if self.mean_fun is None:
            mu = k_star*(k1(self.kernel,1,1)+self.sigma**2)**(-1)*y
        else:
            mu = k_star*(k1(self.kernel,1,1)+self.sigma**2)**(-1)*(y-self.mean_fun(z_train))+self.mean_fun(test)
        cov = k1(self.kernel,1,1)-k_star**2*(k1(self.kernel,1,1)+self.sigma**2)**(-1)+self.sigma**2
        return mu,cov
    
    
    def predict_local_MultiD(self,test,d):     
#        self.gp_prep(train, train_f)

        ind=self.kdt.query(test,p=2)
       
        z_train=self.kdt.data[ind[1]]
#        print('z_train=',z_train)
        
        y=self.data[str(z_train)][d] 
#        except:          
#            y=self.data[str(z_train)][d]
##        print('y=',y)
#        try:
        k_star=k1(self.kernel,test,z_train)
#        except:
#            k_star=k1(self.kernel,test,z_train)
#        print('k_star=',k_star)
        if self.mean_fun is None:
            mu = k_star*(k1(self.kernel,1,1)+self.sigma**2)**(-1)*y
        else:
            mu = k_star*(k1(self.kernel,1,1)+self.sigma**2)**(-1)*(y-self.mean_fun(z_train))+self.mean_fun(test)
        cov = k1(self.kernel,1,1)-k_star**2*(k1(self.kernel,1,1)+self.sigma**2)**(-1)+self.sigma**2
#        print(cov)
        return mu,cov
    
    def predict_local_MultiD_got_train(self,test,z_train,d):     
#        self.gp_prep(train, train_f)
#        ind=self.kdt.query(test,p=2)
#        z_train=np.array([self.kdt.data[i] for i in ind[1]])
#        try:
        y=self.data[str(z_train)][d] 
#        except:          
#            y=self.data[str(z_train)][d]
##        print('y=',y)
#        try:
#            k_star=k1(self.kernel,test,z_train)
#        except:
        k_star=k1(self.kernel,test,z_train)
        if self.mean_fun is None:
            mu = k_star*(k1(self.kernel,1,1)+self.sigma**2)**(-1)*y
        else:
            mu = k_star*(k1(self.kernel,1,1)+self.sigma**2)**(-1)*(y-self.mean_fun(z_train))+self.mean_fun(test)
        cov = k1(self.kernel,1,1)-k_star**2*(k1(self.kernel,1,1)+self.sigma**2)**(-1)+self.sigma**2
        return mu,cov
#    def predict_again(self,test,train):
#        k_star=k(self.kernel,test,train)
#        if self.mean_fun is None:
#            mu = np.matmul(k_star,self.mu_prep)
#        else:
#            mu = np.matmul(k_star,self.mu_prep)+self.mean_fun(test)
#        cov = k(test,test)-np.matmul(np.matmul(k_star,self.K_inv),k_star.T)
#        return mu,covr1
    

        
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
    t=30
    train  = np.linspace(0,5,t)
    sigma = 0.1
    
    #train[1] = train[1]-1
    #train  = np. random.unifospatial.KDTree(np.array([self.data_keys]).T)rm(-5,5,5)
#    test = np.stack((np.linspace(0,5,20),np.linspace(0,5,20))).T
    test=np.array([np.linspace(0,5,t*2)]).T
    test2=np.linspace(0,5,t*4)
#    vsquare=np.vectorize(square3)
    train_f = np.array([np.sin(x)for x in train])
    #train_f = np.random.uniform(-2,2,num)

    
    sigma_f = 1
    l = 0.4
    kernel=lambda x,y:np.exp(-(x-y)**2)*sigma_f
    
    gpr=gp(kernel,None,sigma)
#    mu_prep,cov_prep=gp_prep(kernel,train, train_f,sigma)
    
    
    plt.figure()
    
    #training
    gpr.gp_prep_local(train,train_f)
    
    #testing
    mu,cov=gpr.predict_local(test)
    
    var_=sigma**2-sigma**4/((1+sigma**2)*t)

    #test prediction
    plt.plot(test,mu.T)
    plt.scatter(train,train_f)
    
    std2 = np.sqrt(cov)*2
    
    plt.plot(test,mu.T+std2.T,'--b')
    plt.plot(test,mu.T-std2.T,'--b')
    

    

    


