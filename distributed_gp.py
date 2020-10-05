#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 11:21:06 2019

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



def gp_1(kernel,test,train, train_f,sigma):
    
    def k(x,y):
        C = np.zeros((len(x),len(y)))
        for ind,i in enumerate(x):
            for jnd,j in enumerate(y):
                C[ind,jnd] = kernel(i,j)   
        return C
    num=len(train)
    mu = np.matmul(np.matmul(k(test,train),np.linalg.inv(k(train,train))),train_f)
    cov = k(test,test)-np.matmul(np.matmul(k(test,train),np.linalg.inv(k(train,train)+np.eye(num)*sigma**2)),k(train,test))
    return mu,cov
def k(x,y,kernel):
        C = np.zeros((len(x),len(y)))
        for ind,i in enumerate(x):
            for jnd,j in enumerate(y):
                C[ind,jnd] = kernel(i,j)   
        return C

def gp_prep(kernel,train, train_f,sigma,mean_fun=None):
    
    
    K_inv=np.linalg.inv(k(train,train,kernel)+np.eye(len(train))*sigma**2)
    if mean_fun is not None:
        mu_prep = np.matmul(K_inv,train_f-mean_fun(train))
    else:
#        print(K_inv)
        mu_prep = np.matmul(K_inv,train_f)
    return mu_prep,K_inv

def gp(mu_prep,cov_prep,test,train,kernel,mean_fun=None):
    
   
    k_star=k(test,train,kernel)
    if mean_fun is not None:
        mu = np.matmul(k_star,mu_prep)+mean_fun(test)
    else:
        mu = np.matmul(k_star,mu_prep)
    cov = k(test,test,kernel)-np.matmul(np.matmul(k_star,cov_prep),k_star.T)
    return mu,cov


def gp_(mu_prep,cov_prep,test,train,kernel):
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
    train1  = np.linspace(0,2,11)
    train2 = np.linspace(3,5,11)
    
    #train[1] = train[1]-1
    #train  = np. random.uniform(-5,5,5)
    test_old = np.linspace(0,5,101)
    truth=np.sin(test_old)
    
    nm_point=6
    
    M=np.linspace(0,5,nm_point)
    R=np.array(list(set(test_old).difference(set(M))))
    test=np.concatenate((M,R))
    vsquare=np.vectorize(square3)
    train_f1 = np.sin(train1)
    train_f2=np.sin(train2)
    #train_f = np.random.uniform(-2,2,num)

    sigma = 0.
    sigma_f = 1
    l = 0.3
    kernel=lambda x,y:np.exp(-l*np.linalg.norm(x-y))*sigma_f
    mu_prep1,cov_prep1=gp_prep(kernel,train1, train_f1,sigma)
    mu1,cov1=gp(mu_prep1,cov_prep1,test,train1,kernel)
    
    mu_prep2,cov_prep2=gp_prep(kernel,train2, train_f2,sigma)
    mu2,cov2=gp(mu_prep2,cov_prep2,test,train2,kernel)

    
    plt.scatter(test,mu1)
    plt.plot(test_old,truth,'red')
    
    std2 = np.sqrt(np.diag(cov1))*2
    
    plt.scatter(test,mu1+std2,c='g',label='two std')
    plt.scatter(test,mu1-std2,c='g')
    plt.plot(test_old,truth,'red',label='truth')
    plt.title('training data in [0,2]')
    
    plt.figure()
    plt.scatter(test,mu2)
    
    std2 = np.sqrt(np.diag(cov2))*2
    
    plt.scatter(test,mu2+std2,c='g',label='two std')
    plt.scatter(test,mu2-std2,c='g')
    plt.plot(test_old,truth,'red',label='truth')
    plt.title('training data in [3,5]')
    plt.figure()
    train=np.concatenate((train1,train2))
    train_f=np.sin(train)
    mu_prep,cov_prep=gp_prep(kernel,train, train_f,sigma)
    mu,cov=gp(mu_prep,cov_prep,test,train,kernel)
    plt.scatter(test,mu)
    
    std2 = np.sqrt(np.diag(cov))*2
    
    plt.scatter(test,mu+std2,c='g',label='two std')
    plt.scatter(test,mu-std2,c='g')
    plt.plot(test_old,truth,'red',label='truth')
    
    plt.title('regression using all data')
    
    test=train[2]
#    mu,cov=gp(mu_prep,cov_prep,[test],train,kernel)
    mu,cov=gp_1(kernel,[test],train, train_f,sigma)
    var_=sigma**2-sigma**4/(1+sigma**2)
    print(var_,cov)
    
    """
    plt.figure()
    SigmaMM=cov1[:nm_point,:nm_point]
    SigmaMR=cov1[nm_point:,:nm_point]
    SigmaRM=cov1[:nm_point,nm_point:]
    SigmaRR=cov1[nm_point:,nm_point:]
    
    var1=np.diag(SigmaMM)
    var2=np.diag(cov2[:nm_point,:nm_point])
    
    sigma_bcm=np.linalg.inv(SigmaMM)+np.linalg.inv(cov2[:nm_point,:nm_point])-np.linalg.inv(k(M,M,kernel))
    var_bcm=np.linalg.inv(sigma_bcm)
    mu_bcm=np.matmul(var_bcm,(np.matmul(np.linalg.inv(SigmaMM),mu1[:nm_point])+np.matmul(np.linalg.inv(cov2[:nm_point,:nm_point]),mu2[:nm_point])))
    
    
    
    mu_prep=np.matmul(np.linalg.inv(SigmaMM+var_bcm),mu_bcm-mu1[:nm_point])
    mu_R = np.matmul(SigmaMR,mu_prep)+mu1[nm_point:]
    cov_R = SigmaRR-np.matmul(np.matmul(SigmaMR,np.linalg.inv(SigmaMM+var_bcm)),SigmaRM)
    
    std2R = np.sqrt(np.diag(cov_R))*2
    std2M = np.sqrt(np.diag(var_bcm))*2
    std2=np.concatenate((std2M,std2R))
    mu_=np.concatenate((mu_bcm,mu_R))
    
    
    plt.scatter(test,mu_,label='mean')
    plt.plot(test_old,truth,'red',label='truth')

    
    plt.scatter(test,mu_+std2,c='g',label='two std')
    plt.scatter(test,mu_-std2,c='g')
    plt.legend()
    plt.title('agreement on 1/20 of test data')
    
    plt.figure()
    plt.scatter(test[:nm_point],mu_bcm)
    
    
#    mu_bcm=
#    plt.plot(test,np.sin(test)*2+np.cos(test)*2+test,'--r')
#    
#    plt.plot(test, mu+var_est,'--g')
#    plt.plot(test, mu-var_est,'--g')

#print(max(std2))

"""