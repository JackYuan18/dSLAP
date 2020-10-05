#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 15:08:02 2020

@author: zqy5086@AD.PSU.EDU
"""

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
from math import sin, cos
import time
#from sklearn.neighbors import KDTree
from scipy import spatial
from sklearn.gaussian_process.kernels import RBF

from scipy.linalg import inv
from scipy.linalg.blas import sgemm

class gp:
    def __init__(self, kernel):
        self.k=kernel
        self.K_inv=None
        self.test=None
        self.mu_t=None
        self.mu_g=None
        self.C_g=None
        self.C_t=None
        self.dim=None
        self.var_g=None
    def gp_prep(self,test,dim=1):
        
        self.K_inv=np.diag(np.linalg.inv(self.k(test)))
        self.test=test
        self.dim=dim
        self.C_g=np.diag(self.k(test))
        self.mu_g=np.zeros((len(test),dim))
    def gp_prep_(self,test,dim=1):
        
        self.K_inv=np.linalg.inv(self.k(test))
        self.test=test
        self.dim=dim
        self.C_g=self.k(test)
        self.mu_g=np.zeros((len(test),dim))

        
    
        
    def predict(self,train,y):  
#        print(len(train))
        Kt=self.k(train,self.test)
#        Jt=sgemm(1.0, Kt,self.K_inv)
        Jt=Kt*self.K_inv
        B=np.ones((len(train),))*self.k(train[1])[0]-(Jt*Kt).sum(-1)#sgemm(1.0,Jt,Kt, trans_b=True)
#        print(self.k(train[1]))
        CJ=Jt*self.C_g
#        CJ=sgemm(1.0,self.C_g,Jt,trans_b=True)
        
        self.C_t=B+(Jt*CJ).sum(-1)
        
#        self.C_t=B+sgemm(1.0,Jt,CJ,trans_b=True)
        Gt=CJ.T*1/(self.C_t+0.001)
#        Gt=sgemm(1.0,CJ,1/(self.C_t+0.001))
        
        self.C_g=self.C_g-(Gt*CJ.T).sum(-1)#sgemm(1.0,Gt,CJ,trans_b=True)

#        self.var_g=np.diag(self.C_g)

        self.mu_t=sgemm(1.0,Jt,self.mu_g)
        self.mu_g=self.mu_g+sgemm(1.0,Gt,y-self.mu_t)


        return self.mu_g,self.C_g
    
    def predict_(self,train,y):  
        Kt=self.k(train,self.test)
        Jt=sgemm(1.0, Kt,self.K_inv)
        B=self.k(train)-sgemm(1.0,Jt,Kt, trans_b=True)

        CJ=sgemm(1.0,self.C_g,Jt,trans_b=True)
        self.C_t=B+sgemm(1.0,Jt,CJ)
        Gt=sgemm(1.0,CJ,inv(self.C_t+0.001*np.eye(20)))
        
        self.C_g=self.C_g-sgemm(1.0,Gt,CJ,trans_b=True)

        self.var_g=np.diag(self.C_g)

        self.mu_t=sgemm(1.0,Jt,self.mu_g)
        self.mu_g=self.mu_g+sgemm(1.0,Gt,y-self.mu_t)
        


        return self.mu_g,self.var_g
    
    
    
    
    def predict_MultiD(self,train,y): 
        Jt=np.matmul(self.k(train,self.test),self.K_inv)
        B=self.k(train)-np.matmul(Jt,self.k(self.test,train))
        self.C_t=B+np.matmul(np.matmul(Jt,self.C_g),Jt.T)
        Gt=np.matmul(np.matmul(self.C_g,Jt.T),inv(self.C_t))
        if any(np.linalg.eigvals(self.C_t)<0):
            raise NameError('Variance t less than 0')

        self.C_g=self.C_g-np.matmul(np.matmul(Gt,Jt),self.C_g)
        if any(np.linalg.eigvals(self.C_g)<0):
            raise NameError('Variance g less than 0')
        self.var_g=np.diag(self.C_g)

        self.mu_t=np.matmul(Jt,self.mu_g)
        self.mu_g=self.mu_g+np.matmul(Gt,y-self.mu_t)
#        w,v=np.linalg.eig(self.C_g)
#        print(train)l
#        print(np.min(w))
#        print(np.min(self.var_g))
#        return self.mu_g,np.diag(self.C_g)
        

        

    
    
if __name__=='__main__':
    l=100
    sigma_f=0.0001
    kernel = sigma_f* RBF(0.1)
    
#    np.random.seed(10) 
    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d')
    s=4
    h=10
    eta=lambda x:sin(h*(x[0]+x[1]))+cos(h*(x[0]-x[1]))
    eta2=lambda x:sin(h*([0]+x[1]))-cos(h*(x[0]-x[1]))
    x=np.linspace(0,1,s)
    y=np.linspace(0,1,s)
    test=None
    XX,YY=np.meshgrid(x,y)
    for xx in x:
        for yy in y:
            z=np.array([[xx,yy]])
            if test is None:
                test=z
            else:
                test=np.vstack((test,z))
    
    gpr=gp(kernel)
    gpr.gp_prep(test,2)
    
    num = 20
    t=30
    
    for i in range(15):
        train=np.random.rand(num,2)
        Y=None
        for j in range(num):
            if Y is None:
                Y=np.array([[eta(train[j,:]),eta2(train[j,:])]])
            else:
                Y=np.vstack((Y,np.array([[eta(train[j,:]),eta2(train[j,:])]])))
        start=time.time()
        gpr.predict(train,Y)
#        print(time.time()-start)
        
        ZZ=[]
        for mu in gpr.mu_g[:,0]:
            ZZ.append(mu)
        figure1=plt.figure(1)
        plt.contourf(XX.T,YY.T,np.reshape(ZZ,(s,s)))
#        plt.colorbar()
        plt.scatter(train[:,0],train[:,1])
        
        
   
    TT=[]
    for i in range(len(test)):
        TT.append(eta(test[i,:]))
    figure4=plt.figure(4)
    plt.contourf(XX.T,YY.T,np.reshape(TT,(s,s)))
    plt.colorbar()
    ZZ=[]
#        print(np.min(gpr.var_g))
    for var in gpr.var_g:
        ZZ.append(var)
    figure2=plt.figure(2)
    plt.contourf(XX.T,YY.T,np.reshape(ZZ,(s,s)))
    plt.colorbar()
    plt.scatter(train[:,0],train[:,1])
    plt.title('var')
    
    

    


