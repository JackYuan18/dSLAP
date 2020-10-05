#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 12:15:46 2019

@author: zqy5086@AD.PSU.EDU
"""
from math import pi, sin
x=np.linspace(-pi,pi,100)
COV=[]
MU=[]
TRUTH=[]
par=[20,40,60,80,100]
p=10
sigma
    
X=x[:p]
y=np.sin(X)-0.05

TEST=np.linspace(-pi,pi,1000)
mu_prep,cov_prep=gp_prep(kernel,X, y,sigma,np.sin)
for test in TEST:
     
    mu,cov=gp(mu_prep,cov_prep,[test],X,kernel,np.sin)
    truth=sin(test)-0.05
    MU.append(mu)
    COV.append(cov)
    TRUTH.append(truth)
MU=np.stack(MU)
COV=np.stack(COV)
TRUTH=np.stack(TRUTH)

for i in range(3):
    plt.figure()
    plt.plot(TRUTH, color='blue',label='truth')
    plt.plot(MU[:,0]+2*np.sqrt(COV[:,0][:,0]), color='green', linestyle='dashed',label='var')
    plt.plot(MU[:,0]-2*np.sqrt(COV[:,0][:,0]),color='green',linestyle='dashed')
    plt.plot(MU[:,0],color='red',label='mean')
    plt.title(str(i))
    plt.legend()