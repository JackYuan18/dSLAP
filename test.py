#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 13:59:40 2019

@author: zqy5086@AD.PSU.EDU
"""

mu,cov=gp(mu_prep,cov_prep,train,train,kernel,car_true)

t_end=3
t_step=0.1
t_start=0
car_x=np.array([0.,0.,0])


i=0
test=[]

print('data initialization')
while t_start<t_end:
    test.append(car_x)
    car_xdot=mu[i]
    car_x = car_x+t_step*car_xdot
    
    i=i+1
    t_start=t_start+t_step
test=np.stack(test)

plt.plot(test[:,0],test[:,1])
plt.plot(train[:,0],train[:,1])
