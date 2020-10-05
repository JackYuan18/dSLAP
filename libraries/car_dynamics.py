#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 15:47:05 2019

@author: zqy5086@AD.PSU.EDU
"""
from state_space import state, d, to_string
from math import cos, sin,tan
import numpy as np
from wind import wind

def car_execution_plain(car_x,t_end,t_step,X_safe_tree,X_safe,record,Xfree):
    t_start=0
    train=[]
    train_f=[]
    while t_start<t_end:
        ind = X_safe_tree.query(car_x.reshape(1,-1),1,return_distance=False)
        coordinate = X_safe[ind[0][0]]
        key = to_string(coordinate)
        z_near = Xfree.table[key]
        u=z_near.u[0]
        
        x_dot=car_disturbance(car_x,u)
        car_x=car_x+t_step*x_dot
        t_start=t_start+t_step
        
        data=np.concatenate((car_x,np.array([u])))        
        record.append(car_x)
        train.append(data)
        train_f.append(x_dot)
    return record,train,train_f,car_x

def car_execution(car_x,t_end,t_step,X_temp_tree,X_temp,ball_obs_1,record,Xfree):
    t_start=0
    train=[]
    train_f=[]
    while t_start<t_end:
        x=state(car_x)
        if not d(ball_obs_1,x):
            return False,record,train,train_f,car_x
        
        ind = X_temp_tree.query(car_x.reshape(1,-1),1,return_distance=False)
        coordinate = X_temp[ind[0][0]]
        key = to_string(coordinate)
        z_near = Xfree.table[key]
        if z_near.unsafe:
            return False, record,train,train_f,car_x
        else:
            u=z_near.u[0]
            x_dot=car(car_x,u)
            car_x=car_x+t_step*x_dot
            t_start=t_start+t_step
            data=np.concatenate((car_x,np.array([u])))
            record.append(car_x)
            train.append(data)
            train_f.append(x_dot)
    return True,record,train,train_f,car_x

def car_execution_disturbance(car_x,t_end,t_step,X_temp_tree,X_temp,ball_obs_1,record,Xfree,control_history=None):
    t_start=0
    train=[]
    train_f=[]
    while t_start<t_end:
        x=state(car_x)
        if not d(ball_obs_1,x):
            return False,record,train,train_f,car_x
        
        ind1 = X_temp_tree.query(car_x.reshape(1,-1),1)
        if car_x[2]<0:
            ind2 = X_temp_tree.query((car_x+np.array([0,0,2])).reshape(1,-1),1)
        else:
            ind2 = X_temp_tree.query((car_x-np.array([0,0,2])).reshape(1,-1),1)
        if ind1[0][0][0]<ind2[0][0][0]:
            ind=ind1
        else:
            ind=ind2
#        ind = X_temp_tree.query(car_x.reshape(1,-1),1,return_distance=False)

        coordinate = X_temp[ind[1][0][0]]
        key = to_string(coordinate)
        z = Xfree.table[key]
        if z.unsafe:
            print('z_near unsafe')
        if control_history is None:
            u=z.u[0]
        else:            
            index=min(z.U,key=z.U.get)
            indices=[k for k in z.U if z.U[k]==z.U[index]]
            if len(indices)>1:
                new_ind=np.argmin([control_history[k] for k in indices])
                u=control_history[indices[new_ind]]
                control_history[indices[new_ind]]+=1
            else:
                u=z.U[index]
                z.U[index]+=1
                control_history[indices[new_ind]]+=1
         #################################   
        x_dot=car_disturbance(car_x,u)
        car_x=car_x+t_step*x_dot
        t_start=t_start+t_step
        data=np.concatenate((car_x,np.array([u])))
        record.append(car_x)
        train.append(data)
        train_f.append(x_dot-car(car_x,u))
    return True,record,train,train_f,car_x,control_history
def car(x,u):
    """
    x[0]=x
    x[1]=y
    x[2]=theta
    u[0]=v
    u[1]=phi    
       """  
    v = 0.02
    L = 0.015
#    x=x.reshape(1,3)
    X = np.zeros((3,))
    x=x.reshape((3,))
    X[0] = v*cos(x[2]*3.14)#-wind(x[0]*100,x[1]*100)*0.004
    X[1] = v*sin(x[2]*3.14)
    X[2] = v/L*tan(u)/3.14
    return X


def car_disturbance(x,u):
    """
    x[0]=x
    x[1]=y
    x[2]=theta
    u[0]=v
    u[1]=phi    
       """  
    v = 0.02
    L = 0.015
    X = np.zeros((3,))
#    print(x)
    x=x.reshape((3,))
    X[0] = v*cos(x[2]*3.14)-wind(x[0]*100,x[1]*100)*0.004
    X[1] = v*sin(x[2]*3.14)
    X[2] = v/L*tan(u)/3.14
    return X

def car_true(x):
    """
    x[0]=x
    x[1]=y
    x[2]=theta
    u[0]=v
    u[1]=phi    
       """  
    n=len(x)
    v = 0.005
    L = 0.025
    X = np.zeros((n,3))
    X[:,0] = v*np.cos(x[:,2]*3.14)
    X[:,1] = v*np.sin(x[:,2]*3.14)
    X[:,2] = v/L*np.tan(x[:,3])/3.14
    return X