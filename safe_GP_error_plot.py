


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 14:35:10 2019

@author: zqy5086@AD.PSU.EDU
"""
from math import ceil, floor,exp,sqrt,sin,cos,tan,pi
import numpy as np

import time
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
from gp import gp,gp_prep
from safe_plot import progression_plot
from state_space import State_Space,ball,grid,to_string
from car_dynamics import car, car_true,car_execution, car_disturbance, car_execution_plain
from safety_control import safety_iteration_GP, safety_iteration, SI,SI_GP


def get_estimate_error(xx,yy,zz,mu_prep,cov_prep, train, kernel,Up):
    diff=0
    for i in xx:
        for j in yy:
            for k in zz:
                center=np.array([i,j,k])
                for u in Up:
                    real = car_disturbance(center,u)
                    test = np.concatenate((center,np.array([u])))
                    estimate,cov=gp(mu_prep,cov_prep,[test],train,kernel,car_true)
                    diff= np.linalg.norm(real-estimate)+diff             
    return diff

def get_dynamics_error(xx,yy,zz):
    diff=0
    for i in xx:
        for j in yy:
            for k in zz:
                center=np.array([i,j,k])
                for u in Up:
                    real = car_disturbance(center,u)
                    estimate = car(center,u)
                    diff = np.linalg.norm(real-estimate)+diff
                    
    return diff
if __name__=="__main__":
    #system setup

    P = 6
    p = 1
    M = 0.012
    l = 0.012
    
    hP = 2**(-P)
    eps_P = (hP/(l*M))**(1/2)
    alpha_P = 2*hP+l*eps_P*(hP+M*eps_P)    
        
    #space setup
    Xfree = grid([[0,1],[0,1],[-1,1]])   
    
    #goal setup
    Xgoal1 = State_Space(2,[0.8,0.8,0.9,0.9,-1,1])
#    Bgoal1 = ball(Xgoal1,0.1)

    #obstacle setup
    obs_1 = State_Space(3,[0.4,0.45,0.0,0.6,-1,1])
    
    
    left = obs_1.parameter[0]
    down = obs_1.parameter[2]
    width = obs_1.parameter[1]-obs_1.parameter[0]
    height = obs_1.parameter[3]-obs_1.parameter[2]    
    
    #control setup
    Up = np.linspace(-0.3,0.3,5)

    #disturbance setup
    ball_obs_1 = ball(obs_1,M*eps_P+hP)
    
    #dynamics paramters
    print('grid construction')
    start = time.time()
    
    Xfree.construct(P,ball_obs_1)
    end =time.time()
    
    print('Construction time: ', str(end-start))
    theta = np.linspace(-1,1,2**(P-1)+1)
    angle = theta[-1]
    #error and timing
    t=0
    Time=0
            
    size = 50
    Y, X = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size)) 
    _,Z = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size)) 

    X_safe=[]
    safe_exist=False
    
    #GP setup
    sigma = 0.001
    sigma_f = 1
    l_ = 0.05
    kernel=lambda x,y:np.exp(-l_*np.linalg.norm(x-y))*sigma_f
    
    #initial samples:
    t_end=3
    t_step=0.1
    t_start=0
    
    
    
    x_cor = np.linspace(0,1,2)
    y_cor = np.linspace(0,1,3)
    z_cor = np.linspace(-1,1,3)
    print('getting offline safety controller...')
    X_tree,XX= safety_iteration(car,P,M,l,Xfree,angle,obs_1,Up,False)
    plt.figure()
    progress = False
    
    xx = np.linspace(0,1,size)
    yy = np.linspace(0,1,size)
    zz = np.linspace(-1,1,size)                
    Error =[]
    print('getting initial dynamics error...')
    diff = get_dynamics_error(xx,yy,zz)
    Error.append(diff)
    
    XX_safe = []
    for x in XX:
        s = Xfree.table[to_string(x)]
        if not s.unsafe:
            XX_safe.append(x)
    XX_safe_tree = KDTree(XX_safe)
    
    
    print('Getting the progresion plot...')
    for x in x_cor:
        for y in y_cor:
            for z in z_cor:
                print(x,y,z)
                error = Error
#                if z==1:
#                    progress=True
                record=[]
                train=[]
                train_f=[]
                
                car_x=np.array([x,y,z])
                record.append(car_x)
                
                
                print('data initialization')
                
                record, train_new,train_f_new,car_x=car_execution_plain(car_x,t_end,t_step,XX_safe_tree,XX_safe,record,Xfree)
            
                train=np.stack(train_new)    
                train_f=np.stack(train_f_new)    
                mu_prep,cov_prep=gp_prep(kernel,train, train_f,sigma,car_true)
                
                diff = get_estimate_error(xx,yy,zz,mu_prep,cov_prep, train, kernel,Up)                      
                error.append(diff)
               
                #planner computation, multigrid
                t_step=0.1
                t_end=10
                
                mark = False
                print('starting iterations of error computing...')
                for k in range(6):
                    print(k)    
                    print('safety iteration..')
                    X_temp_tree,X_temp=safety_iteration_GP(P,M,l,mu_prep,cov_prep,Xfree,kernel,train,angle,obs_1,Up,False,progress,[x,y,z])               
                    X_safe = []
                    for x in X_temp:
                        s = Xfree.table[to_string(x)]
                        if not s.unsafe:
                            X_safe.append(x)
                    X_safe_tree = KDTree(X_safe)
                    
                    
                    record, train_new,train_f_new,car_x=car_execution_plain(car_x,t_end,t_step,X_safe_tree,X_safe,record,Xfree)
                    
#                    car_trajectory=np.stack(record)
#                    plt.plot(car_trajectory[:,0],car_trajectory[:,1])
                    train=np.concatenate((train,train_new))
                    train_f=np.concatenate((train_f,train_f_new))
                    
                    mu_prep,cov_prep=gp_prep(kernel,train, train_f,sigma,car_true)   
                    
                    
                    print('error estimation...')
                    diff = get_estimate_error(xx,yy,zz,mu_prep,cov_prep, train, kernel,Up)
                    error.append(diff)
                   
                plt.figure()    
                plt.plot(error)
                plt.title('Initial state: ['+str(x)+', '+str(y)+', '+str(z)+']' )
        
        

    
    
