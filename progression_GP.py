


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
from safe_plot import progression
from state_space import State_Space,ball,grid
from car_dynamics import car, car_true,car_execution, car_disturbance
from safety_control import safety_iteration_GP, safety_iteration, SI,SI_GP


def get_estimate_error(xx,yy,zz,mu_prep,cov_prep, train, kernel):
    diff=0
    for i in xx:
        for j in yy:
            for k in zz:
                center=np.array([i,j,k])
                for u in Up:
                    real = car_disturbance(center,u)
                    estimate,cov=gp(mu_prep,cov_prep,[center],train,kernel,car_true)
                    diff_ = np.linalg.norm(real-estimate)
                    if diff_>diff:
                        diff=diff_
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
                    diff_ = np.linalg.norm(real-estimate)
                    if diff_>diff:
                        diff=diff_
    return diff
if __name__=="__main__":
    #system setup

    P = 5
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
    print('Running Progression_GP')
    start = time.time()
    
    Xfree.construct(P,ball_obs_1)
    end =time.time()

    theta = np.linspace(-1,1,2**(P-1)+1)
    angle = theta[-1]
    #error and timing
    t=0
    Time=0
            
    size = 10
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
    
    
    num = 2
    
    X_tree,XX= safety_iteration(car,P,M,l,Xfree,angle,obs_1,Up, False)
    
    for i in range(size):

        for j in range(size):
            print(i,j)
            t_end=3
            t_step=0.1
            t_start=0
            #initial
            car_x = np.array([X[i][j],Y[i][j], angle])
           
            record=[]
            train=[]
            train_f=[]
            record.append(car_x)
            
            success, record, train_new,train_f_new,car_x=car_execution(car_x,t_end,t_step,X_tree,XX,ball_obs_1,record,Xfree)
        
            train=np.stack(train_new)    
            train_f=np.stack(train_f_new)    
            mu_prep,cov_prep=gp_prep(kernel,train, train_f,sigma,car_true)
            
            if not success:
                    Z[i][j] = success
                    continue


            t_end=10
            
            for k in range(num):
#                print(k)       
                #safety controller
                print('computing safety controller...')
                X_temp_tree,X_temp=safety_iteration_GP(P,M,l,mu_prep,cov_prep,Xfree,kernel,train,angle,obs_1,Up,False,progress,[x,y,z])               
            
                #car execution
                print('car execution...')
                success, record, train_new,train_f_new,car_x=car_execution(car_x,t_end,t_step,X_temp_tree,X_temp,ball_obs_1,record,Xfree)
                train=np.concatenate((train,train_new))
                train_f=np.concatenate((train_f,train_f_new))
                if not success:
                    Z[i][j] = success
                    break

                car_trajectory=np.stack(record)
#                plt.plot(car_trajectory[:,0],car_trajectory[:,1])
                
                #learn model
                print('model updating...')
                mu_prep,cov_prep=gp_prep(kernel,train, train_f,sigma,car_true) 
            Z[i][j] = success
    plt.pcolormesh(X,Y,Z,vmin=0, vmax=1)
    plt.colorbar()
            

    
