#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 17:12:43 2019

@author: zqy5086@AD.PSU.EDU
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 17:59:38 2019

@author: zqy5086@AD.PSU.EDU
"""

import multiprocessing as mp
import numpy as np
import datetime
from six.moves import cPickle
import time
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
from gp import gp,gp_prep
from state_space import ball, gridsearch_construct, to_string,state

from car_dynamics import car_true,car,car_disturbance
from safe_plot import progression_plot
#from safe_plot import progression
from state_space import State_Space,grid,d
from car_dynamics import car_execution, car_execution_disturbance
from safety_control import safety_iteration_GP, SI,SI_GP
import pickle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

def policy_search(center,X_safe,X_safe_tree,Xfree):
        ind1 = X_safe_tree.query(center.reshape(1,-1),1)
        ind2 = X_safe_tree.query((center+np.array([0,0,2])).reshape(1,-1),1)
        if ind1[0][0][0]<ind2[0][0][0]:
            ind=ind1
        else:
            ind=ind2
        coordinate = X_safe[ind[1][0][0]]
        key = to_string(coordinate)
        z = Xfree.table[key]
        if z.u is None:
            print(z.state)
        else:
            u=z.u[0]
        return u,z


def progression(car_model,center,X_safe,X_safe_tree,Xfree,hp,ball_obs):
    t_end = 160.
    t_start = 0.
    t_step=0.13
    ZZ=[]
    ys=[]
    while t_start<t_end:
        ys.append(center)
        
        u,zz=policy_search(center,X_safe,X_safe_tree,Xfree)
        ZZ.append((center,zz.state))
        z1,z2,_=np.vstack(ys).T  
        
        center=center+t_step*car_model(center,u)   
        if center[0]<-0.05 or center[0]>1.05 or center[1]<-0.05 or center[1]>1.05:
            return 1, z1,z2
        x = state(center)                
        if not d(ball_obs,x):
            return 0,z1,z2
        t_start=t_start+t_step
    z11,z22,z=np.vstack(ys).T
    return 1, z11,z22#,ZZ    
    

def testing(i,X,Y):
    ZZ=[]
    for j in range(size):
        center = np.array([X[i][j],Y[i][j], angle])
        if len(X_safe)==0:
            ZZ.append(-1)
        else:
            z,z1,z2 = progression(car_model,center,X_safe,X_safe_tree,Xfree,hP,ball_obs)
            ZZ.append(z)
    return (i,ZZ)
def collect_result(result):
        global results
        results.append(result)

if __name__=="__main__":
    
    
    size=20
    results=[]
    print(datetime.datetime.now())

    P=6
    M = 0.012
    l = 0.012
    Time=0
    Up = np.linspace(-0.3,0.3,5)
    hP = 2**(-P)
    eps_P = (hP)**(1/2)
    alpha_P = 2*hP+l*eps_P*(hP+M*eps_P)   
    
    obs_1 = State_Space(3,[0.4,0.45,0.0,0.6,-1,1])
    ball_obs_1 = ball(obs_1,M*eps_P+hP)
    Xfree = grid([[0,1],[0,1],[-1,1]])  
    Xfree.construct(P,Up) 
    plot_progress=True
    plot_control=True
    
    
    left = obs_1.parameter[0]
    down = obs_1.parameter[2]
    width = obs_1.parameter[1]-obs_1.parameter[0]
    height = obs_1. parameter[3]-obs_1.parameter[2]   
    
    X_unsafe=[]
    model=car
    car_model=car_disturbance
    theta = np.linspace(-1,1,2**(P-1)+1)
    angle = theta[-1]
    initial=np.array([1,0.1, angle])
    
    
    
    
    Y, X = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size)) 
    _,Z = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size)) 
    ball_obs = ball(obs_1,0)
    print('computing backup controller')
    ######################################################################################################
    #######################################backup controller phase ########################################
    start = time.time()

        
    for p in range(1,P+1):
        
        hp = 2**(-p)
        eps_p = (hp/(M*l))**(1/2)
    
        
        Convergence = False
        ball_obs_1 = ball(obs_1,M*eps_p+hp)
    
        #table construction based on dynamics
        need_update=[]
    #        print('getting X_temp')
    
        X_temp,state_temp,X_unsafe = gridsearch_construct(Xfree,p,ball_obs_1)
        X_temp_tree=KDTree(X_temp)
    
        for s in state_temp:
            
            if s.unsafe==False:                           
                        childs = SI(model,s,Xfree,Up,eps_p,X_temp_tree,X_temp,hp)
#                        s.s_=s.s
#                        s.childs = childs
                        
                        if s.s==0:
                            s.unsafe = True
                            X_unsafe.append(s)
                #                Xobs_p.append(s)
                        else:
                            s.unsafe = False
                            need_update.append(s)
                            
    
        Kp=[]
        while not Convergence:
            Convergence = True           
    
            for x in need_update:
    #                print('need update')
    
                flag = any([s.unsafe for s in x.childs])
                if x.unsafe==False and flag == True:
    #                        print('need update')
                        Kp.append(x)
                        need_update.remove(x)
                        
                        Convergence=False
            for x in Kp:
                    
                    SI(model,x,Xfree,Up,eps_p,X_temp_tree,X_temp,hp)                   
                    if x.s==0:
                        X_unsafe.append(x)
                        x.unsafe = True
                        Kp.remove(x)
    
                    elif x.s==x.s_: #value converge and safe
                        Kp.remove(x)
                        x.unsafe = False
    
                    else:
                        x.s_=x.s
        
      
                 
        X_safe = [x.state for x in state_temp if not x.unsafe]
        X_safe_control=[x.u for x in state_temp if not x.unsafe]

            
    
    if plot_control:
            end = time.time()
            t = end-start
            Time = Time+t
            plt.figure()
            
            X_temp_,state_temp,_ = gridsearch_construct(Xfree,P,fix=[2,angle])  

            T = [float(not x.unsafe) for x in state_temp]

            X1 = [float(x.state[0]) for x in state_temp]
            X2 = [float(x.state[1]) for x in state_temp]
            plt.scatter(X1,X2, c=T)
            plt.colorbar() 
            rectangle=plt.Rectangle((left,down),width,height,fc='red')
            plt.gca().add_patch(rectangle)
            plt.xlabel('Time elapsed: '+str(round(Time*1000)/1000)+', '+ str((2**P+1)**3)+' points',fontsize=16)# subgrid construction time: '+str(round((end_-start)*1000)/1000),fontsize=16)
            plt.title('Angle = '+str(angle*3.14)+' Safe points: '+str((2**P+1)**3-len(X_unsafe)),fontsize=16)
    if plot_progress:
        results=[]
        if len(X_safe)>0:
            X_safe_tree=KDTree(X_safe)
        print('computing backup progression map')
        plt.figure()                   
        pool = mp.Pool(mp.cpu_count())
        for i in range(size):
    #        print(i)
        #    testing(i,X,Y)
            pool.apply_async(testing, args=(i,X,Y),callback=collect_result)
        pool.close()
        pool.join()
        results.sort(key=lambda x: x[0])
        for i in range(size):
            for j in range(size):
                Z[i][j]=results[i][1][j]                
        plt.pcolormesh(X,Y,Z,vmin=-1, vmax=1)
        plt.title('Initial Value: ['+str(initial[0])+','+str(initial[1])+','+str(initial[2])+']')
        ##        
    #    plt.plot(z1,z2)
        plt.colorbar() 
        rectangle=plt.Rectangle((left,down),width,height,fc='red')
        plt.gca().add_patch(rectangle)
#####################################################################################################################################3
############################################Learning dynamic phase####################################################################      
    
    
    t_end=5
    t_step=0.1
    t_start=0
    #initial
    car_x = np.array([1,0.1, angle])
    sigma = 0.000001
    sigma_f = 1
    l_ = 0.001
#    kernel=lambda x,y:np.exp(-l_*min(np.linalg.norm(x-y),np.linalg.norm(x+np.array([0,0,2,0])-y)))*sigma_f
    kernel = C(1e-6, (1e-15, 1e-4)) * RBF(1e-12, (1e-15, 1e-9))
    
    record=[]
    train=[]
    train_f=[]
    record.append(car_x)
    
    success, record, train_new,train_f_new,car_x=car_execution_disturbance(car_x,t_end,t_step,X_safe_tree,X_safe,ball_obs_1,record,Xfree)

    train=np.stack(train_new)    
    train_f=np.stack(train_f_new)    
#    mu_prep,cov_prep=gp_prep(kernel,train, train_f,sigma,car_true)
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True).fit(train, train_f)
   ##################################################################################################################################
    #################################################iterations of progression plots #################################################
        
    #car execution
    num=4
    for k in range(num):
        print('car execution...')
        success, record, train_new,train_f_new,car_x=car_execution_disturbance(car_x,t_end,t_step,X_safe_tree,X_safe,ball_obs_1,record,Xfree)
        
        
        print('model updating...')
        train=np.concatenate((train,train_new))
        train_f=np.concatenate((train_f,train_f_new))
        car_trajectory=np.stack(record)
    #                plt.plot(car_trajectory[:,0],car_trajectory[:,1])
        
        #learn model        
#        mu_prep,cov_prep=gp_prep(kernel,train, train_f,sigma,car_true) 
        gpr = GaussianProcessRegressor(kernel=kernel).fit(train, train_f)
        #######################################################################################################################################
        #########################################################ends iterations#################################################################3
    print('test trajectory')
    theta=np.linspace(-1,1,50)
    COV=[]
    MU=[]
    TRUTH=[]
    for th in theta:
        center=np.array([0.9,0.2,th])
#        center=record[0]
        for u in Up:
            test=np.concatenate((center,np.array([u])))
#            mu,cov=gp(mu_prep,cov_prep,np.matrix(test),train,kernel,car_true)
            mu, cov=gpr.predict(np.matrix(test), return_cov=True) 
            truth=car_disturbance(center,u)-car(center,u)
            MU.append(mu)
            COV.append(cov)
            TRUTH.append(truth)
    MU=np.stack(MU)
    COV=np.stack(COV)
    TRUTH=np.stack(TRUTH)
    
    for i in range(3):
        plt.figure()
        plt.plot(TRUTH[:,i], color='blue',label='truth')
        plt.plot(MU[:,0][:,i]+2*np.sqrt(COV[:,0][:,0]), color='green', linestyle='dashed',label='var')
        plt.plot(MU[:,0][:,i]-2*np.sqrt(COV[:,0][:,0]),color='green',linestyle='dashed')
        plt.plot(MU[:,0][:,i],color='red',label='mean')
        plt.title(str(i))
        plt.legend()
#    state_temp=[Xfree.table[to_string(x)] for x in X_temp]
#    
#    ball_obs = ball(obs_1,0)
#    X_safe = [x.state for x in state_temp if not x.unsafe]
#    left = obs_1.parameter[0]
#    down = obs_1.parameter[2]
#    width = obs_1.parameter[1]-obs_1.parameter[0]
#    height = obs_1.parameter[3]-obs_1.parameter[2]   
#    if len(X_safe)>0:
#        X_safe_tree=KDTree(X_safe)
#    center=np.array([1,0.7,-angle])
#    z,z1,z2= progression(car_disturbance,center,X_safe,X_safe_tree,Xfree,hP,ball_obs)
#    
#    plt.figure()
#    plt.plot(car_trajectory[:,0],car_trajectory[:,1])
#    plt.pcolormesh(X,Y,Z,vmin=-1, vmax=1)
#    plt.title('Initial Value: ['+str(initial[0])+','+str(initial[1])+','+str(initial[2])+']')      
#    plt.colorbar() 
#    rectangle=plt.Rectangle((left,down),width,height,fc='red')
#    plt.gca().add_patch(rectangle) 
#    plt.plot(z1,z2)
#    
#    plt.figure()
#    X_temp_,state_temp,_ = gridsearch_construct(Xfree,P,fix=[2,angle])  
#    if save:
#        T = [float(not x.unsafe) for x in state_temp]
#    else:
#        for x in state_temp:
#            if x.u is None:
#                x.unsafe=True
#            else:
#                x.unsafe=False
#        T = [float(not x.unsafe) for x in state_temp]
#        
#    X1 = [float(x.state[0]) for x in state_temp]
#    X2 = [float(x.state[1]) for x in state_temp]
#    
#    plt.scatter(X1,X2, c=T)
#    plt.colorbar() 
#    rectangle=plt.Rectangle((left,down),width,height,fc='red')
#    plt.gca().add_patch(rectangle) 
#    plt.plot(z1,z2)
    

        