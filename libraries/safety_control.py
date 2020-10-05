#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 23:03:36 2019

@author: zqy5086@AD.PSU.EDU

Library of safety controller
"""



import time
import numpy as np
from state_space import ball, gridsearch_construct, to_string
import matplotlib.pyplot as plt

from math import  ceil,sqrt
from car_dynamics import car,car_disturbance
from safe_plot import progression_plot


def policy_search(center,X_temp,X_temp_tree,Xfree,u_):
    ind = X_temp_tree.query(center.reshape(1,-1),1)
    coordinate = X_temp[ind[1][0][0]]
    key = to_string(coordinate)
    z = Xfree.table[key]
#    z1,z2,_=np.vstack(ys).T        
    
    if z.unsafe:
        u=u_
    else:
        u=z.u[0]
    return u   

def SI(model,x,Xfree,Up,eps_p,X_temp_tree,X_temp,alpha_p):
    index=[]
    
    for ind,u in enumerate(Up):
        
        z=x.state+eps_p*model(x.state,u) 
        if z[2]>1: 
            z[2]=z[2]-2
        elif z[2]<-1:
            z[2] = z[2]+2
            
#        unsafe,states = Xfree.safe_kdtree_search(X_temp,z)

        inds1 = X_temp_tree.query_radius(z.reshape(1,-1),alpha_p)
        theta=z[2]
        if theta<-1+alpha_p:
            z[2]=1
            inds2 = X_temp_tree.query_radius(z.reshape(1,-1),-theta+alpha_p-1)
            inds=np.concatenate((inds1[0],inds2[0]))
        elif theta>1-alpha_p:
            z[2]=-1
            inds2 = X_temp_tree.query_radius(z.reshape(1,-1),-theta-alpha_p+1)
            inds=np.concatenate((inds1[0],inds2[0]))
        else:
            inds=inds1[0]
#        ind_dist=np.concatenate((inds1[0][0],inds2[0][0]))
        
#        inx=np.argpartition(ind_dist,n)
#        inds=inx[:n]
        coordinates = [X_temp[ind] for ind in inds]
        keys = [to_string(s) for s in coordinates]
        states = [Xfree.table[key] for key in keys]
        unsafe = any([s.unsafe for s in states])
        if unsafe:
            index.append(ind)   
        else:
#            x.childs[u]=states
            for s in states:
                s.parents[u].add(x) 
                

    x.u = np.delete(Up,index)
    for u in x.u:
        if u not in x.U:
            x.U[u]=0
    if len(x.u)>0:
        x.unsafe=False
    else:
        x.unsafe=True

def SI_GP(x,Xfree,Up,eps_P,X_temp_tree,X_temp,hp,M,l):
    index=[]    
    for ind,u in enumerate(Up):

        #mu, cov<-robot.dataR[str(x.state)+str(u)]
        #std=sqrt(cov)
        
        
#        test=np.concatenate((x.state,np.array([u])))
#        mu, std=gpr.predict(np.matrix(test), return_std=True) 
        mu,cov=gp(mu_prep,cov_prep,np.matrix(test),train,kernel,car_true)
        z=x.state+eps_P*(mu+car(x.state,u))
        z=z[0]
        if z[2]>1: 
            z[2]=z[2]-2
        elif z[2]<-1:
            z[2] = z[2]+2


        alpha_p=hp+l*eps_P*hp/2+M*l*eps_P**2+eps_P*2*std

#        n=min(int(6+ceil(6*n)),len(X_temp))
        inds1 = X_temp_tree.query_radius(z.reshape(1,-1),alpha_p)
        theta=z[2]
        if theta<-1+alpha_p:
            z[2]=1
            inds2 = X_temp_tree.query_radius(z.reshape(1,-1),-theta+alpha_p-1)
            inds=np.concatenate((inds1[0],inds2[0]))
        elif theta>1-alpha_p:
            z[2]=-1
            inds2 = X_temp_tree.query_radius(z.reshape(1,-1),-theta-alpha_p+1)
            inds=np.concatenate((inds1[0],inds2[0]))
        else:
            inds=inds1[0]
#        inx=np.argpartition(ind_dist,n)
#        inds=inx[:n]
        coordinates = [X_temp[ind] for ind in inds]
        keys = [to_string(s) for s in coordinates]
        states = [Xfree.table[key] for key in keys]
        unsafe = any([s.unsafe for s in states])
        
        if unsafe:
            index.append(ind)                   
        else:
#            x.childs[u]=states
            for s in states:
                s.parents[u].add(x)        
                
    x.u = np.delete(Up,index)
    for u in x.u:
        if u not in x.U:
            x.U[u]=0
    if len(x.u)>0:
        x.unsafe=False
    else:
        x.unsafe=True


def safety_iteration(model,P,M,l,Xfree,angle,obs_1,Up,plot_control,plot_progression,size,initial):
    #planner computation, multigrid
    Time=0
    left = obs_1.parameter[0]
    down = obs_1.parameter[2]
    width = obs_1.parameter[1]-obs_1.parameter[0]
    height = obs_1.parameter[3]-obs_1.parameter[2]   
    X_unsafe=[]
    for p in range(1,P+1):
        
#        print('geting grid ',p)
#        print('Time elapsed:', end-start)
        
        start = time.time()
        hp = 2**(-p)
        eps_p = (hp/(M*l))**(1/2)
#        alpha_p = 2*hp+l*eps_p*(hp+M*eps_p)        
        
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
                        s.s_=s.s
                        s.childs = childs
                        
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
 
       
        if plot_control:
            end = time.time()
            t = end-start
            Time = Time+t
            plt.figure()
            
            X_temp_,state_temp,_ = gridsearch_construct(Xfree,p,fix=[2,angle])  
            T = [float(not x.unsafe) for x in state_temp]
            X1 = [float(x.state[0]) for x in state_temp]
            X2 = [float(x.state[1]) for x in state_temp]
            plt.scatter(X1,X2, c=T)
            plt.colorbar() 
            rectangle=plt.Rectangle((left,down),width,height,fc='red')
            plt.gca().add_patch(rectangle)
            plt.xlabel('Time elapsed: '+str(round(Time*1000)/1000)+', '+ str((2**p+1)**3)+' points',fontsize=16)# subgrid construction time: '+str(round((end_-start)*1000)/1000),fontsize=16)
            plt.title('Angle = '+str(angle*3.14)+' Safe points: '+str((2**p+1)**3-len(X_unsafe)),fontsize=16)
        if plot_progression:
            if p>=5:

                Y, X = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size)) 
                _,Z = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size)) 
                results=[]
                progression_plot(car_disturbance,Xfree,obs_1,hp,state_temp,angle,size, X,Y,Z,initial,results)
                plt.xlabel('p = '+str(p))             
                
            
    X_safe = [x.state for x in state_temp if not x.unsafe]
    if len(X_safe)>0:
        X_safe_tree=KDTree(X_safe)     
    return X_safe,X_safe_tree

def safety_iteration_GP(P,M,l,mu_prep,cov_prep,Xfree,kernel,train,angle,obs_1,Up,plot_safe,plot_progression,initial,size):
    X_unsafe=[]
    Time=0
    left = obs_1.parameter[0]
    down = obs_1.parameter[2]
    width = obs_1.parameter[1]-obs_1.parameter[0]
    height = obs_1.parameter[3]-obs_1.parameter[2]  
    results=[]
    for p in range(1,P+1):
        
#        print('geting grid ',p)
#        print('Time elapsed:', end-start)
        
        start = time.time()
        hp = 2**(-p)
        eps_p = (hp/(M*l))**(1/2)
#        alpha_p = 2*hp+l*eps_p*(hp+M*eps_p)        
        
        Convergence = False
        ball_obs_1 = ball(obs_1,M*eps_p+hp)

        #table construction based on dynamics
        need_update=[]
#        print('getting X_temp')
#        start_ = time.time()
        X_temp,state_temp,X_unsafe = gridsearch_construct(Xfree,p,ball_obs_1)
        X_temp_tree=KDTree(X_temp)
#        if p==1:
#            X_safe_tree_=KDTree(X_temp)
#        end_ = time.time()
#        print('Done getting X_temp')
#        num = 0       
        for s in state_temp:
            
            if s.unsafe==False:   
                        childs = SI_GP(s,Xfree,Up,eps_p,X_temp_tree,X_temp,mu_prep,cov_prep,train,kernel,hp)
                        s.s_=s.s
                        s.childs = childs
               
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
                    SI_GP(x,Xfree,Up,eps_p,X_temp_tree,X_temp,mu_prep,cov_prep,train,kernel,hp)
                    if x.s==0:
                        X_unsafe.append(x)
                        x.unsafe = True
                        Kp.remove(x)

                    elif x.s==x.s_: #value converge and safe
                        Kp.remove(x)
                        x.unsafe = False

                    else:#    z1,z2,_=np.vstack(ys).T        
    
                        x.s_=x.s
 
        end = time.time()
        t = end-start
        Time = Time+t
        if plot_safe:
        
            if p>=4:
                plt.figure()    
                X_temp_,state_temp,_ = gridsearch_construct(Xfree,p,fix=[2,angle])  
                T = [float(not x.unsafe) for x in state_temp]
                X1 = [float(x.state[0]) for x in state_temp]
                X2 = [float(x.state[1]) for x in state_temp]
                plt.scatter(X1,X2, c=T)   
                
                plt.colorbar() 
                rectangle=plt.Rectangle((left,down),width,height,fc='red')
                plt.gca().add_patch(rectangle)
                plt.xlabel('Time elapsed: '+str(round(Time*1000)/1000)+', '+ str((2**p+1)**3)+' points',fontsize=16)# subgrid construction time: '+str(round((end_-start)*1000)/1000),fontsize=16)
                plt.title('Angle = '+str(angle*3.14)+' Safe points: '+str((2**p+1)**3-len(X_unsafe)),fontsize=16)
                
        if plot_progression:
            if p>=6:

                Y, X = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size)) 
                _,Z = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size)) 
                progression_plot(car_disturbance,Xfree,obs_1,hp,state_temp,angle,size, X,Y,Z,initial,results)
                plt.xlabel('p = '+str(p))
    X_safe = [x.state for x in state_temp if not x.unsafe]
    if len(X_safe)>0:
        X_safe_tree=KDTree(X_safe)               
    return X_safe_tree,X_safe