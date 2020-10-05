#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 11:53:14 2019

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
#from gp import gp,gp_prep
from state_space import ball, gridsearch_construct, to_string,state

from car_dynamics import car_true,car,car_disturbance
#from safe_plot import progression_plot
#from safe_plot import progression
from state_space import State_Space,grid,d
from car_dynamics import car_execution_disturbance
from safety_control import safety_iteration_GP, SI, SI_GP
import pickle

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
def policy_search(center,X_safe,X_safe_tree,Xfree, control_history):
        ind1 = X_safe_tree.query(center.reshape(1,-1),1,dualtree=False, breadth_first=False)
        if center[2]<0:
            ind2 = X_safe_tree.query((center+np.array([0,0,2])).reshape(1,-1),1)
        else:
            ind2 = X_safe_tree.query((center-np.array([0,0,2])).reshape(1,-1),1)
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
            
            index=min(z.U,key=z.U.get)
            indices=[k for k in z.U if z.U[k]==z.U[index]]
            if len(indices)>1:
                new_ind=np.argmin([control_history[k] for k in indices])
                u=control_history[indices[new_ind]]
                control_history[new_ind]+=1
            else:
                u=z.U[index]
                z.U[index]+=1
                control_history[index]+=1
            
            
        return u,z,control_history


    
def progression(car_model,center,X_safe,X_safe_tree,Xfree,hp,ball_obs,Up):
    print('hi')
    t_end = 730.
    t_start = 0.
    t_step=0.01
    ZZ=[]
    ys=[]
    z1=None
    z2=None
    control_history=dict()
    for u in Up:
        control_history[u]=0
    while t_start<t_end:
        x = state(center)                
        if not d(ball_obs,x):
            return 0,z1,z2,control_history#,U
        ys.append(center)
        
        u,zz,control_history=policy_search(center,X_safe,X_safe_tree,Xfree,control_history)
        ZZ.append((center,zz.state))
        z1,z2,_=np.vstack(ys).T  
#        U.append(u)
        center=center+t_step*car_model(center,u)   
        if center[0]<-0.01 or center[0]>1.01 or center[1]<-0.01 or center[1]>1.01:
            return 1, z1,z2,control_history
        
        t_start=t_start+t_step
    z11,z22,z=np.vstack(ys).T
    print(control_history)
    return 1, z11,z22,control_history #,U #,ZZ    


def testing(i,X,Y):
    ZZ=[]
    for j in range(size):
        center = np.array([X[i][j],Y[i][j], angle])
        if len(X_safe)==0:
            ZZ.append(-1)
        else:
            z,z1,z2 = progression(car_model,center,X_safe,X_safe_tree,Xfree,hp,ball_obs,Up)
            ZZ.append(z)
    return (i,ZZ)
def collect_result(result):
        global results
        results.append(result)
        
def parent_update(s,need_update):    
    for u in s.parents:
        for x in s.parents[u]:
            if not x.unsafe:
                index=np.where(x.u==u)
                if len(index[0])>0:
                    x.u=np.delete(x.u,index[0][0])
                if len(x.u)==0:
                    x.unsafe=True
#                    X_unsafe.append(x)
                    need_update.append(x)
 
if __name__=="__main__":
    save=True
    num=4
    size=50
    results=[]
    print(datetime.datetime.now())
    print('save is', save)
    P=5
    M = 0.005
    l = 0.005
    Time=0
    Up = np.linspace(-0.3,0.3,5)

    ratio=2.5
    
    obs_1 = State_Space(3,[0.47,0.52,0.25,0.75,-1,1])
    Xfree = grid([[0,1],[0,1],[-1,1]])  
    Xfree.construct(P,Up) 
    plot_progress=False
    plot_control=True
    
    
    left = obs_1.parameter[0]
    down = obs_1.parameter[2]
    width = obs_1.parameter[1]-obs_1.parameter[0]
    height = obs_1.parameter[3]-obs_1.parameter[2]   
    
    X_unsafe=[]
    model=car
    car_model=car_disturbance
    theta = np.linspace(-1,1,2**(P-1)+1)
    angle = theta[-1]
    initial=np.array([1,0.8, angle])
        
    T_end=10
#    kernel = C(0.1, (1e-3, 1)) * RBF(0.001, (1e-5, 1e-2))
    kernel = C(1e-4, (1e-6, 1e-3)) * RBF(0.05, (1e-2, 1))
    Y, X = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size)) 
    _,Z = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size)) 
    ball_obs = ball(obs_1,0)
    leaf_size=200
    
    control_history=dict()
    for u in Up:
        control_history[u]=0
    
    print('computing backup controller')
    ######################################################################################################
    #######################################backup controller phase ########################################
    start = time.time()
    if save:        
        for p in range(1,P+1):
            print(p)
            hp = 2**(-p)
            eps_p = (hp/(ratio*M*l))**(1/2)
            alpha_p=hp+l*eps_p*hp/2+M*l*eps_p**2
            ball_obs_1 = ball(obs_1,M*eps_p+hp)
        
            #table construction based on dynamics
            need_update=[]       
            X_temp,state_temp,X_unsafe = gridsearch_construct(Xfree,p,ball_obs_1)
            X_temp_tree=KDTree(X_temp, leaf_size=leaf_size, metric='chebyshev')
#            cnt=0         
            
            for s in state_temp:    
#                print(cnt)
#                cnt+=1
                if not s.unsafe:                                                     
                    SI(model,s,Xfree,Up,eps_p,X_temp_tree,X_temp,alpha_p)
                    if s.unsafe:                               
#                        X_unsafe.append(s)
                        need_update.append(s)

            for x in need_update:
                parent_update(x,need_update)     
            
      
                 
        X_safe = [x.state for x in state_temp if not x.unsafe]
#        X_safe_control=[x.u for x in state_temp if not x.unsafe]
        
        if len(X_safe)>0:
            X_safe_tree=KDTree(X_safe, leaf_size=leaf_size, metric='manhattan')
#        with open("safe_tree_backup.save", "wb") as f:
#            cPickle.dump(X_safe_tree, f)
#        with open("X_safe_backup.save", "wb") as f:len(X_safe_)/(2**p+1)**3>0.2
#            cPickle.dump(X_safe, f)
#        with open("X_safe_control_backup.save","wb") as f:
#            cPickle.dump(X_safe_control,f)
#    else:
#        print('loading...')
#        with open("safe_tree_backup.save", "rb") as fp:   # Unpickling
#            X_safe_tree = cPickle.load(fp)
#        with open("X_safe_backup.save", "rb") as fp:   # Unpickling
#            X_safe = cPickle.load(fp)
#        with open("X_safe_control_backup.save","rb") as fp:
#            X_safe_control = cPickle.load(fp)
#        for x,u in zip(X_safe,X_safe_control):
#            key=to_string(x)
#            Xfree.table[key].u=u
            
    
    if plot_control:
            print('plotting control grid...')
            end = time.time()
            t = end-start
            Time = Time+t
            plt.figure()
            T=[]
            for angle in theta:
                X_temp_,state_temp,_ = gridsearch_construct(Xfree,P,fix=[2,angle])  
                if save:
                    T.append([float(not x.unsafe) for x in state_temp])
                else:
                    for x in state_temp:
                        if x.u is None:
                            x.unsafe=True
                        else:
                            x.unsafe=False
                    T = np.array([float(not x.unsafe) for x in state_temp])
            T=np.stack(T)
            T=np.sum(T,axis=0)/len(theta)
            T[T==0]=-1
            X1 = [float(x.state[0]) for x in state_temp]
            X2 = [float(x.state[1]) for x in state_temp]
            center=initial
            z,z1,z2,control_record = progression(car_model,center,X_safe,X_safe_tree,Xfree,hp,ball_obs,Up)
            X_backup=X_safe
            X_backup_tree=X_safe_tree
            Xbackup=Xfree
            plt.scatter(X1,X2, c=T, vmin=0, vmax=1)
            plt.plot(z1,z2,color='red')
            plt.colorbar() 
            rectangle=plt.Rectangle((left,down),width,height,fc='red')
            plt.gca().add_patch(rectangle)
            plt.xlabel('Time elapsed: '+str(round(Time*1000)/1000)+', '+ str(len(X_temp))+' points',fontsize=16)# subgrid construction time: '+str(round((end_-start)*1000)/1000),fontsize=16)
            plt.title(' Safe points: '+str(len(X_safe)),fontsize=16)
   
    t_end=T_end
    t_step=0.1
    t_start=0
    #initial
    car_x = initial
       
    record=[]
    train=[]
    train_f=[]
    record.append(car_x)
    
    success, record, train_new,train_f_new,car_x,control_history=car_execution_disturbance(car_x,t_end,t_step,X_safe_tree,X_safe,ball_obs,record,Xfree,control_history)
    car_trajectory=np.stack(record)
        
        
    if plot_progress:
        results=[]
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
#        plt.title('Initial Value: ['+str(initial[0])+','+str(initial[1])+','+str(initial[2])+']')
     
#        plt.plot(z1,z2)
        plt.colorbar() 
        rectangle=plt.Rectangle((left,down),width,height,fc='red')
        plt.gca().add_patch(rectangle)
        plt.plot(car_trajectory[:,0],car_trajectory[:,1],color='blue')        
#####################################################################################################################################3
############################################Learning dynamic phase####################################################################      
 
    
    
    train=np.stack(train_new)    
    train_f=np.stack(train_f_new)    
#    mu_prep,cov_prep=gp_prep(kernel,train, train_f,sigma,car_true)
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True).fit(train, train_f)
    M_=np.max(train_f)
    M=max([M,M_])
    
    Xfree_ = grid([[0,1],[0,1],[-1,1]])  
    Xfree_.construct(P,Up) 
    switch=[]
    k_=0
    safe_index=0;
    safe_index_old=0;
    tol=0.1
    p_=1
   ##################################################################################################################################
    #################################################iterations of progression plots #################################################
    for k in range(num):
#        print(k)   
#        filename='X_safe_GP_'+str(k)+'.save'
#        filename2='safe_tree_GP_'+str(k)+'.save'
#        filename3='X_safe_control_GP_'+str(k)+'.save'
        #safety controller
        print('computing safety controller...')
        ############################################################################################################
        ##################################################GP controller phase########################################
        start = time.time()
        Time=0
        t=0
       
        if save:           
            for p in range(p_,P+1): 
                print(k,p)                         
                hp = 2**(-p)
                eps_p = (hp/(ratio*M*l))**(1/2)
                ball_obs_1 = ball(obs_1,M*eps_p+hp)
                alpha_p=2*hp+l*eps_p*hp+M*l*eps_p**2
                #table construction based on dynamics
                need_update=[]
                X_temp,state_temp,X_unsafe = gridsearch_construct(Xfree_,p,ball_obs_1)
                X_temp_tree=KDTree(X_temp,leaf_size=leaf_size,metric='chebyshev')
                for s in state_temp:
                    if not s.unsafe:   
                            SI_GP(s,Xfree_,Up,eps_p,X_temp_tree,X_temp,gpr,hp,M,l)                      
                            if s.unsafe:                               
#                                X_unsafe.append(s)
                                need_update.append(s)
                for x in need_update:
                    parent_update(x,need_update)     
                
            
                X_safe_ = [x.state for x in state_temp if not x.unsafe]
                sp=len(X_safe_)/len(X_temp)
                
                if sp>safe_index:
                    print('switching')
                    safe_index=sp
                            ######################################################################################################################
        #################################################33#progression plot###################################################3
                    end = time.time()
                    t = end-start-t
                    if plot_control:
                        print('plotting controller plot...')                       
                        Time = Time+t
                        plt.figure()                        
                        theta = np.linspace(-1,1,2**(p-1)+1)
                        T=[]
                        for angle in theta:
                            X_temp_,state_temp,_ = gridsearch_construct(Xfree_,p,fix=[2,angle])  
                            if save:
                                T.append([float(not x.unsafe) for x in state_temp])
                            else:
                                for x in state_temp:
                                    if x.u is None:
                                        x.unsafe=True
                                    else:
                                        x.unsafe=False
                                T = np.array([float(not x.unsafe) for x in state_temp])
                        T=np.stack(T)
                        T=np.sum(T,axis=0)/len(theta)
                        T[T==0]=-1
                        X1 = [float(x.state[0]) for x in state_temp]
                        X2 = [float(x.state[1]) for x in state_temp]
                        plt.scatter(X1,X2, c=T, cmap='viridis', vmin=0, vmax=1)
                        plt.plot(car_trajectory[:,0],car_trajectory[:,1],color='blue')
                        plt.colorbar() 
                        rectangle=plt.Rectangle((left,down),width,height,fc='red')
                        plt.gca().add_patch(rectangle)
                        plt.xlabel('Time elapsed: '+str(round(Time*1000)/1000)+', '+ str(len(X_temp))+' points',fontsize=16)# subgrid construction time: '+str(round((end_-start)*1000)/1000),fontsize=16)
                        plt.title(' Safe points: '+str(len(X_safe_)),fontsize=16)
                    
                    ball_obs = ball(obs_1,0)
                    
                    print('in safe controller car execution...')
                    t_end=t
                    
                    success, record, _,_,car_x,control_history=car_execution_disturbance(car_x,t_end,t_step,X_safe_tree,X_safe,ball_obs,record,Xfree,control_history)

                    
                    
                    if not success:
                        print('collision!!!!')
                    X_safe=X_safe_
                    X_safe_tree=KDTree(X_safe, leaf_size=leaf_size,metric='manhattan')
                    car_trajectory_=np.stack(record)
                    Xfree=Xfree_
                    switch.append(car_trajectory_[-1,:2])
                    
                    print('car execution...')
                   
                    t_end=T_end
                    success, record, train_new,train_f_new,car_x,control_history=car_execution_disturbance(car_x,t_end,t_step,X_safe_tree,X_safe,ball_obs,record,Xfree,control_history)
                    print('model updating...')
                    train=np.concatenate((train,train_new))
                    train_f=np.concatenate((train_f,train_f_new))
                    car_trajectory=np.stack(record)
                        
                    if plot_progress:
                        print('plotting progression plot...')
                        results=[] 
                        print('progression map...')
                        plt.figure()                       
                        pool = mp.Pool(mp.cpu_count())
                        for i in range(size):
                #            print(i)        ######################################################################################################################
        #################################################33#progression plot###################################################3
                            pool.apply_async(testing, args=(i,X,Y),callback=collect_result)
                        pool.close()
                        pool.join()
                        results.sort(key=lambda x: x[0])
                        for i in range(size):
                            for j in range(size):
                                Z[i][j]=results[i][1][j]                   
                        plt.pcolormesh(X,Y,Z,vmin=-1, vmax=1)
#                        plt.title('Initial Value: ['+str(initial[0])+','+str(initial[1])+','+str(initial[2])+']')      
                    #    plt.plot(z1,z2)
                        plt.colorbar() 
                        rectangle=plt.Rectangle((left,down),width,height,fc='red')
                        plt.gca().add_patch(rectangle)      
                        plt.plot(car_trajectory[:,0],car_trajectory[:,1],color='blue')
                        
                        plt.plot(car_trajectory_[:,0],car_trajectory_[:,1],color='black') 
                        switch_points=np.stack(switch)
                        plt.scatter(switch_points[:,0],switch_points[:,1], color='black')
                        
                        ########################################################################################
                        ################################ learning ###################################################
                    print('update uncertainty...')
                    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True).fit(train, train_f)
                    M_=np.max(train_f)
                    M=max([M,M_])
                        
                    if safe_index-safe_index_old<tol and safe_index>tol:
                        print('refine grid ...')
                        safe_index_old=safe_index
                        p_=p+1
                        break
                    else:
                        safe_index_old=safe_index
                        start=time.time()
                        p_=p
                        break
    print('safety test...')
    t_end=430
    success, record, _,_,car_x,control_history=car_execution_disturbance(car_x,t_end,t_step,X_safe_tree,X_safe,ball_obs,record,Xfree,control_history)
    car_trajectory=np.stack(record)
    
    plt.figure()
    plt.pcolormesh(X,Y,Z,vmin=-1, vmax=1)

    plt.colorbar() 
    rectangle=plt.Rectangle((left,down),width,height,fc='red')
    plt.gca().add_patch(rectangle)      
    plt.plot(car_trajectory[:,0],car_trajectory[:,1],color='blue')
    
    plt.plot(car_trajectory_[:,0],car_trajectory_[:,1],color='black') 
    switch_points=np.stack(switch)
    plt.scatter(switch_points[:,0],switch_points[:,1], color='black')
                    

                     
#            with open(filename, "wb") as fp:   #Pickling
#                cPickle.dump(X_safe, fp)
#            with open(filename2, "wb") as fp:   #Pickling
#                cPickle.dump(X_safe_tree, fp)
#            with open(filename3,"wb") as fp:
#                cPickle.dump(X_safe_control,fp)
#        else:
#            print('loading...')
#            with open(filename, "rb") as fp:   # Unpickling
#                X_safe = cPickle.load(fp)
#            with open(filename2, 'rb') as f:
#                X_safe_tree = cPickle.load(f)
#            with open(filename3,'rb') as f:
#                X_safe_control=cPickle.load(f)
#            for x,u in zip(X_safe,X_safe_control):
#                key=to_string(x)
#                Xfree.table[key].u=u

        
        
#        plt.plot(car_trajectory[:,0],car_trajectory[:,1])        
        #learn model        
        
    #######################################################################################################################################
    #########################################################ends iterations#################################################################3
#    print('test trajectory')
#    
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
##    plt.plot(z1,z2)
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
#    
