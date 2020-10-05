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
import sys
sys.path.append('/home/AD.PSU.EDU/zqy5086/Publication/ICRA2021/GPR')
sys.path.append('/home/AD.PSU.EDU/zqy5086/Publication/ICRA2021/libraries')
sys.setrecursionlimit(1500)
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
    t_end = 830.
    t_start = 0.
    t_step=0.13
    ZZ=[]
    ys=[]
    z1=None
    z2=None
    while t_start<t_end:
        x = state(center)                
        if not d(ball_obs,x):
            return 0,z1,z2
        ys.append(center)
        
        u,zz=policy_search(center,X_safe,X_safe_tree,Xfree)
        ZZ.append((center,zz.state))
        z1,z2,_=np.vstack(ys).T  
        
        center=center+t_step*car_model(center,u)   
        if center[0]<-5.01 or center[0]>5.01 or center[1]<-5.01 or center[1]>5.01:
            return 1, z1,z2
        
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
            z,z1,z2 = progression(car_model,center,X_safe,X_safe_tree,Xfree,hp,ball_obs)
            ZZ.append(z)
    return (i,ZZ)
def collect_result(result):
        global results
        results.append(result)
        
def parent_update(S):
    for s in S:
        for u in s.parents:
            for x in s.parents[u]:
                if not x.unsafe:
                    index=np.where(x.u==u)
                    if len(index[0])>0:
                        x.u=np.delete(x.u,index[0][0])
                    if len(x.u)==0:
                        x.unsafe=True
#                        X_unsafe.append(x)
                        S.append(x)
                    
def obstacle_collision_avoidance(Xfree,p):
    print(p)                         
    hp = 2**(-p)*10
    eps_p = (hp/(M*l))**(1/2)
    ball_obs_1 = ball(obs_1,M*eps_p+hp)

    #table construction based on dynamics
    need_update=[]
    X_temp,state_temp,X_unsafe = gridsearch_construct(Xfree,p,ball_obs_1) #identify obstacle unsafe state
    X_temp_tree=KDTree(X_temp)
    
    #fused GPR prediction at state_temp       
    for s in state_temp:
        if not s.unsafe:   
                SI_GP(s,Xfree,Up,eps_p,X_temp_tree,X_temp,hp)               
                if s.unsafe:                               
                    X_unsafe.append(s)
                    need_update.append(s)
    for x in need_update:
        parent_update(x)   
    

    X_safe= [x.state for x in state_temp if not x.unsafe]
    return X_safe
if __name__=="__main__":
    save=True
    num=5
    size=50
    results=[]
    print(datetime.datetime.now())
    print('save is', save)
    P=4
    M = 0.005
    l = 0.005
    Time=0
    Up = np.linspace(-0.3,0.3,5)
#    hP = 2**(-P)
#    eps_P = (hP)**(1/2)  
    
#    obs_1 = State_Space(3,[-0.5,0.5,-3.5,2.5,-1,1])
    obs_1 = State_Space(3,[0.4,0.45,0.0,0.6,-1,1])
#    Xfree = grid([[-5,5],[-5,5],[-1,1]])  
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
    initial=np.array([4,2.0, angle])
        
    T_end=15
    kernel = C(1.0, (1e-3, 10)) * RBF(0.01, (1e-7, 10))
    Y, X = np.meshgrid(np.linspace(-5, 5, size), np.linspace(-5, 5, size)) 
    _,Z = np.meshgrid(np.linspace(-5, 5, size), np.linspace(-5, 5, size)) 
    ball_obs = ball(obs_1,0)
    print('computing backup controller')
    ######################################################################################################
    #######################################backup controller phase ########################################
    start = time.time()
    if save:        
        p=P
        print(p)
        hp = 2**(-p)/2
        eps_p = (hp/(l*M))**(1/2)
        alpha_p=2*hp+l*eps_p*hp+M*l*eps_p**2

        ball_obs_1 = ball(obs_1,M*eps_p+hp)
    
        #table construction based on dynamics
        need_update=[]       
        X_temp,state_temp = gridsearch_construct(Xfree,p,ball_obs_1)
        X_temp_tree=KDTree(X_temp)
    
    
        #fused GPR predict over state_temp times U_p
    
    
        for s in state_temp:               
            if not s.unsafe:                                                     
                SI(model,s,Xfree,Up,eps_p,X_temp_tree,X_temp,alpha_p+0.01)#alpha_p
                if s.unsafe:                               
#                        X_unsafe.append(s)
                    need_update.append(s)
        print(len(need_update))     
#        for x in need_update:
        parent_update(need_update)     
                
                 
        X_safe = [x.state for x in state_temp if not x.unsafe]
        print(len(X_safe)) 
#        X_safe_control=[x.u for x in state_temp if not x.unsafe]
        
        if len(X_safe)>0:
            X_safe_tree=KDTree(X_safe)
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
            end = time.time()
            t = end-start
            Time = Time+t
            plt.figure()
            state_temp_=[x for x in state_temp if x.state[2]==angle]
#            X_temp_,state_temp,_ = gridsearch_construct(Xfree,P,fix=[2,angle])  
            if save:
                T = [float(x.unsafe) for x in state_temp_]
            else:
                for x in state_temp:
                    if x.u is None:
                        x.unsafe=True
                    else:
                        x.unsafe=False
                T = [float(not x.unsafe) for x in state_temp_]
            X1 = [float(x.state[0]) for x in state_temp_]
            X2 = [float(x.state[1]) for x in state_temp_]
            center=initial
            plt.scatter(X1,X2, c=T)
#            z,z1,z2 = progression(car_model,center,X_safe,X_safe_tree,Xfree,hp,ball_obs)
#            plt.plot(z1,z2,color='red')
#            plt.colorbar() 
            rectangle=plt.Rectangle((left,down),width,height,fc='red')
            plt.gca().add_patch(rectangle)
#            plt.xlabel('Time elapsed: '+str(round(Time*1000)/1000)+', '+ str((2**P+1)**3)+' points',fontsize=16)# subgrid construction time: '+str(round((end_-start)*1000)/1000),fontsize=16)
#            plt.title('Angle = '+str(angle*3.14)+' Safe points: '+str((2**P+1)**3-len(X_unsafe)),fontsize=16)
    if plot_progress:
        t_end=T_end
        t_step=0.1
        t_start=0
        #initial
        car_x = initial
           
        record=[]
        train=[]
        train_f=[]
        record.append(car_x)
        
        success, record, train_new,train_f_new,car_x=car_execution_disturbance(car_x,t_end,t_step,X_safe_tree,X_safe,ball_obs,record,Xfree)
        car_trajectory=np.stack(record)
        
        
        
        results=[]
#        if len(X_safe)>0:
#            X_safe_tree=KDTree(X_safe)
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
        plt.plot(car_trajectory[:,0],car_trajectory[:,1],color='cyan')        
#####################################################################################################################################3
############################################Learning dynamic phase####################################################################      
"""
    
    
    train=np.stack(train_new)    
    train_f=np.stack(train_f_new)    
#    mu_prep,cov_prep=gp_prep(kernel,train, train_f,sigma,car_true)
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True).fit(train, train_f)
    
    Xfree_ = grid([[-5,5],[-5,5],[-1,1]])  
    Xfree_.construct(P,Up) 
    switch=[]
   ##################################################################################################################################
    #################################################iterations of progression plots #################################################
    for k in range(num):
#        print(k)   
        filename='X_safe_GP_'+str(k)+'.save'
        filename2='safe_tree_GP_'+str(k)+'.save'
        filename3='X_safe_control_GP_'+str(k)+'.save'
        #safety controller
        print('computing safety controller...')
        ############################################################################################################
        ##################################################GP controller phase########################################
        start = time.time()
        Time=0
        t=0
        if save:           
            for p in range(1,P+1): 
                print(k,p)                         
                hp = 2**(-p)*10
                eps_p = (hp/(M*l))**(1/2)
                ball_obs_1 = ball(obs_1,M*eps_p+hp)
        
                #table construction based on dynamics
                need_update=[]
                X_temp,state_temp,X_unsafe = gridsearch_construct(Xfree_,p,ball_obs_1) #identify obstacle unsafe state
                X_temp_tree=KDTree(X_temp)
                
                #fused GPR prediction at state_temp       
                for s in state_temp:
                    if not s.unsafe:   
                            SI_GP(s,Xfree_,Up,eps_p,X_temp_tree,X_temp,gpr,hp)               
                            if s.unsafe:                               
                                X_unsafe.append(s)
                                need_update.append(s)
                for x in need_update:
                    parent_update(x)   
                
            
                X_safe_ = [x.state for x in state_temp if not x.unsafe]

                if len(X_safe_)/(2**p+1)**3>0.2:
                            ######################################################################################################################
        #################################################33#progression plot###################################################3
                    end = time.time()
                    t = end-start-t
                    if plot_control:
                        print('plotting controller plot...')                       
                        Time = Time+t
                        plt.figure()                        
                        X_temp_,state_temp,_ = gridsearch_construct(Xfree_,p,fix=[2,angle])  
                        if save:
                            T = [float(not x.unsafe) for x in state_temp]
                        else:
                            for x in state_temp:
                                if x.u is None:
                                    x.unsafe=True
                                else:
                                    x.unsafe=False
                            T = [float(not x.unsafe) for x in state_temp]
                            
                        X1 = [float(x.state[0]) for x in state_temp]
                        X2 = [float(x.state[1]) for x in state_temp]
                        plt.scatter(X1,X2, c=T)
                        plt.colorbar() 
                        rectangle=plt.Rectangle((left,down),width,height,fc='red')
                        plt.gca().add_patch(rectangle)
                        plt.xlabel('Time elapsed: '+str(round(Time*1000)/1000)+', '+ str((2**p+1)**3)+' points',fontsize=16)# subgrid construction time: '+str(round((end_-start)*1000)/1000),fontsize=16)
                        plt.title('Angle = '+str(angle*3.14)+' Safe points: '+str((2**p+1)**3-len(X_unsafe)),fontsize=16)
                    if plot_progress:
                        print('plotting progression plot...')
                        results=[] 
                        ball_obs = ball(obs_1,0)
                        
                        print('in safe controller car execution...')
                        t_end=t
                        success, record, _,_,car_x=car_execution_disturbance(car_x,t_end,t_step,X_safe_tree,X_safe,ball_obs,record,Xfree)
                        if not success:
                            print('collision!!!!')
                        X_safe=X_safe_
                        X_safe_tree=KDTree(X_safe_)
                        car_trajectory_=np.stack(record)
                        Xfree=Xfree_
                        switch.append(car_trajectory_[-1,:2])
                        
                        print('car execution...')
                        t_end=T_end
                        success, record, train_new,train_f_new,car_x=car_execution_disturbance(car_x,t_end,t_step,X_safe_tree,X_safe,ball_obs,record,Xfree)
                        print('model updating...')
                        train=np.concatenate((train,train_new))
                        train_f=np.concatenate((train_f,train_f_new))
                        car_trajectory=np.stack(record)
                        
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
                        plt.plot(car_trajectory[:,0],car_trajectory[:,1],color='cyan')
                        
                        plt.plot(car_trajectory_[:,0],car_trajectory_[:,1],color='black') 
                        switch_points=np.stack(switch)
                        plt.scatter(switch_points[:,0],switch_points[:,1], color='black')
                    break
                    
        gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True).fit(train, train_f)

"""