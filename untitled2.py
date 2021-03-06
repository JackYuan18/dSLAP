#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 17:59:38 2019

@author: zqy5086@AD.PSU.EDU
"""

import multiprocessing as mp
results=[]
def collect_result(result):
        global results
        results.append(result)
        print('called')
        print(results)
def policy_search(center,X_safe,X_safe_tree,Xfree):
        ind = X_safe_tree.query(center.reshape(1,-1),1)
        coordinate = X_safe[ind[1][0][0]]
        key = to_string(coordinate)
        z = Xfree.table[key]
        u=z.u[0]
        return u  
def testing(i,X,Y):
    ZZ=[]
    for j in range(10):
        center = np.array([X[i][j],Y[i][j], angle])
        if len(X_safe)==0:
            ZZ.append(-1)
        else:
            z,z1,z2 = progression(car_model,center,X_safe,X_safe_tree,Xfree,hp,ball_obs)
            ZZ.append(z)
    return (i,ZZ)

def progression(car_model,center,X_safe,X_safe_tree,Xfree,hp,ball_obs):
    t_end = 100.
    t_start = 0.
    t_step=0.1

    ys=[]
    while t_start<t_end:
        ys.append(center)
        
        u=policy_search(center,X_safe,X_safe_tree,Xfree)
        z1,z2,_=np.vstack(ys).T  
        
        center=center+t_step*car_model(center,u)   
        if center[0]<-0.05 or center[0]>1.05 or center[1]<-0.05 or center[1]>1.05:
            return 1, z1,z2
        x = state(center)                
        if not d(ball_obs,x):
            return 0,z1,z2
        t_start=t_start+t_step
    z11,z22,z=np.vstack(ys).T
    return 1, z11,z22       
def testtest():
    global results
    results=[]
       
    
    
    size=10
    Y, X = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size)) 
    _,Z = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size)) 
    ball_obs = ball(obs_1,0)
    X_safe = [x.state for x in state_temp if not x.unsafe]
    left = obs_1.parameter[0]
    down = obs_1.parameter[2]
    width = obs_1.parameter[1]-obs_1.parameter[0]
    height = obs_1.parameter[3]-obs_1.parameter[2]   
    if len(X_safe)>0:
        X_safe_tree=KDTree(X_safe)
    
    plt.figure()
    
    
               
    pool = mp.Pool(mp.cpu_count())
    for i in range(10):
        print(i)
        pool.apply_async(testing, args=(i,X,Y),callback=collect_result)
    pool.close()
    pool.join()
    results.sort(key=lambda x: x[0])
    for i in range(10):
        for j in range(10):
            Z[i][j]=results[i][1][j]
    
        
                
    plt.pcolormesh(X,Y,Z,vmin=-1, vmax=1)
    plt.title('Initial Value: ['+str(initial[0])+','+str(initial[1])+','+str(initial[2])+']')
    ##        
    #        plt.plot(z1,z2)
    plt.colorbar() 
    rectangle=plt.Rectangle((left,down),width,height,fc='red')
    plt.gca().add_patch(rectangle)
