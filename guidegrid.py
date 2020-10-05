#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 11:30:25 2019

@author: zqy5086@AD.PSU.EDU
"""

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
from state_space import State_Space,grid,d,state
from car_dynamics import car_execution_disturbance
from safety_control import safety_iteration_GP, SI, SI_GP
import pickle

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
def policy_search(center,X_safe,X_safe_tree,Xfree,GuideGrid,model=None,gpr=None):
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
        dist=1000
        control=None
        for u in z.u:
            test=np.concatenate((center,u))
            mu=gpr.predict(np.matrix(test)) 
            x=center+eps_p*mu
            dist_=np.linalg.norm(x-GuideGrid.to_visit)
            if dist_<dist:
                dist=dist_
                control=u

            
            
        return control,z


class guidepoint:
    def __init__(self,state):
        self.state=state
        self.val=-1000000
        self.virtual_val=-1000000
        self.neighbor=[]
        self.parent=None

class GuideGrid:
    def __init__(self,Xfree,control,obs,p):
        self.table={}
        self.x=[]
        self.r=0.5**p
        self.min_val=0
        self.vitural_min_val=0
        self.future_min_val=0            
        self.tree=None
        self.to_visit=guidepoint([])
        self.to_visit.val=2
        self.to_Visit=None
        
        Dim = Xfree.dim
        x = [[]]*Dim
        
        for dim in range(Dim):
              pp=(Xfree.parameters[dim][1]-Xfree.parameters[dim][0])*2**p+1
              x[dim] = np.linspace(Xfree.parameters[dim][0],Xfree.parameters[dim][1],pp)

        for u in control:
            for i in x[0]:
                for j in x[1]:
                    for k in x[2]:
                        y=np.array([i,j,k,u])
                        s=state(y[:3])
                        ystate = guidepoint(y)
                        key=to_string([i,j,k,u])
                        
                        if obs is not None:
                           if d(obs,s):
                                self.table[key]=ystate
                                self.x.append(y)
 
                        else:
                            self.table[key]=ystate
                            self.x.append(y)
                            
        self.tree=KDTree(self.x,leaf_size=30,metric='manhattan')                       
        
        
    def search_nearest(self,x):
        
        
#        nearest=np.zeros(len(x))
#        u=x[-1]
#        
#        nearest=np.where(x%self.r>self.r/2,self.r*(x//self.r+1),self.r*(x//self.r))
#        th=nearest[2]
        while x[2]>1:
            x[2]-=2
        while x[2]<-1:
            x[2]+=2
        
        nearest=self.tree.query(np.array([x]), return_distance=False)
        nearest=self.x[nearest[0][0]]
        
#        nearest=np.where(nearest>1,1,nearest)
#        nearest=np.where(nearest<0,0,nearest)
#        nearest[-1]=u
#        nearest[2]=th
#        for i in range(len(x)-1):
#            if x[i]%self.r>self.r/2:
#                nearest[i]=min(self.r*(x[i]//self.r+1),
#            else:
#                nearest[i]=self.r*(x[i]//self.r)
#        nearest[-1]=x[-1]
        
        nearest=self.table[to_string(nearest)]
        return nearest
    
    def search_neighbor(self,x):
        neighbor=[]
        x=x.state
        X=[]
        for i in range(len(x)-1):
            X.append([x[i]-self.r,x[i],x[i]+self.r])
        X.append([x[-1]-0.15,x[-1],x[-1]+0.15])
        for i in X[0]:
            for j in X[1]:
                for k in X[2]:
                    for u in X[3]:                    
                        key=to_string([i,j,k,u])
                        if key==to_string(x):
                            continue
                        try: 
                            neighbor.append(self.table[key])
                        except KeyError:
                            pass
        return neighbor
    
    def construct_graph(self):
        for k in self.table:
            
            self.table[k].neighbor=self.search_neighbor(self.table[k])
            
    def value_iteration(self,x):
        old=x.val
        x.val=max([x.val for x in x.neighbor])-1
        num=np.argmax([x.val for x in x.neighbor])
        x.parent=x.neighbor[num]
        if x.val<self.to_visit.val:
#            self.min_val=x.val
            self.to_visit=x
            self.to_Visit=[x]
        if x.val==self.to_visit.val:
            self.to_Visit.append(x)
        if x.val!=old:
            return x.neighbor
        else:
            return None
                       
    def virtual_value_iteration(self,x):
        
        old=x.virtual_val
        x.virtual_val=max([x.virtual_val for x in x.neighbor])-1
        if x.virtual_val<self.virtual_min_val:
            self.virtual_min_val=x.virtual_val
        if x.virtual_val!=old:
            return x.neighbor
        else:
            return None
#            for y in x.neighbor:
#                self.virtual_value_iteration(y)
                
    def back_to_reality(self):
        self.virtual_min_val=self.min_val
        for k in self.table:
            self.table[k].virtual_val=self.table[k].val
            
    def find_control(self,x,u,U,eps_p):
        min_dist=100000
        future_min_dist=100000
        current=np.concatenate((x,[u]))
        index=np.argmin([np.linalg.norm(current-y.state) for y in self.to_Visit])
        to_visit=self.to_Visit[index]
        self.to_visit=to_visit
        for u in U:
            p=np.concatenate((x,[u]))
            x=x+eps_p*model(x,u)
            x=x+eps_p*model(x,u)            
            dist=np.linalg.norm(p-to_visit.state)
            for v in U:
                pp=np.concatenate((x,[v]))
                future_dist=np.linalg.norm(pp-to_visit.state)
                if future_dist<future_min_dist:
                    future_min_dist=future_dist
            distance=dist+future_min_dist
            if future_min_dist<min_dist:
                min_dist=future_min_dist
                control=u
        return control
    
    def find_control_MPC(self,x,u,U,eps_p):
        min_dist=100000
        future_min_dist=100000
        current=np.concatenate((x,[u]))
        index=np.argmin([np.linalg.norm(current-y.state) for y in self.to_Visit])
        to_visit=self.to_Visit[index]
        self.to_visit=to_visit
        for u in U:
            p=np.concatenate((x,[u]))            
#            x=x+eps_p*model(x,u)            
#            dist=np.linalg.norm(p-to_visit.state)
            x=x+eps_p*model(x,u)
            for v in U:
                x=x+eps_p*model(x,u)
                pp=np.concatenate((x,[v]))
                future_dist=np.linalg.norm(pp-to_visit.state)
                if future_dist<future_min_dist:
                    future_min_dist=future_dist
            distance=dist+future_min_dist
            if distance<min_dist:
                min_dist=distance
                control=u
        return control
                
                
    def guide(self,x,U,eps_p):
        Val=-100000
        
        for u in U:
            p=np.concatenate((x,[u]))

            nearest=self.search_nearest(p)
            if nearest.val==1:
                val_=self.min_val
               
                x=x+eps_p*model(x,u)           
                
                future_val=2
                for v in U:
                    pp=np.concatenate((x,[v]))
                    future_nearest=self.search_nearest(pp)
                    if future_nearest.val==1:
                        future_val_=self.min_val
                    else:
                        self.back_to_reality()
                        future_nearest.virtual_val=1
                        update=nearest.neighbor
                        
                        while len(update)>0:
                            need_update=[]
                            for y in update:
                                if y.virtual_val!=1:
                                    need_update_=self.virtual_value_iteration(y)
                                    if need_update_ is not None:
                                        need_update+=need_update_
                            update=set(need_update)
                        
                        future_val_=self.virtual_min_val
                    if future_val_<future_val:
                        future_val=future_val_
                       
            else:
                self.back_to_reality()
                nearest.virtual_val=1
                
                update=nearest.neighbor
                        
                while len(update)>0:
                    need_update=[]
                    for y in update:
                        if y.virtual_val!=1:
                            need_update_=self.virtual_value_iteration(y)
                            if need_update_ is not None:
                                need_update+=need_update_
                    update=set(need_update)
                val_=self.virtual_min_val
                
                x=x+eps_p*model(x,u)
                x=x+eps_p*model(x,u)
                future_val=2
                for v in U:
                    pp=np.concatenate((x,[v]))
                    future_nearest=self.search_nearest(pp)
                    if future_nearest.val==1:
                        future_val_=self.min_val
                    else:
                        self.back_to_reality()
                        nearest.virtual_val=1
                        future_nearest.virtual_val=1
                        update=nearest.neighbor
                        
                        while len(update)>0:
                            need_update=[]
                            for y in update:
                                if y.virtual_val!=1:
                                    need_update_=self.virtual_value_iteration(y)
                                    if need_update_ is not None:
                                        need_update+=need_update_
                            update=set(need_update)
                            
                        future_val_=self.virtual_min_val
                    if future_val_<future_val:
                        future_val=future_val_
           
            if val_+future_val>Val:
                Val=val_+future_val
                control=u
        return control
                
                
        
           

def progression(car_model,center,X_safe,X_safe_tree,Xfree,hp,ball_obs):
    t_end = 730.
    t_start = 0.
    t_step=0.1
    ZZ=[]
    ys=[]
    z1=None
    z2=None
 
    while t_start<t_end:
        x = state(center)                
        if not d(ball_obs,x):
            return 0,z1,z2#,U0
        ys.append(center)
        
        u,zz=policy_search(center,X_safe,X_safe_tree,Xfree)
        ZZ.append((center,zz.state))
        z1,z2,_=np.vstack(ys).T  
#        U.append(u)
        center=center+t_step*car_model(center,u)   
        if center[0]<-0.01 or center[0]>1.01 or center[1]<-0.01 or center[1]>1.01:
            return 1, z1,z2
        
        t_start=t_start+t_step
    z11,z22,z=np.vstack(ys).T
    return 1, z11,z22#,U #,ZZ    
    

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

    
    obs_1 = State_Space(3,[0.47,0.52,0.25,0.75,-1,1])
    Xfree = grid([[0,1],[0,1],[-1,1]])  
    Xfree.construct(P,Up) 
    plot_progress=True
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
    print('computing backup controller')
    
    0
    hP=2**(-P)
    eps_P = (hP/(2.5*M*l))**(1/2)
    guidegrid=GuideGrid(Xfree,Up,ball(obs_1,M*eps_P+hP),1)
    guidegrid.construct_graph()
    record=[initial]
    center=initial
    history=None
    nearest_history=None
    u=0.0
    for t in range(1500):
        x = np.concatenate((center,[u]))
        center=center+1*car(center,u)
        record.append(center)
        
        nearest=guidegrid.search_nearest(x)
#        if nearest is not nearest_history:
#            nearest_history=nearest
#            print(nearest.state,t)
        
        if nearest.val!=1:
            nearest.val=1
            need_update=nearest.neighbor            
#        need_update=guidegrid.value_iteration(nearest)
            while len(need_update)>0:
                need_update_=[]
                for y in need_update:
                    if y.val!=1:
                        to_update=guidegrid.value_iteration(y)
                        if to_update is not None:
                            need_update_+=to_update
                need_update=need_update_
        if guidegrid.to_visit.state is not history:
            print(guidegrid.to_visit.state,guidegrid.to_visit.val,t)
            history=guidegrid.to_visit.state
        u=guidegrid.find_control2(center,u,Up,eps_P)
        
        
        
        
    car_trajectory=np.stack(record)
    plt.plot(car_trajectory[:,0],car_trajectory[:,1],color='blue')       
    plt.xlim([-0.5,1.5])
    plt.ylim([-0.5,1.5])
            
        
        

  