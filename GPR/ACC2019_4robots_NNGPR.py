#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 16:40:01 2020

@author: zqy5086@AD.PSU.EDU
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 10:55:49 2019

@author: zqy5086@AD.PSU.EDU
"""
import sys
sys.path.append('/home/AD.PSU.EDU/zqy5086/Publication/ICRA2021/GPR')
sys.path.append('/home/AD.PSU.EDU/zqy5086/Publication/ICRA2021/libraries')
import random
import numpy as np
import matplotlib.pyplot as plt
from gp2 import gp,k
from math import sin,cos,sqrt,exp
import matplotlib
from car_dynamics import car
from scipy import spatial
from state_space import ball, gridsearch_construct, to_string,grid,d
from sklearn.neighbors import BallTree as KDTree
from safety_control import safety_iteration_GP
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 

import time
np.random.seed(10)  

def clear_utility(U_childs):
    for u in U_childs:
        for s in U_childs[u]:
            s.r=0
#            s.from_u=None

def assign_learning_utility(x,u,U_childs=None,Temp_set=None,aggregate=False):
    
    if aggregate:
#        print('aggregating...')
        for child in x.childs[u]:
            try:
#                R=sqrt(x.cov[u])
                R=x.r_parent+np.sqrt(np.abs(x.cov[u]))
#                if x.cov[u] >0:
#                    print('yes')
            except KeyError:
                R=x.r_parent
            except ValueError:                                  
                print(x.cov[u])
            if R>child.r:
           
                child.r=R
                child.from_u=x.from_u
                            
    else:
        for child in x.childs[u]:
            try:
                if sqrt(x.cov[u])>child.r:
                    child.r_parent=np.sqrt(np.abs(x.cov[u]))
                    child.from_u=u

#                    
            except:
                continue

                
    return Temp_set
                
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

def one_step_optimal_control(x):
    min_val=float('inf')
#    min_child=None
    u_min=None
    for u in x.u:
    
            Childs=list(x.childs[u])
            Childs_to_goal=[c.to_goal for c in Childs]
                      
            min_ind=np.argmin(Childs_to_goal)
            min_val_=Childs_to_goal[min_ind]
#            min_child_=Childs[min_ind]
            if min_val_<min_val:
                min_val=min_val_
#                min_child=min_child_
                u_min=u
    return min_val, u_min

def optimal_val_in_states(S,tk,goal=None,psi=1):

    if goal is None:
        min_val=min([(1-exp(-psi*(tk)-1))*s.to_goal-exp(-psi*tk-1)*s.r for s in S])
#        print('optimal')

        return min_val,None
    else:
        
        min_val=min([np.linalg.norm(s[0,:-1]-goal[0,:-1]) for s in S])

        return min_val,None
def optimal_control(U_childs,tk,goal=None,psi=1):
    min_val=float('inf')
    u_min=None
    for u in U_childs:
        
        if len(U_childs[u])>0:
            min_val_,from_u=optimal_val_in_states(U_childs[u],tk,goal,psi=psi)
#            print(u,min_val_)
            if min_val_<min_val:
                min_val=min_val_
                u_min=u
#                print('optimal', u,min_val)
#                if from_u is None:
#                    u_min=u
#                else:
#                    u_min=from_u
    return u_min
class robot:
    """
    each robot records its own state (x), data collected (X,y), prediction on Z_M (Z_M)
    """
    def __init__(self,init,dynamics,car, M,l,sigma,ID,size):
        self.x=init
        self.Z=None
        self.id=ID
        self.X=None
        self.X_=None
        self.Y=None
        
        self.R=[]
        
        self.gpr=None
        self.mu_M=None
        self.cov_M=None
        self.mu_M_hat=None
        self.cov_M_hat=None
        
        self.mu_R=None
        self.cov_R=None
        self.mu_R_hat=None
        self.cov_R_hat=None
        
        self.sigma=sigma
        
#        self.mu_full=None
#        self.cov_full=None
#        
#        self.theta=0
#        self.theta_=0
#        self.xi=0
#        self.xi_=0False
#        self.r_theta=0
#        self.r_xi=0
#        self.iota=0
#        self.iota_=0f to_obs<dist:
                
#        self.r_iota=0
    
        
        
        self.data=dict()
        self.data_keys=[]
        self.dataR=dict()
        self.dataR_keys=[]
        self.dataM=dict()
        self.dataM_keys=[]
        self.kdt=None                
        self.gamma=2
        
        self.goal=None
        self.Up=None
        self.X_safe_set=None
        self.X_safe=None
        self.Xfree=None
        self.dynamics=dynamics
        self.car=car       
        self.M=M
        self.l=l
        self.policy=None
        self.X_safe=None
#        self.X_safe_=None
        self.state_temp=None
        self.eps_p=None
        self.reach=None
        self.X_temp_tree=None
        self.X_temp=None
        self.hp=None
        self.state_temp=None
        
        self.stop=False
        self.dist_to_collision=[]
        self.size=size
        self.ica_time=[]
        self.oca_time=[]
        self.al_time=0
        self.AL_time=[]
        self.learning_time=0
        self.Learning_time=[]
    def to_goal(self):
        print('robot '+str(self.id)+' is '+str(np.linalg.norm(self.x[0,:-1]-self.goal[0,:-1]))+' away from goal')
        
    def count_inclusion(self):
        cnt=0
        CNT=0
        for x in self.state_temp:
        
            for u in x.cov:               
                truth=self.dynamics(x.state,u)-self.car(x.state,u)  
                mu=x.mu[u]
                cov=x.cov[u]
                if all(truth<=mu+2*np.sqrt(cov)) and all(truth>=mu-2*np.sqrt(cov)):
                    cnt=cnt+1
                CNT=CNT+1
        if CNT==0:
            self.R.append(0)
        else:
            self.R.append(cnt/CNT)
    def Reach(self,robot,init=False):
#        states=set()
        x=robot.nearest_X()
#        if init:
#            pass 
#        else:
#            x.gpr.predict(self.X[-20:,:],self.Y[-20:,:])
        for ind,u in enumerate(self.Up):
#            x=robot.x.reshape((3,))
            
            if init:
                z=x.state+robot.eps_p*robot.car(x.state,u)
            else:
                
                mu=x.gpr.mu_g[ind,:]
                cov=x.gpr.C_g[ind]
    #            mu,cov=robot.gp_local_single(np.concatenate((x,np.array([u]))))
                z=x.state+robot.eps_p*(mu+robot.car(x.state,u))
#            print(cov)
                std=np.sqrt(np.abs(cov))
            alpha_p=2*robot.hp+robot.l*robot.eps_p*robot.hp+robot.M*robot.l*robot.eps_p**2+0.5*(self.M*self.eps_p+self.hp)
            z=z.reshape(1,-1)
            if init:
                inds1 = self.X_temp_reduced_tree.query_radius(z[:,:-1],alpha_p+2*self.size)#aplt.gca().add_patch(circle1)lpha_p+self.gamma*std
            else:
                inds1 = self.X_temp_reduced_tree.query_radius(z[:,:-1],alpha_p+2*self.size+robot.gamma*std)
            inds=inds1[0]

            coordinates = [self.X_temp[ind] for ind in inds]
            keys = [to_string(s) for s in coordinates]

            states = [self.Xfree.table[key] for key in keys]
        return states
    def SI(self,x,Alpha_p):
        index=[]   
        alpha_p=Alpha_p
        for ind,u in enumerate(self.Up):
                                
            z=x.state+self.eps_p*self.car(x.state,u)
            while z[2]>1: 
                z[2]=z[2]-2
            while z[2]<-1:
                z[2] = z[2]+2
            z=z.reshape(1,-1)
#            print(z,alpha_p)
            inds1 = self.X_temp_tree.query_radius(z,alpha_p)
#            print(inds1)
            theta=z[0,2]
            if theta<-1+alpha_p:
                z[0,2]=1
                inds2 = self.X_temp_tree.query_radius(z,-theta+alpha_p-1)
                inds=np.concatenate((inds1[0],inds2[0]))
            elif theta>1-alpha_p:
                z[0,2]=-1
                inds2 = self.X_temp_tree.query_radius(z,-theta-alpha_p+1)
                inds=np.concatenate((inds1[0],inds2[0]))
            else:
                inds=inds1[0]
            inds=set(inds)
            coordinates = [self.X_temp[ind] for ind in inds]
            keys = [to_string(s) for s in coordinates]
            states = set([self.Xfree.table[key] for key in keys])
            unsafe = any([s.unsafe for s in states])
#            print(u,unsafe,inds)

            if unsafe:
                index.append(ind)                   
            else:
                for s in states:
                    s.parents[u].add(x)        
                    x.childs[u].add(s)

        x.u = np.delete(self.Up,index)

        if len(x.u)>0:
            x.unsafe=False
        else:
            x.unsafe=True
        
        
    def SI_GP(self, x,Alpha_p):
        index=[]    
        start=time.time()
        MU, COV=x.gpr.predict(self.X[-60:-40,:],self.Y[-60:-40,:])
        self.learning_time=self.learning_time+time.time()-start
        ss=x.state
        for ind,u in enumerate(self.Up):
            mu=MU[ind,:]

            x.cov[u]=COV[ind]

            x.mu[u]=mu
            z=ss+self.eps_p*(mu+self.car(ss,u))           
            std=sqrt(np.abs(x.cov[u]))           

            alpha_p=Alpha_p+self.gamma*std*self.eps_p
#            if d(ball_obs,z,True)>alpha_p+ball_radius:
            while z[2]>1: 
                z[2]=z[2]-2
            while z[2]<-1:
                z[2] = z[2]+2
            
            z=z.reshape(1,-1)
            inds1 = self.X_temp_tree.query_radius(z,alpha_p)
            theta=z[0,2]
            if theta<-1+alpha_p:
                z[0,2]=1
                inds2 =self.X_temp_tree.query_radius(z,-theta+alpha_p-1)
                inds=np.concatenate((inds1[0],inds2[0]))
            elif theta>1-alpha_p:
                z[0,2]=-1
                inds2 = self.X_temp_tree.query_radius(z,-theta-alpha_p+1)
                inds=np.concatenate((inds1[0],inds2[0]))
            else:
                inds=inds1[0]
            inds=set(inds)
            coordinates = [self.X_temp[ind] for ind in inds]
            keys = [to_string(s) for s in coordinates]
            states = [self.Xfree.table[key] for key in keys]
            unsafe = any([s.unsafe for s in states])
            
            if unsafe:
                index.append(ind)                   
            else:
                for s in states:
                    s.parents[u].add(x)        
                    x.childs[u].add(s)
            if len(x.childs[u])==0:
                x.cov[u]=0
   
        x.u = np.delete(self.Up,index)
        if len(x.u)>0:
            x.unsafe=False
        else:
            x.unsafe=True
    def inter_robot_collision_avoidance(self,robo_network,init=False):
        state_temp=self.oca_safe_states
        for robot in robo_network:
            if robot.id<self.id and not robot.stop:
#                print(robot.id,self.id)
                self.other_reach=self.Reach(robot,init)
                parent_update(self.other_reach)
                state_temp=state_temp-set(self.other_reach)
        if state_temp is not None and len(state_temp)>0:
            X_safe= [x.state for x in state_temp]
#            print('after ica:',len(X_safe))
            self.X_safe=KDTree(X_safe)
            
            self.X_safe_set=X_safe
    
    def obstacle_collision_avoidance_GP(self, obs,p,final=False):

        self.X_safe=None
        self.X_safe_set=None
        if final:
            hP = 2**(-p)/2
            eps_P = (hP/(self.M*self.l))**(1/2)
            alpha_P=2*hP+self.l*eps_P*hP+self.M*self.l*eps_P**2
            p=p-1
        
        hp = 2**(-p)/2
        eps_p = (hp/(self.M*self.l))**(1/2)
        alpha_p=2*hp+self.l*eps_p*hp+self.M*self.l*eps_p**2
        
        
        dist=0.8*(self.M*eps_p+hp)
        ball_radius=0.5*(self.M*eps_p+hp)+self.size
        ball_obs=ball(obs,ball_radius)

        need_update=[]
#        start=time.time()
        if final:
#            print('robot '+str(self.id))
            self.X_temp,self.state_temp, self.X_temp_reduced = gridsearch_construct(self.Xfree,p,self.Up,obs=ball_obs, final=final,goal=self.goal,dist=dist)
        else:
            self.X_temp,self.state_temp, self.X_temp_reduced = gridsearch_construct(self.Xfree,p,self.Up,obs=ball_obs) #identify obstacle unsafe state
#        print(time.time()-start)
        self.X_temp_tree=KDTree(self.X_temp)

        
        self.X_temp_reduced_tree=KDTree(self.X_temp_reduced)

                #fused GPR prediction at state_temp     
        self.learning_time=0
        for s in self.state_temp:
            if not s.unsafe:   
                if s.to_goal<dist and final:
                    self.hp=hP
                    self.eps_p=eps_P
                    
                    self.SI_GP(s,alpha_P)  
                else:
                    self.hp=hp
                    self.eps_p=eps_p
                    self.SI_GP(s,alpha_p)  
                if s.unsafe:                               
                    need_update.append(s)
        self.Learning_time.append(self.learning_time)
        parent_update(need_update) 

        safe_states=set()
#        X_safe=[]
        for x in self.state_temp:
            if not x.unsafe:
#                X_safe.append(x.state)
                safe_states.add(x)

        self.oca_safe_states=safe_states

    def obstacle_collision_avoidance(self,obs,p):
        
        self.hp = 2**(-p)/2
        self.eps_p = (self.hp/(self.M*self.l))**(1/2)
        
#        for ob in obs:
#        ball_obs=ball(obs,self.size)
        ball_obs=ball(obs,0.5*(self.M*self.eps_p+self.hp)+self.size)
        alpha_p=2*self.hp+self.l*self.eps_p*self.hp+self.M*self.l*self.eps_p**2

        #table construction based on dynamics
        need_update=[]
        
        self.X_temp,self.state_temp, X_temp_reduced = gridsearch_construct(self.Xfree,p,self.Up,obs=ball_obs) #identify obstacle unsafe state
        
        self.X_temp_tree=KDTree(self.X_temp)
    
        X_temp_reduced_tree=KDTree(X_temp_reduced)
        self.X_temp_reduced=X_temp_reduced
        self.X_temp_reduced_tree=X_temp_reduced_tree
        for s in self.state_temp:
            if not s.unsafe:   
                    self.SI(s,alpha_p)     
                    if s.unsafe:                               
                        need_update.append(s)

        parent_update(need_update) 

        safe_states=set()
        X_safe=[]
        for x in self.state_temp:
            if not x.unsafe:
                X_safe.append(x.state)
                safe_states.add(x)

        if len(X_safe)>0:
            self.X_safe=KDTree(X_safe)
            self.X_safe_set=X_safe
            self.oca_safe_states=safe_states
        else:
            raise NameError('No prior safe states!' )
    def nearest(self):
        (_,ind)=self.X_safe.query(self.x)
        z=self.X_safe.data[ind[0,0]]
        key=to_string(z)
        zstate=self.Xfree.table[key]
        return zstate
    def nearest_X(self):
        (_,ind)=self.X_temp_tree.query(self.x)
        z=self.X_temp_tree.data[ind[0,0]]
        key=to_string(z)
        zstate=self.Xfree.table[key]
        return zstate
    
    
    def active_learning(self,T,tk,psi=1):  
        
        x=self.nearest()     
      
        
        if x.u_min is None:

            Next_childs=dict()
            U_childs=dict()
    
            for u in x.u:
    
                U_childs[u]=x.childs[u]
                Next_childs[u]=x.childs[u]
                assign_learning_utility(x,u, U_childs)    
    
            
    #        for t in range(T-1): 
    #            Temp_set=dict()
    #            for u in x.u:
    #                Temp_set[u]=set()
            for u in x.u:
                for s in Next_childs[u]:
                    for v in s.u:
                        assign_learning_utility(s,v,U_childs,aggregate=True)
            for u in x.u:
                for s in Next_childs[u]:
                    for v in s.u:
                        U_childs[u]=U_childs[u].union(s.childs[v])
    
    
            
    #                            
    #            Next_childs=Temp_set
            x.u_min=optimal_control(U_childs,tk,psi=psi)
            x.r=0
            clear_utility(U_childs)
        return x.u_min
    
    def active_learning_mu(self,T,tk):  
        
        x=self.nearest()     

#        if x.u_min is None:
        Next_childs=dict()
        U_childs=dict()
        
        for u in x.u:
            U_childs[u]=list()
            Next_childs[u]=list()
            try:
                z= x.state+self.eps_p*(self.car(x.state,u)+x.mu[u])
            except KeyError:
                z= x.state+self.eps_p*(self.car(x.state,u))

            z=z.reshape((1,3))
#        robo_network.add_robot(r8)
            while z[0,2]>1: 
                z[0,2]=z[0,2]-2
            while z[0,2]<-1:
                z[0,2] = z[0,2]+2
            U_childs[u].append(z)
            Next_childs[u].append(z)
  
        for t in range(T-1): 
            Temp_set=dict()
            for u in x.u:
                Temp_set[u]=list()
            for u in x.u:
                for s in Next_childs[u]:
                    for v in self.Up:      
                        try:
                            mu,cov=self.gp_local_single(np.concatenate((s.state,np.array([u]))))                           
                            z= s.state+self.eps_p*(self.car(s.state,u)+mu)                                                        
                        except AttributeError:
                            z= s+self.eps_p*(self.car(s,u))    
                        U_childs[u].append(z)
                        Temp_set[u].append(z)

                            
            Next_childs=Temp_set
        x.u_min=optimal_control(U_childs,tk,goal=self.goal)
#            clear_utility(U_childs)x.cov[u]
        return x.u_min
    
    def active_learning_x(self,T,tk):  
        
        x=self.x.reshape((3,))  
        Next_childs=dict()
        U_childs=dict()
        
        for u in self.Up:
            U_childs[u]=list()
            Next_childs[u]=list()
            z= x+self.eps_p*(self.car(x,u))

            z=z.reshape((1,3))
            while z[0,2]>1: 
                z[0,2]=z[0,2]-2
            while z[0,2]<-1:
                z[0,2] = z[0,2]+2
            U_childs[u].append(z)
            Next_childs[u].append(z)
  

        
        for t in range(T-1): 
            Temp_set=dict()
            for u in self.Up:
                Temp_set[u]=list()
            for u in self.Up:
                for s in Next_childs[u]:
                    for v in self.Up:      
                        z= s+self.eps_p*(self.car(s,u))    
                        U_childs[u].append(z)
                        Temp_set[u].append(z)

                            
            Next_childs=Temp_set
        u=optimal_control(U_childs,tk,goal=self.goal)
#            clear_utility(U_childs)
        return u
    
    def fuse_gpr(self,Z_R):
        for d in range(3):
            Z_M=[z for z in self.dataM_keys if self.dataM[str(z)][d][1]<self.data[str(z)][d][1] and self.dataM[str(z)][d][2]<self.data[str(z)][d][1]]
                    
            if len(Z_M)<1:
                for zr in Z_R:
                    robot.dataR[str(zr)][d]=robot.data[str(zr)][d]
            else:
                robot.kdt=spatial.KDTree(np.array(Z_M))   
                ind=self.kdt.query(np.array(Z_R),p=2)
                z_train=np.array([self.kdt.data[i] for i in ind[1]])
                g=np.array([[k1(self.gpr.kernel,Z_R[i],z_train[i])*min(self.data[str(z_train[i])][d][1], self.data[str(Z_R[i])][d][1]) for i in range(len(Z_R))]])
                g=g/1
                k_star=g*np.array([[(self.data[str(z_train[i])][d][1])**(-1) for i in range(len(Z_R))]])
                
                u_=np.array([self.dataM[str(zm)][d][0]-self.data[str(zm)][d][0] for zm in z_train])
                
                u_local=np.array([[self.data[str(z)][d][0] for z in Z_R]])
                cov_local=np.array([[self.data[str(z)][d][1] for z in Z_R]]).T
                mu_fuse=k_star.T*u_+u_local.T
                cov_fuse=cov_local+k_star.T**2*np.array([self.dataM[str(zm)][d][1]-self.data[str(zm)][d][1] for zm in z_train])
                
                mu_iter=iter(mu_fuse)
                cov_iter=iter(cov_fuse)
                
                for zr in Z_R:
                    self.dataR[str(zr)][d]=(next(mu_iter),next(cov_iter))
                    
    def assignZR(self,Z_R):
        for z in Z_R:
            self.data_keys.append(z)
    def assignZM(self,Z_M):
        for z in Z_M:
            self.dataM_keys.append(z)
    def gp_setup(self,kernel,mean_fun):
        self.gpr=gp(kernel,mean_fun,self.sigma)   
    
    def gp_prep_local(self):
        self.gpr.gp_prep_local(self.X, self.Y)
#    def gp_full(self,Z_,X,Y):
#        self.mu_full,self.cov_full=self.gpr.predict(Z_,X,Y)
    def gp_full(self, Z):
        
        mu,cov=self.gpr.predict_local(Z)
        mu=np.reshape(mu,(mu.size,))
        cov=np.reshape(cov,(cov.size,))
        mu_iter=iter(mu)
        cov_iter=iter(cov)
        for z in Z:
            self.data[str(z)]=(next(mu_iter),next(cov_iter))
            
    def gp_full_3D(self, Z):
        Z[0,2]=Z[0,2]%2-1
        for dd in range(3):
            mu,cov=self.gpr.predict_local_MultiD(Z,dd)
            mu=np.reshape(mu,(mu.size,))
            cov=np.reshape(cov,(cov.size,))
            mu_iter=iter(mu)
            cov_iter=iter(cov)
            for z in Z:
                self.data[str(z)][dd]=(next(mu_iter),next(cov_iter))
    def gp_local_single(self, Z):
        MU=np.zeros((1,3))
        SIGMA=np.zeros((1,3))
        if Z[2]>1:
#            print('jump')
            (d1,ind1)=self.gpr.kdt.query(Z)
            Z[2]=Z[2]-2
            (d2,ind2)=self.gpr.kdt.query(Z)
            if d1<d2:
                ind=ind1
            else:
                ind=ind2
            z_train=self.gpr.kdt.data[ind] 
            for dd in range(3):
                MU[0,d],SIGMA[0,d]=self.gpr.predict_local_MultiD_got_train(Z,z_train,dd)
        
        elif Z[2]>-1:
#            print('jump')
            (d1,ind1)=self.gpr.kdt.query(Z)
            Z[2]=Z[2]+2
            (d2,ind2)=self.gpr.kdt.query(Z)
            if d1<d2:
                ind=ind1
            else:
                ind=ind2
            z_train=self.gpr.kdt.data[ind] 
            for dd in range(3):
                MU[0,dd],SIGMA[0,dd]=self.gpr.predict_local_MultiD_got_train(Z,z_train,dd)
        else:
#            print('normal')
            for dd in range(3):
                MU[0,dd],SIGMA[0,dd]=self.gpr.predict_local_MultiD(Z,dd)
#            MU[d]=np.reshape(mu,(mu.size,))U_childs[u_].remove(child)
               
#            SIGMA[d]=np.reshape(cov,(cov.size,))
        
#            mu_iter=iter(mu)
#            cov_iter=iter(cov)
#            for z in Z:
#                self.data[str(z)][d]=(next(mu_iter),next(cov_iter))
        return MU, SIGMA
    def run(self):
        self.x=self.x+self.f(self.x)*0.1
        
        self.X=np.vstack((self.X,self.x))
        
    
    def run_car(self,u):
        if self.X is None:
            self.X=np.hstack((self.x,np.array([[u]])))
            self.X_=np.hstack((self.x,np.array([[u]])))
        else:
            self.X=np.vstack((self.X,np.hstack((self.x,np.array([[u]])))))
            self.X_=np.vstack((self.X_,np.hstack((self.x,np.array([[u]])))))
        self.x=self.x+self.dynamics(self.x,u)*0.2
        while self.x[0,2]>1:
            self.x[0,2]=self.x[0,2]-2
#            print(self.x[0,2])
        while self.x[0,2]<-1:
            self.x[0,2]=self.x[0,2]+2
#            print(self.x[0,2])
    def run_car_no_sample(self,u):
        if self.X is None:
            self.X_=np.hstack((self.x,np.array([[u]])))
        else:
            self.X_=np.vstack((self.X_,np.hstack((self.x,np.array([[u]])))))
        self.x=self.x+self.dynamics(self.x,u)*0.2
        while self.x[0,2]>1:
            self.x[0,2]=self.x[0,2]-2
        while self.x[0,2]<-1:
            self.x[0,2]=self.x[0,2]+2

    def run_and_sample(self,eta):
        self.run()
        self.Y=np.vstack((self.Y,eta(self.x)))
    
    def run_and_sample_MultiD(self,tk,network,obs,psi=1):
        if np.linalg.norm(self.x[0,:-1]-self.goal[0,:-1])>0.06 and np.max(self.x[0,:-1])<2 and np.min(self.x[0,:-1])>-1:
            if self.X_safe is None:
                print('robot'+str(self.id)+' has no safe control! Random action is executed')
                ind=np.random.randint(len(self.Up)-1) 
                u=self.Up[ind]
                
            else:
                if self.x[0,1]>1 or self.x[0,1]<0 or self.x[0,0]>1 or self.x[0,0]<0:
                    start=time.time()
                    u=self.active_learning_x(2,tk)
                    self.al_time=self.al_time+time.time()-start
#                    print('active x')
                else:
                    start=time.time()
                    u=self.active_learning(2,tk,psi=psi)
                    self.al_time=self.al_time+time.time()-start
                    
#                    print('robot '+str(self.id)+' AL: ',time.time()-start)
            if u is None:
                start=time.time()
                u=self.active_learning_x(2,tk)
                self.al_time=self.al_time+time.time()-start

            x_dot=self.dynamics(self.x, u)
            if self.Y is None:
                self.Y=np.array([x_dot-self.car(self.x,u)])
            else:    
                self.Y=np.vstack((self.Y,x_dot-self.car(self.x,u)))
            
            self.M=max((self.M, np.max(np.abs(x_dot[:-1]))))
            
            self.run_car(u)
#            dist=float('inf')
            
            dist=d(obs,self.x[0],dist=True)
#            if to_obs<dist:
#                dist=to_obs
#            dist=d(obs,self.x[0],dist=True)
#            print('to obstacle:',dist)
            for robot in network:
                if robot.id != self.id and not robot.stop:
                    to_collision=np.linalg.norm(self.x[0,:-1]-robot.x[0,:-1])
                    if to_collision<dist:
                        dist=to_collision
                        if dist<self.size*2:
                            self.stop=True
                            robot.stop=True
                            print('Collision take place at robot '+str(robot.id)+'!!!!!!!!!!!!!!!!!!!!!!!!!')
#            print('to robots:',dist)
            if dist<self.size*2:
                self.stop=True
                print('Collision take place at robot '+str(self.id)+'!!!!!!!!!!!!!!!!!!!!!!!!!')
            else:
                self.dist_to_collision.append(dist)
                if len(self.Y)>1:
                    np.random.seed(len(self.Y))
                    ind=np.random.randint(len(self.Y)-1) 
                    x_=np.array([self.X[ind][:3]])
                    if abs(self.x[0,2]-x_[0,2])>1:
                        if x_[0,2]<0:
                            x_[0,2]=x_[0,2]+2
                        else:
                            x_[0,2]=x_[0,2]-2
                    u_=self.X[ind][3]
                    x_dot_=self.dynamics(x_,u_)
                    l_=np.abs(x_dot[:-1]-x_dot_[:-1])/np.linalg.norm((self.X[-1]-self.X[ind])*np.array([1,1,3.14,1]))
        #            print(self.X[-1],self.X[ind])
                    self.l=np.max((self.l, np.max(l_)))
        else:
            if np.max(self.x[0,:-1])>2 or np.min(self.x[0,:-1])<-1:
                print('robot'+str(self.id)+' out of bound!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#            print(self.X[-2],self.X[-1],self.id)
            else:
                print('robot'+str(self.id)+' reach goal!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            self.stop=True

            
    def run_and_sample_MultiD_no_sample(self,tk,network,obs):
        if np.linalg.norm(self.x[0,:-1]-self.goal[0,:-1])>0.05 and np.max(self.x[0,:-1])<2 and np.min(self.x[0,:-1])>-1:
            if self.X_safe is None:
                print('robot'+str(self.id)+' has no safe control! Random action is executed')
                ind=np.random.randint(len(self.Up)-1) 
                u=self.Up[ind]
            else:
                if self.x[0,1]>1 or self.x[0,1]<0 or self.x[0,0]>1 or self.x[0,0]<0:
                    u=self.active_learning_x(2,tk)
                else:
                    u=self.active_learning(2,tk)
            if u is None:
                u=self.X[-1,-1]#self.active_learning_mu(2,tk)


            self.run_car_no_sample(u)
            dist=float('inf')
            for ob in obs:
                to_obs=d(ob,self.x[0],dist=True)
                if to_obs<dist:
                    dist=to_obs
#            print('to obstacle:',dist)
            for robot in network:
                if robot.id != self.id and not robot.stop:
                    to_collision=np.linalg.norm(self.x[0,:-1]-robot.x[0,:-1])
                    if to_collision<dist:
                        dist=to_collision
                    if dist<self.size*2:
                        self.stop=True
                        robot.stop=True
                        print('Collision take place at robot '+str(robot.id)+'!!!!!!!!!!!!!!!!!!!!!!!!!')
                        print('robot '+str(robot.id)+': '+str(robot.x))
#            print('to robots:',dist)
            if dist<self.size*2:
                self.stop=True
                print('Collision take place at robot '+str(self.id)+'!!!!!!!!!!!!!!!!!!!!!!!!!onvehicle')
                print('robot '+str(self.id)+': '+str(self.x))
            else:
                self.dist_to_collision.append(dist)

        else:
            if np.max(self.x[0,:-1])>2 or np.min(self.x[0,:-1])<-1:
                print('robot'+str(self.id)+' out of bound!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#            print(self.X[-2],self.X[-1],self.id)
            else:
                print('robot'+str(self.id)+' reach goal!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            self.stop=True
        
             
        
class robot_network:
    def __init__(self,A):
        self.network=[]
        self.A=A
        
        self.M=0
        self.dataM=None
        self.X=None
        self.Y=None
        self.data=dict()
        self.data_keys=[]
#        self.sigma=sigma
        self.gpr=None
#   
    def control_init(self, car,car_disturbance, X_dim,M,l,Up,obs,P,kernel):
        for robot in self.network:
            robot.car=car
            robot.dynamics=car_disturbance
            Xfree=grid(X_dim)
            
            robot.Xfree=Xfree.construct(P,Up,robot.goal,kernel)
            robot.M=M
            robot.l=l
            robot.Up=Up
            if P==5:
                robot.obstacle_collision_avoidance(obs,P-1)
            else:
                robot.obstacle_collision_avoidance(obs,P)
            
            robot.inter_robot_collision_avoidance(self.network,init=True)
    def safe_planner(self,obs_1,p,GP=False,final=False):
        for robot in self.network:
            if not robot.stop:
                
#                print('robot'+str(robot.id))
                if GP:
                    start=time.time()
                    robot.obstacle_collision_avoidance_GP(obs_1,p,final)
                    robot.oca_time.append(time.time()-start-robot.learning_time)
#                    print('robot '+str(robot.id)+' oca: ',robot.oca_time)
                else:
                    robot.obstacle_collision_avoidance(obs_1,p)

        for robot in self.network:
            if not robot.stop:
#                print('robot'+str(robot.id))
                start=time.time()
                robot.inter_robot_collision_avoidance(self.network)
                robot.ica_time.append(time.time()-start)
#                print('robot '+str(robot.id)+' ica: ',robot.ica_time)
    def oca(self,obs,p):
        for robot in self.network:
            robot.obstacle_collision_avoidance_GP(obs,p)
    def add_robot(self,robot):
        self.network.append(robot)
    
    def distance_to_goal(self):
        for robo in self.network:
            robo.to_goal()
    def count_inclusion(self):
        for robo in self.network:
            robo.count_inclusion()
    
    
    
    
    
    def assignZR(self, Z_R):
        for robot in self.network:
            robot.assignZR(Z_R)
        for z in Z_R:
            self.data_keys.append(z)
    def assignZM(self,Z_M):
        for robot in self.network:
            robot.assignZM(Z_M)
    def distributed_gpr(self,Z_M,t):
        self.M=len(Z_M)
        self.Z_M=Z_M
        A=self.A(t)
        for index,robot in enumerate(self.network):
            child.r[u]=x.r+x.cov[u]
            mu_M=np.array([[robot.data[str(z)][0] for z in Z_M]]).T
            cov_M=np.array([[robot.data[str(z)][1] for z in Z_M]]).T
            
            r_theta=mu_M/cov_M
            theta=sum(A[index,ind]*(robo.theta_-robot.theta_) for ind,robo in enumerate(self.network))
            robot.theta=robot.theta_+theta+r_theta-robot.r_theta
            robot.r_theta=r_theta
            
            r_xi=1/cov_M
            xi=sum(A[index,ind]*(robo.xi_-robot.xi_)for ind,robo in enumerate(self.network))
            robot.xi=robot.xi_+xi+r_xi-robot.r_xi
            robot.r_xi=r_xi
                      
            r_iota=cov_M
            iota=sum(A[index,ind]*(robo.iota_-robot.iota_)for ind,robo in enumerate(self.network))
            robot.iota=robot.iota_+iota+r_iota-robot.r_iota
            robot.r_iota=r_iota
            
#            k_MM=np.reshape(np.diag(k(robot.gpr.kernel,Z_M,Z_M)),(self.M,1))+robot.sigma**2
            cov_M_hat=robot.xi**(-1)
            mu_M_hat=cov_M_hat*robot.theta
            
            muM=iter(mu_M_hat)
            covM=iter(cov_M_hat)
            iotaM=iter(robot.iota)
            for x in self.Z_M:
                robot.dataM[str(x)]=(next(muM),next(covM),next(iotaM))
#                robot.dataM_keys.add(x)
        for robot in self.network:
            robot.theta_=robot.theta
            robot.xi_=robot.xi
            robot.iota_=robot.iota
            
    def distributed_gpr_MultiD(self,Z_M,t):
        self.M=len(Z_M)
        self.Z_M=Z_M
        A=self.A(t)
        for d in range(3):
            for index,robot in enumerate(self.network):
                
                mu_M=np.array([[robot.data[str(z)][0][d] for z in Z_M]]).T
                cov_M=np.array([[robot.data[str(z)][1][d] for z in Z_M]]).T
                
                r_theta=mu_M/cov_M
                theta=sum(A[index,ind]*(robo.theta_[d]-robot.theta_[d]) for ind,robo in enumerate(self.network))
                robot.theta=robot.theta_[d]+theta+r_theta[d]-robot.r_theta[d]
                robot.r_theta[d]=r_theta
                
                r_xi=1/cov_M
                xi=sum(A[index,ind]*(robo.xi_[d]-robot.xi_[d])for ind,robo in enumerate(self.network))
                robot.xi[d]=robot.xi_[d]+xi+r_xi-robot.r_xi[d]
                robot.r_xi[d]=r_xi
                          
                r_iota=cov_M
                iota=sum(A[index,ind]*(robo.iota_[d]-robot.iota_[d])for ind,robo in enumerate(self.network))
                robot.iota[d]=robot.iota_[d]+iota+r_iota-robot.r_iota[d]
                robot.r_iota[d]=r_iota
                
    #            k_MM=np.reshape(np.diag(k(robot.gpr.kernel,Z_M,Z_M)),(self.M,1))+robot.sigma**2
                cov_M_hat=robot.xi[d]**(-1)
                mu_M_hat=cov_M_hat*robot.theta[d]
                
                muM=iter(mu_M_hat)
                covM=iter(cov_M_hat)
                iotaM=iter(robot.iota)
                for x in self.Z_M:
                    robot.dataM[str(x)][d]=(next(muM),next(covM),next(iotaM))
    #                robot.dataM_keys.add(x)
            for robot in self.network:
                robot.theta_[d]=robot.theta[d]
                robot.xi_[d]=robot.xi[d]
                robot.iota_[d]=robot.iota[d]
    def fuse_gpr(self, Z_R):

        for robot in self.network:
            
            Z_M=[z for z in robot.dataM_keys if robot.dataM[str(z)][1]<robot.data[str(z)][1] and robot.dataM[str(z)][2]<robot.data[str(z)][1]]
            
                
#            Z_R=robot.data_keys
            if len(Z_M)<1:
                for zr in Z_R:
                    robot.dataR[str(zr)]=robot.data[str(zr)]
            else:
#            Z_R=[z for z in robot.data_keys if z not in Z_M]
                robot.kdt=spatial.KDTree(np.array(Z_M))   
                ind=robot.kdt.query(np.array(Z_R),p=2)
                z_train=np.array([robot.kdt.data[i] for i in ind[1]])
                g=np.array([[k1(robot.gpr.kernel,Z_R[i],z_train[i])*min(robot.data[str(z_train[i])][1], robot.data[str(Z_R[i])][1]) for i in range(len(Z_R))]])
                g=g/1
                k_star=g*np.array([[(robot.data[str(z_train[i])][1])**(-1) for i in range(len(Z_R))]])
                
                u_=np.array([robot.dataM[str(zm)][0]-robot.data[str(zm)][0] for zm in z_train])
                
                u_local=np.array([[robot.data[str(z)][0] for z in Z_R]])
                cov_local=np.array([[robot.data[str(z)][1] for z in Z_R]]).T
                mu_fuse=k_star.T*u_+u_local.T
                cov_fuse=cov_local+k_star.T**2*np.array([robot.dataM[str(zm)][1]-robot.data[str(zm)][1] for zm in z_train])
                
                mu_iter=iter(mu_fuse)
                cov_iter=iter(cov_fuse)
                
                for zr in Z_R:
                    robot.dataR[str(zr)]=(next(mu_iter),next(cov_iter))
  
        
    def fuse_gpr_MultiD(self, Z_R):
#        for d in range(3):
            for robot in self.network:
                robot.fuse_gpr(Z_R)
#                Z_M=[z for z in robot.dataM_keys if robot.dataM[str(z)][d][1]<robot.data[str(z)][d][1] and robot.dataM[str(z)][d][2]<robot.data[str(z)][d][1]]
#                
#                    
#    #            Z_R=robot.data_keys
#                if len(Z_M)<1:
#                    for zr in Z_R:
#                        robot.dataR[str(zr)][d]=robot.data[str(zr)][d]
#                else:
#    #            Z_R=[z for z in robot.data_keys if z not in Z_M]
#                    robot.kdt=spatial.KDTree(np.array(Z_M))   
#                    ind=robot.kdt.query(np.array(Z_R),p=2)
#                    z_train=np.array([robot.kdt.data[i] for i in ind[1]])
#                    g=np.array([[k1(robot.gpr.kernel,Z_R[i],z_train[i])*min(robot.data[str(z_train[i])][d][1], robot.data[str(Z_R[i])][d][1]) for i in range(len(Z_R))]])
#                    g=g/1
#                    k_star=g*np.array([[(robot.data[str(z_train[i])][d][1])**(-1) for i in range(len(Z_R))]])
#                    
#                    u_=np.array([robot.dataM[str(zm)][d][0]-robot.data[str(zm)][d][0] for zm in z_train])
#                    
#                    u_local=np.array([[robot.data[str(z)][d][0] for z in Z_R]])
#                    cov_local=np.array([[robot.data[str(z)][d][1] for z in Z_R]]).T
#                    mu_fuse=k_star.T*u_+u_local.T
#                    cov_fuse=cov_local+k_star.T**2*np.array([robot.dataM[str(zm)][d][1]-robot.data[str(zm)][d][1] for zm in z_train])
#                    
#                    mu_iter=iter(mu_fuse)
#                    cov_iter=iter(cov_fuse)
#                    
#                    for zr in Z_R:
#                        robot.dataR[str(zr)][d]=(next(mu_iter),next(cov_iter))

    def run_and_sample(self,eta):
        for robot in self.network:
            robot.run_and_sample(eta)
    def run_and_sample_MultiD(self,tk,obs,psi=1):
        for robot in self.network:
            
            if not robot.stop:
                robot.run_and_sample_MultiD(tk,self.network,obs,psi=psi)
    def gpr_setup(self, kernel,mean_fun):
       def run_and_sample(self,eta):
        for robot in self.network:
            robot.run_and_sample(eta)
        for robot in self.network:
            robot.gp_setup(kernel,mean_fun)
    def gpr_prep(self):
        for robot in self.network:
            if not robot.stop:
                robot.gp_prep_local()
    def gpr_local(self,Z_R):
#        Z=np.concatenate((Z_M,Z_R))
        for robot in self.network:
            robot.gp_full(Z_R)
    
    def gpr_local_multiD(self,Z_R):
#        Z=np.concatenate((Z_M,Z_R))
        for robot in self.network:
            robot.gp_full_3D(Z_R)

            
    def full_gpr(self, Z_R,kernel,mean_fun):
        for robot in self.network:
            if self.X is None:
                self.X=robot.X
                self.Y=robot.Y
            else:
                self.X=np.vstack((self.X,robot.X))
                self.Y=np.vstack((self.Y,robot.Y))
        self.gpr=gp(kernel,mean_fun,self.sigma)   
        self.gpr.gp_prep_local(self.X,self.Y)
        
        mu,cov=self.gpr.predict_local(Z_R)
        mu=np.reshape(mu,(mu.size,))
        cov=np.reshape(cov,(cov.size,))

        mu_iter=iter(mu)
        cov_iter=iter(cov)
        for z in Z_R:
            self.data[str(z)]=(next(mu_iter),next(cov_iter))

#            for s in Next_childs[u]:
#                for v in s.u:
#                    as
            
    def print_network_results(self,sumError,plot,eta1):
        fontsize=14

        cbarsize=14
        ZZ=[]
        
#        x=Z_R[i,0]
#        y=Z_R[i,1]
        N=len(self.data_keys)
        for x in self.data_keys:
            truth=eta1(x)
#            ZZ.append(self.data[str(x)][0])def run_and_sample(self,eta):
        for robot in self.network:
            robot.run_and_sample(eta)
            ZZ.append(np.linalg.norm(self.data[str(x)][0]-truth))
        sumError.append(sum(ZZ)/N)
        if plot:
            fig5=plt.figure()
            n=int(sqrt(len(self.data_keys)))
            size=n
            X=np.linspace(0,10,size)
            Y=np.linspace(0,10,size)
            XX,YY=np.meshgrid(X,Y)
            plt.scatter(self.X[:,0],self.X[:,1])
    #        plt.scatter(self.X[0,0],self.X[0,1])
    #        plt.legend(prop={'size': 15})
            plt.contourf(XX.T,YY.T,np.reshape(ZZ,(n,n)),cmap='magma',labelsize=fontsize)
            cbar=plt.colorbar()
            cbar.ax.tick_params(labelsize=cbarsize)
    
            #    plt.title('Error of collaborative GPR for robot'+str(robo),fontsize=16)
                
            #    plt.xlabel('consensus on '+str((M*20)**2*100/361)+'% of test data',fontsize=18)
            fig5.savefig('fullerror.png')
        #plot of variance of fusion GPR
            fig6=plt.figure()

        ZZ=[]
        for x in self.data_keys:
            ZZ.append(self.data[str(x)][1])
        if plot:
            
            plt.contourf(XX.T,YY.T,np.reshape(ZZ,(n,n)),cmap='magma',labelsize=fontsize)
            cbar=plt.colorbar()
            cbar.ax.tick_params(labelsize=cbarsize) 
        #    plt.title('Variance of collaborative GPR for robot'+str(robo),fontsize=16)
        #    plt.xlabel('consensus on '+str((M*20)**2*100/361)+'% of test data',fontsize=18)
    #        plt.plot(self.X[:,0],self.X[:,1],label='robot'+str(robo+1))
    #        plt.scatter(self.X[0,0],self.X[0,1])
    #        plt.legend(prop={'size': 15})
            fig6.savefig('fullvar.png')
        
        return sumError

def learning(robo_network,eta, Z_M, T=100):

    for t in range(T):
        robo_network.run_and_sample(eta)
    
    robo_network.gpr_prep()
    robo_network.gpr_local(Z_M)     
    robo_network.distributed_gpr(Z_M,t)  
    
    
def learning_MultiD_no_communication_no_sample(robo_network,kernel,tk,obs,T=100):
    for t in range(T):
        for r in robo_network.network:
            if not r.stop:
                r.run_and_sample_MultiD_no_sample(tk,robo_network.network,obs)
#    robo_network.gpr_prep()
#
def learning_MultiD_no_communication(robo_network,kernel,tk,obs,T=100,psi=1):
#    for robot in robo_network.network:
#        robot.X_safe=robot.X_safe_
    for r in robo_network.network:
        r.al_time=0
    for t in range(T):
#        print(t)
        robo_network.run_and_sample_MultiD(tk,obs,psi=psi)
    for r in robo_network.network:
        if r.al_time>0:
            r.AL_time.append(r.al_time)
#    robo_network.gpr_prep()
#        print('t=',t, 'gpr local')
#    robo_network.gpr_local_multiD(Z_M)      
#        print('t=',t, 'gpr Z_M')
#    robo_network.distributed_gpr_MultiD(Z_M,t)  
def learning_MultiD(robo_network,eta, Z_M, T=100):
    cbarsize=14

    for t in range(T):
#        print('t=',t, 'sampling')
        robo_network.run_and_sample_MultiD()
#        print('t=',t, 'gpr training')
#        if t%10==0:
    robo_network.gpr_setup(kernel,mean_fun=None)
    robo_network.gpr_prep()
#        print('t=',t, 'gpr local')
    robo_network.gpr_local_multiD(Z_M)      
#        print('t=',t, 'gpr Z_M')
    robo_network.distributed_gpr_MultiD(Z_M,t)  


if __name__=='__main__':
    number=4
    dynamics=lambda x:np.random.multivariate_normal([0,0], np.eye(2), 1)
    eta=lambda x:sin(x[0,0]+x[0,1])+cos(x[0,0]-x[0,1])
    eta1=lambda x:sin(x[0]+x[1])+cos(x[0]-x[1])
    A=lambda t: 0.5*(1-(-1)**t)*np.array([[0.5,0,0.5,0],[0,0.5,0,0.5],[0.5,0,0.5,0],[0,0.5,0,0.5]])+0.5*(1+(-1)**t)*np.array([[0.5,0.25,0.,0.25],[0.25,0.5,0.25,0.],[0.,0.25,0.5,0.25],[0.25,0.,0.25,0.5]])

    sigma=0.1
#    kernel=lambda x,y:np.exp(-np.linalg.norm(x-y)**2/l)*sigma_f
    M=0.1
    R=40
    Z_M=get_Z_M(np.linspace(0,10,int(M*R+1)),np.linspace(0,10,int(M*R+1)))
    robo_network=robot_network(A)
    r1=robot(np.array([0,0,0]),dynamics, car, np.array([10,0,0]), 0.5, 0.5)
    r2=robot(np.array([1,0,0]),dynamics, car, np.array([10,0,0]), 0.5, 0.5)
    r3=robot(np.array([2,0,0]),dynamics, car, np.array([10,0,0]), 0.5, 0.5)
    r4=robot(np.array([4,0,0]),dynamics, car, np.array([10,0,0]), 0.5, 0.5)
    
    learning(robo_network,eta, Z_M)
#    fontsize=14
#    legendsize=10
#    cbarsize=14
#    np.random.seed(10) 
#    dynamics=lambda x:np.random.multivariate_normal([0,0], np.eye(2), 1)
#    eta=lambda x:sin(x[0,0]+x[0,1])+cos(x[0,0]-x[0,1])
#    eta1=lambda x:sin(x[0]+x[1])+cos(x[0]-x[1])
#    A=lambda t: 0.5*(1-(-1)**t)*np.array([[0.5,0,0.5,0],[0,0.5,0,0.5],[0.5,0,0.5,0],[0,0.5,0,0.5]])+0.5*(1+(-1)**t)*np.array([[0.5,0.25,0.,0.25],[0.25,0.5,0.25,0.],[0.,0.25,0.5,0.25],[0.25,0.,0.25,0.5]])
#    robo=0
#    l=0.5
#    sigma_f=1
#    sigma=0.1
#    kernel=lambda x,y:np.exp(-np.linalg.norm(x-y)**2/l)*sigma_f
#    number=4
#    T=100
#    R=40
#    MM=[0.5]
#    sumError=[]
#    sumError_local=[]
#    sumError_fuse=[]
#    plot=False
#    
#    for M in MM:
##    M=0.4
#        np.random.seed(10) 
#        print('T = ',T,' M = ',M,' R = ',R)
#        Z_M=get_Z_M(np.linspace(0,10,int(M*R+1)))
#        Z_R=get_Z_R(np.linspace(0,10,int(R+1)))
#        robo_network=robot_network(number,dynamics,eta,sigma,Z_M,Z_R)
#        robo_network.gpr_setup(kernel,mean_fun=None)
#        ZZ=[]
#    
##    Z=np.zeros((R-1,R-1))
#        N=len(Z_R)
#        for x in Z_R:    
#            truth=eta1(x)
##            ZZ.append(r1.data[str(x)][0])
#            ZZ.append(np.linalg.norm(truth))
#        SE=sum(ZZ)/N
#        sumError.append(SE)
#        for i in range(number):
#            sumError_local.append([SE])
#            sumError_fuse.append([SE])
#        for t in range(T):
#            print('t=',t, 'sampling')
#            robo_network.run_and_sample(eta)
#            print('t=',t, 'gpr training')
#    #        if t%10==0:
#            robo_network.gpr_prep()
#            print('t=',t, 'gpr local')
#            robo_network.gpr_local(Z_R)      
#            print('t=',t, 'gpr Z_M')
#            robo_network.distributed_gpr(Z_M,A(t))  
#            print('t=',t,'fusing gpr')
#            robo_network.fuse_gpr()
#            robo_network.full_gpr(Z_R,kernel,mean_fun=None)
#            for robo in range(len(robo_network.network)):
#                sumError_local[robo],sumError_fuse[robo]=print_result(robo,T,M,sumError_local[robo],sumError_fuse[robo],plot)
#            sumError=robo_netwo#fused GPR prediction at s       rk.print_network_results(sumError,plot)
#        
#    #    r1=robot(np.array([[5,5]]).T,dynamics,eta,sigma)
#    #    for t in range(T*10):
#    #        r1.run_and_sample(eta)
#    #    r1.gp_prep()
#    #    r1.gp_full(Z_R)
#    #    plt.contourf(XX,YY,np.reshape(r1.mu_R,(n,n)))   
#            
#    #visualization of function to learn
#        lw=4
#        if plot:
#            fig0=plt.figure()
#            size=R-1
#            X=np.linspace(0,10,size)
#            Y=np.linspace(0,10,size)
#            XX,YY=np.meshgrid(X,Y)
#            Z=np.zeros((size,size))
#            for i in range(size):
#                for j in range(size):
#                    Z[i][j]=eta1([X[i],Y[j]])
#            plt.contourf(XX.T,YY.T,Z,cmap='magma')
#            cbar=plt.colorbar()
#            cbar.ax.tick_params(labelsize=cbarsize) 
#            fig0.savefig('ground truth of eta.png')
#    #        plt.title('ground truth of eta',fontsize=15)
#        #print trajectory of robots
#            fig1=plt.figure(figsize=[4.8,4.0])
#            for i,robo in enumerate(robo_network.network):
#                if i ==0:
#                    ln='o'
#                elif i==1:
#                    ln='s'
#                elif i==2:
#                    ln='*'
#                else:
#                    ln='X'
#                plt.plot(robo.X[:,0],robo.X[:,1])
#                plt.scatter(robo.X[0,0],robo.X[0,1],marker=ln,s=lw*30,label='robot '+str(i+1))
#                plt.legend(prop={'size':15})
#    #            plt.xlabel('T = '+str(T),fontsize=18)
#            plt.xlim(0,10)
#            plt.ylim(0,10)
#            fig1.savefig('trajectory_T'+str(T)+'M'+str(M*100)+'.png')
#    #plot of error of full gp  
##        for i in range(number):
#        lw=4
#        
#        for i in range(4):
#            figerror=plt.figure()
#            plt.plot(sumError,'-',linewidth=lw,label='cNN-GPR')
##            if i==0:
#            lb1='local GPR'
#            lb2='RaDGPR'
##            else:
##                lb1=None
##                lb2=None
#            plt.plot(sumError_local[i],'--', linewidth=lw,label=lb1)
#            plt.plot(sumError_fuse[i],':', linewidth=lw,label=lb2)
##            plt.plot(0.8,0.8,color=[1-0.25*i, 0, 1-0.25*i],label='robot'+str(i+1))
##            plt.legend(prop={'size':15},loc=9, bbox_to_anchor=(0.8,1.2), ncol=3)
#            plt.xlabel('robot'+str(i+1), fontsize=18)
##            plt.ylabel('Average predictive mean errors',fontsize=18)
#            figerror.savefig('temporalerror'+str(i+1)+'.png')
