#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 15:41:57 2019

@author: zqy5086@AD.PSU.EDU
"""
import numpy as np
from recursive_gp import gp

class state:
    """
    a data structure for optimal time function that contains:
    x: state (np.array)
    u: control input at x
    w: disturbance at x
    t: the optimal time at x
    """
    def __init__(self, state,kernel):
        self.state = state
        self.u = None

        self.parents=dict()
        self.unsafe = False
        self.childs=dict()

        self.childs_=dict()
        self.U=dict()
        self.to_goal=None
        self.u_min=None
        self.X_temp_tree=None
        self.cov=dict()
        self.mu=dict()

        self.r=0
        self.r_parent=0
        self.from_u=None
        self.gpr=gp(kernel)
#        self.gpr=dict()
        self.kernel=kernel

def d(ball, x, dist=None):
    try:
        diff = abs(x.state[:-1]-ball.center[:-1])-np.array([ball.width/2, ball.length/2])
    except AttributeError:
        diff = abs(x[:-1]-ball.center[:-1])-np.array([ball.width/2, ball.length/2])
    distance = np.max(diff)
    
    if dist is None:
    
        if distance>=ball.radius:
            return True
        else:
            return False
    else:
        return distance

class ball:
    """
    if area ball:
        input center as [[x_,x],[y_, y]]
    
    """
    def __init__(self, center, radius):
        if len(center)==0:
            
            self.center = np.mean(np.reshape(np.asarray(center.parameter),(len(center.parameter)//2,2)), axis=1)
            self.Ball = True
            self.width = center.parameter[1]-center.parameter[0]
            self.length = center.parameter[3]-center.parameter[2]
        else:
            self.center = center
            self.Ball = False
            self.width = None
            self.length = None
        self.radius = radius     
        
def gridsearch_construct(Xfree,p,Up,obs=None,fix=None,final=False,goal=None,dist=None):
        Dim = Xfree.dim
        x = [[]]*Dim
        X=[]
        Z=[]
        X_reduced=[]
        if final:
            p=p+1
#            print(p)
            if fix is None:
                for dim in range(Dim):
    #                  pp=(Xfree.parameters[dim][1]-Xfree.parameters[dim][0])*2**p+1
                      x[dim] = np.linspace(Xfree.parameters[dim][0],Xfree.parameters[dim][1],2**p+1)
    
            else:
                for dim in range(Dim):
                      if dim!=fix[0]:
    #                      pp=(Xfree.parameters[dim][1]-Xfree.p1arameters[dim][0])*2**p+1
                          x[dim] = np.linspace(Xfree.parameters[dim][0],Xfree.parameters[dim][1],2**p+1)
                      else:
                          x[fix[0]] = [fix[1]]


            for i in x[0]:
                for j in x[1]:
                    for k in x[2]:
                        y=np.array([i,j,k])
                        if np.linalg.norm(y[:2]-goal[0,:2])<dist:
    #                    ystate = state(y)
                            key=to_string([i,j,k])
                            z= Xfree.table[key]
                            z.u_min=None
                            for u in Up:
                                z.parents[u]=set()
                                z.childs[u]=set()
                            if obs is None:
                                z.unsafe=False
                            elif d(obs,z.state):
                                z.unsafe=False
                               
                            else:                   
                                z.unsafe=True
                            X_reduced.append(y[:-1])
                            X.append(y)                      
                            Z.append(z)
            p=p-1
                            
                                       
      
            
        if fix is None:
            for dim in range(Dim):
#                  pp=(Xfree.parameters[dim][1]-Xfree.parameters[dim][0])*2**p+1
                  x[dim] = np.linspace(Xfree.parameters[dim][0],Xfree.parameters[dim][1],2**p+1)

        else:
            for dim in range(Dim):
                  if dim!=fix[0]:
#                      pp=(Xfree.parameters[dim][1]-Xfree.p1arameters[dim][0])*2**p+1
                      x[dim] = np.linspace(Xfree.parameters[dim][0],Xfree.parameters[dim][1],2**p+1)
                  else:
                      x[fix[0]] = [fix[1]]

#        keys=[]
        
        for i in x[0]:
            for j in x[1]:
                for k in x[2]:
                    y=np.array([i,j,k])
#                    ystate = state(y)
                    key=to_string([i,j,k])
                    z= Xfree.table[key]
                    z.u_min=None
                    for u in Up:
                        z.parents[u]=set()
                        z.childs[u]=set()
                    if obs is None:
                        z.unsafe=False
#                    elif i==0 or j==0 or j==1 or i==1:
#                        z.unsafe=True
                    elif d(obs,z.state):
#                        safe=True
##                        for ob in obs:
#                        safe= bool(d(obs,z))*safe
#                        if safe:
                        z.unsafe=False
                       
                    else:                   
                        z.unsafe=True
                    X_reduced.append(y[:-1])
                    X.append(y)                      
                    Z.append(z)
        
        return X,Z,X_reduced





def to_string(z):
    key=''
    for i in z:
        key=key+str(i)
    return key


class State_Space:
    
    def __init__(self, dim, parameter):
        self.dim = dim
        self.parameter = parameter
        self.samples=[]
        self.length = len(parameter)
    
    
    def measure(self):
        L = 1
        for n in range(self.dim):
            L = L*(self.parameter[2*n+1]-self.parameter[2*n])
        return L
    
    def sample(self):
        s = np.zeros((self.dim,1))
        for n in range(self.dim):
            s[n] = np.random.uniform(self.parameter[2*n], self.parameter[2*n+1],1)
        self.samples.append(state(s))   
        return state(s)
        
    def __len__(self):
        return 0        
        

class grid:
    """
    define grid
    input: 
    dimension (list of lists): specify the [min, max] for each dimension
    """
    
    def __init__(self,dimension):
        self.dim = len(dimension)
        self.parameters = dimension
        self.x = [] 
        self.tree = None
        self.table={}

        
    def construct(self, p,Up,goal,kernel):

        x = [[]]*self.dim
#        p=p-1
        for dim in range(self.dim):
#              pp=(self.parameters[dim][1]-self.parameters[dim][0])*2**p+1
              x[dim] = np.linspace(self.parameters[dim][0],self.parameters[dim][1],2**p+1)
 
        for i in x[0]:
            for j in x[1]:
                for k in x[2]:
                    y=np.array([i,j,k])                    
                    ystate = state(y,kernel)
                    TEST=None
                    for u in Up:
                        ystate.parents[u]=set()
                        ystate.childs[u]=set()
                        test=np.concatenate((y,np.array([u])))
#                        ystate.gpr=gp(ystate.kernel)
#                        ystate.gpr[u].gp_prep(np.reshape(test,(1,4)),1)
                        if TEST is None:
                            TEST=test
                        else:
                            TEST=np.vstack((TEST,test))

                    self.x.append(y)

                    ystate.gpr.gp_prep(TEST,3)
                    key=to_string([i,j,k])
                    self.table[key]=ystate
                    ystate.to_goal=np.linalg.norm(ystate.state[:-1]-goal[0,:-1])


        return self
#    def kdtree_search(self, center, radius):
#        inds = self.tree.query_radius(center.reshape(1,-1),radius, count_only=False,return_distance=False)
#        coordinates = [self.x[ind] for ind in inds[0]]
#        keys = [to_string(s) for s in coordinates]
#        states = [self.table[key] for key in keys]
#        
#        T=[x.vn_ for x in states]
#        t = min(T)
#        return t, states
#    
#    def safe_kdtree_search(self, center):
#        inds = self.tree.query(center.reshape(1,-1),27)
#        coordinates = [self.x[ind] for ind in inds[0]]
#        keys = [to_string(s) for s in coordinates]
#        states = [self.table[key] for key in keys]
#        unsafe = any([s.unsafe for s in states])
#        
#
#        return unsafe,states
#    
#    def get_safe_kdtree_search(self, center, radius):
#        inds = self.tree.query_radius(center.state.reshape(1,-1),radius, count_only=False,return_distance=False)
#        coordinates = [self.x[ind] for ind in inds[0]]
#        keys = [to_string(s) for s in coordinates]
#        states = [self.table[key] for key in keys]
#
#        for x in states:
#            x.safe=True
#            x.s = center.s
#            x.u = center.u
#            
#    def get_neighbor(self,center):
#        ind = self.tree.query(center.reshape(1,-1), k=1, return_distance=False, dualtree=False, breadth_first=False)
#        coordinate = self.x[ind]
#        key = to_string(coordinate)
#        state = self.table[key]
#        return state.s