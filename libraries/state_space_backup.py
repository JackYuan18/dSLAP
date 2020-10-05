#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 15:41:57 2019

@author: zqy5086@AD.PSU.EDU
"""
import numpy as np

class state:
    """
    a data structure for optimal time function that contains:
    x: state (np.array)
    u: control input at x
    w: disturbance at x
    t: the optimal time at x
    """
    def __init__(self, state):
        self.state = state
        self.u = None
        self.u_= None
        self.parents=dict()
        self.unsafe = False
        self.childs=dict()
        self.parents_=dict()
#        self.unsafe = False
        self.childs_=dict()
        self.U=dict()
        self.to_goal=None
        self.u_min=None
        self.X_temp_tree=None
        self.cov=dict()
        self.mu=dict()
        self.cov_=dict()
        self.mu_=dict()
        self.r=0
        self.from_u=None
#        self.add_childs=False

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
        
def gridsearch_construct(Xfree,p,Up,obs=None,fix=None):
        Dim = Xfree.dim
        x = [[]]*Dim
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
        X=[]
        Z=[]
        X_reduced=[]
        for i in x[0]:
            for j in x[1]:
                for k in x[2]:
                    y=np.array([i,j,k])
#                    ystate = state(y)
                    key=to_string([i,j,k])
                    z= Xfree.table[key]
                    z.u_min=None
                    for u in Up:
                        z.parents_[u]=set()
                        z.childs_[u]=set()
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
    
#    def recursive_assign(self,Rn,iter_Rn, dim, Dim, z):
#        try:
#            z[dim] = next(iter_Rn[dim])
##            print(z)
#            if dim == Dim - 1: #and: for fence escape
#                if Dim>1:
##                    if abs(z[0]-z[1])>0.1:
#
#                    y = np.array(z)
#    #                y = np.reshape(y,(self.dim,1))
#                    ystate = state(y)
#                    self.x.append(y)
#                    key=to_string(z)
#                        
#                    self.table[key]=ystate0.2  Evaluation scenario
#                else:
#                    y = np.array(z)
#                    ystate = state(y)
#                    self.x.append(y)
#                    key=to_string(z)
#                            
#                    self.table[key]=ystate
#                    
#                self.recursive_assign(Rn,iter_Rn,dim, Dim,z)
#            else:            
#                self.recursive_assign(Rn,iter_Rn,dim+1, Dim,z)
#        except StopIteration:
#            if dim == 0:
#                return None
#            else:
#                iter_Rn[dim] = iter(Rn[dim])
#                self.recursive_assign(Rn,iter_Rn,dim-1, Dim,z)
        
    def construct(self, p,Up,goal=None):

        x = [[]]*self.dim
        
        for dim in range(self.dim):
#              pp=(self.parameters[dim][1]-self.parameters[dim][0])*2**p+1
              x[dim] = np.linspace(self.parameters[dim][0],self.parameters[dim][1],2**p+1)
 
        for i in x[0]:
            for j in x[1]:
                for k in x[2]:
                    y=np.array([i,j,k])                    
                    ystate = state(y)
                    for u in Up:
                        ystate.parents[u]=set()
                        ystate.childs[u]=set()
#                        ystate.parents_[u]=set()
#                        ystate.childs_[u]=set()
#                        ystate.r=0
                    self.x.append(y)
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