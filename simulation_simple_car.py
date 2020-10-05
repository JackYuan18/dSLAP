#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 14:40:10 2019

@author: zqy5086@AD.PSU.EDU
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 22:41:16 2019

@author: robotics
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 13:17:05 2019

@author: zqy5086@AD.PSU.EDU
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 14:35:10 2019

@author: zqy5086@AD.PSU.EDU
"""
from math import ceil, floor,exp,sqrt,sin,cos,tan,pi
import numpy as np
import sys
import time
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import multiprocessing as mp
from functools import partial
from scipy import signal
from tempfile import TemporaryFile
import matplotlib.patches as pat

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
        self.w = None
        self.dim = len(state)
        self.s=1
        self.s_=1
        self.childs=[]
        self.flag = False
        self.unsafe = False
        self.safe = False
        
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
    
    def recursive_assign(self,Rn,iter_Rn, dim, Dim, z):
        try:
            z[dim] = next(iter_Rn[dim])
#            print(z)
            if dim == Dim - 1: #and: for fence escape
                if Dim>1:
#                    if abs(z[0]-z[1])>0.1:

                    y = np.array(z)
    #                y = np.reshape(y,(self.dim,1))
                    ystate = state(y)
                    self.x.append(y)
                    key=to_string(z)
                        
                    self.table[key]=ystate
                else:
                    y = np.array(z)
                    ystate = state(y)
                    self.x.append(y)
                    key=to_string(z)
                            
                    self.table[key]=ystate
                    
                self.recursive_assign(Rn,iter_Rn,dim, Dim,z)
            else:            
                self.recursive_assign(Rn,iter_Rn,dim+1, Dim,z)
        except StopIteration:
            if dim == 0:
                return None
            else:
                iter_Rn[dim] = iter(Rn[dim])
                self.recursive_assign(Rn,iter_Rn,dim-1, Dim,z)
        
    def construct(self, p,obs):
        x = [[]]*self.dim
        Xobs=[]
        for dim in range(self.dim):
              
              x[dim] = np.linspace(self.parameters[dim][0],self.parameters[dim][1],2**p+1)
 
        for i in x[0]:
            for j in x[1]:
                for k in x[2]:
                    y=np.array([i,j,k])                    
                    ystate = state(y)
                    if not d(obs,ystate):
                        ystate.s=0
                        ystate.flag = True
                        ystate.unsafe = True
                        Xobs.append(x)
                    
                    self.x.append(y)
                    key=to_string([i,j,k])
                    self.table[key]=ystate
        #assign states
#        iter_Rn = [iter(xi) for xi in x]
#        z = np.zeros(self.dim)
#        dim = 0
#        self.recursive_assign(x,iter_Rn, dim, self.dim, z)
        self.tree=KDTree(self.x, leaf_size=2)
        return Xobs
    def kdtree_search(self, center, radius):
        inds = self.tree.query_radius(center.reshape(1,-1),radius, count_only=False,return_distance=False)
        coordinates = [self.x[ind] for ind in inds[0]]
        keys = [to_string(s) for s in coordinates]
        states = [self.table[key] for key in keys]
        
        T=[x.vn_ for x in states]
        t = min(T)
#        ind = T.index(t)
#        print(min(T))
        return t, states
    
    def safe_kdtree_search(self, center, radius):
        inds = self.tree.query_radius(center.reshape(1,-1),radius, count_only=False,return_distance=False)
        coordinates = [self.x[ind] for ind in inds[0]]
        keys = [to_string(s) for s in coordinates]
        unsafe = any([self.table[key].flag for key in keys])
        
#        print(min(T))
        return unsafe
        

def to_string(z):
    key=''
    for i in z:
        key=key+str(i)
    return key
    

                
def gridsearch(x,grid,alpha):
    
    dim_index = []
    for i in range(x.dim):
        index=[]
        low = (x.state[i]-alpha)*grid.resolution
        high = (x.state[i]+alpha)*grid.resolution
        ind = ceil(low)
        
        while ind<=floor(high):
            index.append(ind)
            dim_index.append(index/grid.resolution)
            ind=ind+1
    
    results=[]
    results= gridsearch_construct(dim_index, x.dim)
    
    keys=[]
    keys=[to_string(x) for x in results]
    
    states=[grid.table[key] for key in keys]
    return min([x.vn for x in states])

def gridsearch_construct(Xfree,p,obs=None,fix=None):
        Dim = Xfree.dim
        x = [[]]*Dim
        if fix is None:
            for dim in range(Dim):
                  
                  x[dim] = np.linspace(Xfree.parameters[dim][0],Xfree.parameters[dim][1],2**p+1)

        else:
            for dim in range(Dim):
                  if dim!=fix[0]:
                      x[dim] = np.linspace(Xfree.parameters[dim][0],Xfree.parameters[dim][1],2**p+1)
                  else:
                      x[fix[0]] = [fix[1]]

        keys=[]
        for i in x[0]:
            for j in x[1]:
                for k in x[2]:
                    y=np.array([i,j,k])
                    ystate = state(y)
                    if obs is None:
                        key=to_string([i,j,k])
                        keys.append(key)
                    
                    elif d(obs,ystate):
                        key=to_string([i,j,k])
                        keys.append(key)

        
        return keys
        
            
def gridsearch_recursive_assign(Rn,iter_Rn, dim, Dim, z):
        coordinates = []
        try:
            z[dim] = next(iter_Rn[dim])
#            print(z)
            if dim == Dim - 1: #and: for fence escape              
                if abs(z[0]-z[1])>0.1:                                            
                        coordinates.append(to_string(z))                                   
                gridsearch_recursive_assign(Rn,iter_Rn,dim, Dim,z)
            else:            
                gridsearch_recursive_assign(Rn,iter_Rn,dim+1, Dim,z)
        except StopIteration:
            if dim == 0:
                return coordinates
            else:
                iter_Rn[dim] = iter(Rn[dim])
                gridsearch_recursive_assign(Rn,iter_Rn,dim-1, Dim,z)   
                
                
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
        
        
def set_ball(S, ball, manipulation):
#    if manipulation is 'minus':
#        if not ball.Ball:
#            return [x for x in S if np.linalg.norm(x.state-ball.center)>ball.radius]
#        else:
#            
#            return [x for x in S if  np.linalg.norm(np.max([abs(x.state-ball.center)-np.array([[ball.width], [ball.length]]),np.zeros((x.dim,1))],axis=1))>ball.radius]    
        
    if manipulation is 'intersection':
        return S.kdtree_search(ball.center,ball.radius)
        
        
def d(ball, x):
#    print(x.state, ball.center)
    
    
    if np.linalg.norm(np.max([abs(x.state[:-1]-ball.center[:-1])-np.array([ball.width, ball.length]),np.zeros(x.dim-1,)],axis=0))>=ball.radius:
        return True
    else:
        return False
        


def find_t(u,W,x,h,alpha,S):
    t=0
    S = None
    for w in W:
        z=state(x.state+h*f(u,w))

        T,s=S.kdtree_search(z.state,alpha)
        if T>t:
            t=T
            S=s
    return t,S

#      
def f(u,w):
        s=np.zeros((2))
        s[0] = u
        s[1] = w
        return s
      
def sub_VI(S, U, x,h,alpha):

    t_=float('inf')
    X = []
    for u in U:
        if x.state[0]-x.state[1]>0.1:
            w = 0.1
        else:
            w = -0.1

        z=state(x.state+h*f(u,w))
        t,s=S.kdtree_search(z.state,alpha)
        
        X=X+s
        if t<t_:
            t_=t
#            X = s
#    if X is None:
#        X =s
    return t_,X


def get_grid(i,X,Y, eps_p,hp,alpha_p,size): 
        Z=[[]]*size
        for j in range(size):
            if abs(X[i][j]-Y[i][j])<0.1:
                Z[j]=1
            elif X[i][j]>0 and X[i][j]<1: 
            
                t,_=sub_VI(Xfree,Up,state([X[i][j],Y[i][j]]),eps_p,alpha_p)
                Z[j] = 1-exp(-(eps_p-hp+t))
            else:
                Z[j] = 0
        return Z

def get_errors(i,X,Y, eps_p,hp,alpha_p,size, Xfree): 

        Z=[[]]*size
        for j in range(size):
            if abs(X[i][j]-Y[i][j])<0.1:
                Z[j]=0
            elif X[i][j]>0 and X[i][j]<1: 
            
                t,_=sub_VI(Xfree,Up,state([X[i][j],Y[i][j]]),eps_p,alpha_p)
                Z[j] = t
                
#                Z[j] = 1-exp(-(eps_p-hp+t))
                
                if Y[i][j]<0.1:
#                    t=1-exp(-(1-X[i][j])*10)
                    t = (1-X[i][j])*10
                else:
#                    t=1-exp(-X[i][j]*10)
                    t = X[i][j]*10
                    
                Z[j] = abs(Z[j]-t)
            else:
                Z[j] = 0
                
           
        return sum(Z)   
   
def get_error(eps_p,hp,alpha_p,size,pool,Xfree):
    Y, X = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size)) 
    _,Z = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size)) 
    Z=[pool.apply_async(get_errors,args=(i,X, Y, eps_p,hp,alpha_p,size,Xfree)) for i in range(size)]
    ZZZ = [p.get() for p in Z]
    error = sum(ZZZ)
    error = error/(size**2)
    return error

   
def plot_set_value(size,processes):
    global Z
    Y, X = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size)) 
    _,Z = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size)) 

    start=time.time()
    pool=mp.Pool(processes=6)
    Z=[pool.apply_async(get_grid,args=(i,X,Y,size)) for i in range(size)]
    end = time.time()
    print(end-start)
        
    plt.figure()
    plt.pcolormesh(X,Y,Z)
    plt.colorbar() 
    



def car(x,u):
    """
    x[0]=x
    x[1]=y
    x[2]=theta
    u[0]=v
    u[1]=phi    
       """  
    v = 0.1
    L = 0.025
    X = np.zeros((3))
    X[0] = v*cos(x[2])
    X[1] = v*sin(x[2])
    X[2] = v/L*tan(u)
    return X
    
def SI(x,Xfree,Up,alpha,Xobs,eps_P):
    den = len(Up)
    index=[]
    for ind,u in enumerate(Up):
        z=state(x.state+eps_P*car(x.state,u))
        unsafe = Xfree.safe_kdtree_search(z.state,alpha)
        
#        if states is not None:
#            x.childs=x.childs+states
#        else:
#            index.append(ind)
            
        if unsafe:
            index.append(ind)#                    num = num +1
                
    x.u = np.delete(Up,index)
    x.s = len(x.u)/den
    

if __name__=="__main__":
    #system setup
    pool = mp.Pool(processes=6)
    sys.setrecursionlimit(1500000)
    P = 6
    p = 1
    M = 0.1
    l = 0.1
    
    hP = 2**(-P)
    eps_P = (hP/(M*l))**(1/2)
    alpha_P = 2*hP+l*eps_P*(hP+M*eps_P)    
        
    #space setup
    Xfree = grid([[0,1],[0,1],[0,2*pi]])   
    
    #goal setup
    Xgoal1 = State_Space(2,[0.8,0.8,0.9,0.9,0,2*pi])
#    Bgoal1 = ball(Xgoal1,0.1)

    #obstacle setup
    obs_1 = State_Space(3,[0.4,0.6,0.4,0.6,0,2*pi])
    ball_obs_1 = ball(obs_1,M*eps_P+hP)
    
    
    
    #control setup
    Up = np.linspace(-0.1,0.1,5)

    #disturbance setup
    
    
    #dynamics paramters
    print('grid construction')
    Xobs=Xfree.construct(P,ball_obs_1)

    
    #error and timing
    Error = [[]]*P
    start=time.time()
    E = []
    size = 200
    t=0
            
            
    
    #planner computation, multigrid
    for p in range(1,5):
        hp = 2**(-p)
        eps_p = (hp/(M*l))**(1/2)
        alpha_p = 2*hp+l*eps_p*(hp+M*eps_p)        
        
        Convergence = False
        
        Xobs_p=Xobs
        #table construction based on dynamics
        need_update=[]
        print('getting X_temp')
        X_temp = gridsearch_construct(Xfree,p,ball_obs_1)
        print('Done getting X_temp')
        num = 0
        for x in X_temp:
            s = Xfree.table[x]
            
            print('p = ',p,'x ', num)
            num=num+1
            
            #identify initial unsafe states
            if d(ball_obs_1,s): #check if out of the obstable
                        SI(s,Xfree,Up,alpha_P+l*hp,Xobs_p,eps_P)
                        s.s_=s.s
                        if s.s==0:
                            s.flag = True
                            s.unsafe = True
                            Xobs_p.append(s)
                        else:
                            s.flag = False
                            need_update.append(s)
#            else:
#                        s.s=0
#                        s.flag = True
#                        Xobs_p.append(s)
        #update value, dynamic programming
        print('updating...')
        Kp=[]
        while not Convergence:
            Convergence = True           

            for x in need_update: 
                #we can remove the ones already in Kp, for further speedup
                flag = any([s.flag for s in x.childs])
                if x.flag==False and flag == True:
                        Kp.append(x)
                        need_update.remove(x)
                        
                        Convergence=False
            for x in Kp:
                    
                    SI(x,Xfree,Up,alpha_P+l*hp,Xobs_p,eps_P)
                    if x.s==0:
                        x.flag=True
                        x.unsafe = True
                        need_update.remove(x)
                        Kp.remove(x)
                        Xobs_p.append(x)   
                    if x.s==x.s_: #value converge and safe
                        x.flag = True
                        need_update.remove(x)
                        Kp.remove(x)
                        x.safe = True
                    else:
                        x.s_=x.s
         
        

#        t = time.time()-start+t
#        error = get_error(eps_p,hp,alpha_p,size,pool,Xfree)
#        E.append([t,error])
#        start=time.time()
        
                    

    end = time.time()
    theta = np.linspace(0,2*pi,2**(p-1)+1)
    #plot grid
    angle = theta[-1]
    X_temp = gridsearch_construct(Xfree,p,fix=[2,angle])
    T = [-float(Xfree.table[x].s) for x in X_temp]
    X1 = [float(Xfree.table[x].state[0]) for x in X_temp]
    X2 = [float(Xfree.table[x].state[1]) for x in X_temp]
#    vn = 1-np.exp(T)
    plt.figure(1)
    rectangle=plt.Rectangle((0.4,0.4),0.2,0.2,fc='red')
    plt.scatter(X1,X2, c=T)
    plt.gca().add_patch(rectangle)
    plt.colorbar()
    plt.xlabel('Evader')
    plt.ylabel('Pursuer')
    plt.title('Angle = '+str(angle))
    print('Time elapsed:', end-start)
    
    
    #plot obstacle circle
    X1 = [float(Xfree.table[x].state[0]) for x in X_temp]
    X2 = [float(Xfree.table[x].state[1]) for x in X_temp]           
    
    Y1 = [float(Xfree.table[x].state[0]) for x in X_temp if d(ball_obs_1,Xfree.table[x])]
    Y2 = [float(Xfree.table[x].state[1]) for x in X_temp if d(ball_obs_1,Xfree.table[x])]  
#    vn = 1-np.exp(T)
    plt.figure(2)
    plt.scatter(X1,X2)
    plt.scatter(Y1,Y2)
    rectangle=plt.Rectangle((0.4,0.4),0.2,0.2,fc='red')
    plt.gca().add_patch(rectangle)
    plt.title('plot obstacle circle')
#    plt.colorbar()
    
    
    #plot unsafe state
    X1 = [float(Xfree.table[x].state[0]) for x in X_temp]
    X2 = [float(Xfree.table[x].state[1]) for x in X_temp]           
    
    Y1 = [float(Xfree.table[x].state[0]) for x in X_temp if Xfree.table[x] in Xobs_p]
    Y2 = [float(Xfree.table[x].state[1]) for x in X_temp if Xfree.table[x] in Xobs_p]  
#    vn = 1-np.exp(T)
    plt.figure(3)
    plt.scatter(X1,X2)
    plt.scatter(Y1,Y2)
    rectangle=plt.Rectangle((0.4,0.4),0.2,0.2,fc='red')
    plt.gca().add_patch(rectangle)
    plt.title('plot unsafe states')
#    plt.colorbar()
    
    #plot interpolated value map    
#    Y, X = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size)) 
#    _,Z = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size)) 
#    Z=[pool.apply_async(get_grid,args=(i,X, Y, eps_p,hp,alpha_p,size)) for i in range(size)]
#    ZZZ = [p.get() for p in Z]      
#    plt.figure(2)
#    plt.pcolormesh(X,Y,ZZZ)
#    plt.colorbar() 
 
    
    
    
    
    
    
    
        

