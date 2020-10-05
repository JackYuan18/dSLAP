#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 10:55:49 2019

@author: zqy5086@AD.PSU.EDU
"""

import numpy as np
import matplotlib.pyplot as plt
from gp2 import gp,k
from math import sin,cos,sqrt
import matplotlib
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 
class robot:
    """
    each robot records its own state (x), data collected (X,y), prediction on Z_M (Z_M)
    """
    def __init__(self,init,dynamics,eta,sigma):
        self.f=dynamics
        self.x=init
        
        self.X=init
        self.Y=np.array([eta(init)])
        
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
        
        self.mu_full=None
        self.cov_full=None
        
        self.theta=0
        self.theta_=0
        self.xi=0
        self.xi_=0
        self.r_theta=0
        self.r_xi=0
        
    def gp_setup(self,kernel,mean_fun):
        self.gpr=gp(kernel,mean_fun,self.sigma)   
    
    def gp_prep(self):
        self.gpr.gp_prep(self.X, self.Y)
#    def gp_full(self,Z_,X,Y):
#        self.mu_full,self.cov_full=self.gpr.predict(Z_,X,Y)
    def gp_full(self, Z):
        self.mu_full,self.cov_full=self.gpr.predict(Z,self.X)
#        self.cov_full=np.diag(Sigma_full)
    def gp_M(self, Z_M):
        self.mu_M,Sigma_M=self.gpr.predict(Z_M,self.X)
        self.cov_M=np.reshape(np.diag(Sigma_M),(len(Z_M),1))
    def gp_R(self,Z_R):
        self.mu_R,Sigma_R=self.gpr.predict(Z_R,self.X)
        self.cov_R=np.reshape(np.diag(Sigma_R),(len(Z_R),1))
    def run(self):
        self.x=self.x+dynamics(self.x)*0.1
        self.X=np.vstack((self.X,self.x))
    def run_and_sample(self,eta):
        self.run()
        self.Y=np.vstack((self.Y,eta(self.x)))
        
class robot_network:
    def __init__(self,number,dynamics,eta,sigma):
        self.network=[robot(np.random.rand(1,2)*6+3,dynamics,eta,sigma) for i in range(number)]
        self.A=None
        self.size=number
        self.M=0
    def distributed_gpr(self,Z_M,A):
        self.M=len(Z_M)
        self.A=A
        for index,robot in enumerate(self.network):
            r_theta=robot.mu_M/ robot.cov_M
            theta=sum(self.A[index,ind]*(robo.theta_-robot.theta_) for ind,robo in enumerate(self.network))
            robot.theta=robot.theta_+theta+r_theta-robot.r_theta
            robot.r_theta=r_theta
            
            r_xi=1/robot.cov_M
            xi=sum(self.A[index,ind]*(robo.xi_-robot.xi_)for ind,robo in enumerate(self.network))
            robot.xi=robot.xi_+xi+r_xi-robot.r_xi
            robot.r_xi=r_xi
            
#            if index==1:
#                print(' robot.xi=', robot.xi)
            
            
            k_MM=np.reshape(np.diag(k(robot.gpr.kernel,Z_M,Z_M)),(self.M,1))+robot.sigma**2
            robot.cov_M_hat=(self.size*robot.xi-(self.size-1)*k_MM**(-1))**(-1)
            robot.mu_M_hat=self.size*robot.cov_M_hat*robot.theta
        for robot in self.network:
            robot.theta_=robot.theta
            robot.xi_=robot.xi
    def fuse_gpr(self,Z_R):
        for robot in self.network:
            L=np.matmul(robot.cov_full[self.M:,:self.M],np.linalg.inv(robot.cov_full[:self.M,:self.M]))
            
            mu_=robot.mu_M_hat-robot.mu_M
#            print(mu_)
            robot.mu_R_hat=np.matmul(L,mu_)+robot.mu_R
#            print(robot.mu_R_hat-robot.mu_R)
            robot.cov_R_hat=robot.cov_R+np.diag(np.matmul(np.matmul(L,(robot.cov_M_hat*np.eye(self.M)-robot.cov_M*np.eye(self.M))),L.T)).reshape((len(robot.cov_R),1))
    def run_and_sample(self,eta):
        for robot in self.network:
            robot.run_and_sample(eta)
    def gpr_setup(self, kernel,mean_fun):
        for robot in self.network:
            robot.gp_setup(kernel,mean_fun)
    def gpr_prep(self):
        for robot in self.network:
            robot.gp_prep()
    def gpr_local(self,Z_M,Z_R):
        for robot in self.network:
            robot.gp_full(np.vstack((Z_M,Z_R)))
            robot.gp_M(Z_M)
            robot.gp_R(Z_R)
def get_Z_M(linspace):
    Z_M=[]
    for i in linspace:
        for j in linspace:
            Z_M.append(np.array([i,j]))
    return np.array(Z_M)

def get_Z_R(linspace):
    Z_R=[]

    for i in linspace:
        for j in linspace:
            a=np.array([i,j])
#            if not any((a == x).all() for x in Z_M):
            Z_R.append(a)
    return np.array(Z_R)
                
def print_result(robo,T,M):
    fontsize=14
    legendsize=6
    cbarsize=14
    fig2=plt.figure()
    r1=robo_network.network[robo]
#    Z=np.vstack((Z_M,Z_R))
    n=int(sqrt(len(Z_R)))
    ZZ=[]
    size=n
    X=np.linspace(0,10,size)
    Y=np.linspace(0,10,size)
    XX,YY=np.meshgrid(X,Y)
#    Z=np.zeros((R-1,R-1))
    for i in range(len(r1.mu_R)):
        x=Z_R[i,0]
        y=Z_R[i,1]
        truth=eta1([x,y])
        ZZ.append(np.linalg.norm(r1.mu_R[i]-truth))
    plt.plot(r1.X[:,0],r1.X[:,1],label='robot'+str(robo))
    plt.scatter(r1.X[0,0],r1.X[0,1])
    plt.legend(prop={'size': 15})
    plt.contourf(XX.T,YY.T,np.reshape(ZZ,(n,n)),cmap='magma',labelsize=fontsize)
    cbar=plt.colorbar()
    cbar.ax.tick_params(labelsize=cbarsize) 
#    if M*100%10==0:
#        M=M/10
    fig2.savefig('localerror'+str(T)+'M0_'+str(M)+'robot'+str(robo)+'.png')
#plot of variance of local GPR
    fig3=plt.figure()
    ZZ=[]
    for i in range(len(r1.mu_R)):
        ZZ.append(r1.cov_R[i])
    plt.contourf(XX.T,YY.T,np.reshape(ZZ,(n,n)),cmap='magma',labelsize=fontsize)
    cbar=plt.colorbar()
    cbar.ax.tick_params(labelsize=cbarsize) 
#    plt.title('Variance of local GPR for robot'+str(robo),fontsize=16)
#    plt.xlabel('consensus on '+str((M*20)**2*100/361)+'% of test data',fontsize=18)
    plt.plot(r1.X[:,0],r1.X[:,1],label='robot'+str(robo))
    plt.scatter(r1.X[0,0],r1.X[0,1])
    plt.legend(prop={'size': 15})
#    if M*100%10==0:
#        M=M/10
    fig3.savefig('localvar'+str(T)+'M0_'+str(M)+'robot'+str(robo)+'.png')
#plot of error of fusion  
    fig4=plt.figure()
    ZZ=[]
    for i in range(len(r1.mu_R)):
        x=Z_R[i,0]
        y=Z_R[i,1]
        truth=eta1([x,y])
        ZZ.append(np.linalg.norm(r1.mu_R_hat[i]-truth))
    plt.plot(r1.X[:,0],r1.X[:,1],label='robot'+str(robo))
    plt.scatter(r1.X[0,0],r1.X[0,1])
    plt.legend(prop={'size': 15})
    plt.contourf(XX.T,YY.T,np.reshape(ZZ,(n,n)),cmap='magma',labelsize=fontsize)
    cbar=plt.colorbar()
    cbar.ax.tick_params(labelsize=cbarsize) 
#    plt.title('Error of collaborative GPR for robot'+str(robo),fontsize=16)
    if M*100%10==0:
        M=M/10
#    plt.xlabel('consensus on '+str((M*20)**2*100/361)+'% of test data',fontsize=18)
    fig4.savefig('fusionerror'+str(T)+'M0_'+str(M)+'robot'+str(robo)+'.png')
#plot of variance of fusion GPR
    fig5=plt.figure()
    ZZ=[]
    for i in range(len(r1.mu_R)):
        ZZ.append(r1.cov_R_hat[i])
    plt.contourf(XX.T,YY.T,np.reshape(ZZ,(n,n)),cmap='magma',labelsize=fontsize)
    cbar=plt.colorbar()
    cbar.ax.tick_params(labelsize=cbarsize) 
#    plt.title('Variance of collaborative GPR for robot'+str(robo),fontsize=16)
#    plt.xlabel('consensus on '+str((M*20)**2*100/361)+'% of test data',fontsize=18)
    plt.plot(r1.X[:,0],r1.X[:,1],label='robot'+str(robo))
    plt.scatter(r1.X[0,0],r1.X[0,1])
    plt.legend(prop={'size': 15})
#    if M*100%10==0:
#        M=M/10
    fig5.savefig('fusionvar'+str(T)+'M0_'+str(M)+'robot'+str(robo)+'.png')
  
if __name__=='__main__':
    fontsize=14
    legendsize=10
    cbarsize=14
    np.random.seed(10) 
    dynamics=lambda x:np.random.multivariate_normal([0,0], np.eye(2), 1)
    eta=lambda x:sin(x[0,0]+x[0,1])+cos(x[0,0]-x[0,1])
    eta1=lambda x:sin(x[0]+x[1])+cos(x[0]-x[1])
    A=lambda t: 0.5*(1-(-1)**t)*np.array([[0.5,0,0.5,0],[0,0.5,0,0.5],[0.5,0,0.5,0],[0,0.5,0,0.5]])+0.5*(1+(-1)**t)*np.array([[0.5,0.25,0.,0.25],[0.25,0.5,0.25,0.],[0.,0.25,0.5,0.25],[0.25,0.,0.25,0.5]])
    
    l=0.1
    sigma_f=1
    sigma=0.1
    kernel=lambda x,y:np.exp(-np.linalg.norm(x-y)**2/l)*sigma_f
    number=4
    T=100
    
    R=20
    MM=[0.1,0.2,0.25,0.5,1]
    for M in MM:
#    M=0.4
        np.random.seed(10) 
        print('T = ',T,' M = ',M,' R = ',R)
        Z_M=get_Z_M(np.linspace(0,10,M*R+1))
        Z_R=get_Z_R(np.linspace(0,10,R+1))
        robo_network=robot_network(number,dynamics,eta,sigma)
        robo_network.gpr_setup(kernel,mean_fun=None)
        for t in range(T):
            print('t=',t, 'sampling')
            robo_network.run_and_sample(eta)
            print('t=',t, 'gpr training')
    #        if t%10==0:
            robo_network.gpr_prep()
            print('t=',t, 'gpr local')
            robo_network.gpr_local(Z_M,Z_R)      
            print('t=',t, 'gpr Z_M')
            robo_network.distributed_gpr(Z_M,A(t))  
            print('t=',t,'fusing gpr')
        robo_network.fuse_gpr(Z_R)
        
    #    r1=robot(np.array([[5,5]]).T,dynamics,eta,sigma)
    #    for t in range(T*10):
    #        r1.run_and_sample(eta)
    #    r1.gp_prep()
    #    r1.gp_full(Z_R)
    #    plt.contourf(XX,YY,np.reshape(r1.mu_R,(n,n)))   
            
    #visualization of function to learn
        fig1=plt.figure()
        size=R-1
        X=np.linspace(0,10,size)
        Y=np.linspace(0,10,size)
        XX,YY=np.meshgrid(X,Y)
        Z=np.zeros((size,size))
        for i in range(size):
            for j in range(size):
                Z[i][j]=eta1([X[i],Y[j]])
        plt.contourf(XX.T,YY.T,Z,cmap='magma')
        cbar=plt.colorbar()
        cbar.ax.tick_params(labelsize=cbarsize) 
#        plt.title('ground truth of eta',fontsize=15)
    #print trajectory of robots
        for i,robo in enumerate(robo_network.network):
            plt.plot(robo.X[:,0],robo.X[:,1],label='robot'+str(i))
            plt.scatter(robo.X[0,0],robo.X[0,1])
            plt.legend(prop={'size':15})
            plt.xlabel('T = '+str(T),fontsize=18)
        fig1.savefig('trajectory_T'+str(T)+'M'+str(M*100)+'.png')
    #plot of error of full gp  
        for i in range(number):
            print_result(i,T,M)