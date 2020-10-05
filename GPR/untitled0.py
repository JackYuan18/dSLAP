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

import numpy as np
import matplotlib.pyplot as plt
from gp2 import gp,k,k1
from math import sin,cos,sqrt
import matplotlib
from scipy import spatial
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 
class robot:
    """
    each robot records its own state (x), data collected (X,y), prediction on Z_M (Z_M)
    """
    def __init__(self,init,dynamics,eta,sigma,Z_M,Z_R):
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
        
        
        self.data=dict()
        self.data_keys=[]
        self.dataR=dict()
        self.dataR_keys=[]
        self.dataM=dict()
        self.dataM_keys=[]
        self.kdt=None
        
        for z in Z_R:
            self.dataR_keys.append(z)
            self.data_keys.append(z)
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
            
#        self.cov_full=np.diag(Sigma_full)
    def gp_M(self, Z_M):        
        self.mu_M,Sigma_M=self.gpr.predict_local(Z_M)
        self.cov_M=np.diag(Sigma_M)
        self.mu_R,Sigma_R=self.gpr.predict(Z_R,self.X)
        self.cov_R=np.reshape(np.diag(Sigma_R),(len(Z_R),1))
    def run(self):
        self.x=self.x+dynamics(self.x)*0.1
        self.X=np.vstack((self.X,self.x))
    def run_and_sample(self,eta):
        self.run()
        self.Y=np.vstack((self.Y,eta(self.x)))
        
class robot_network:
    def __init__(self,number,dynamics,eta,sigma,Z_M,Z_R):
        self.network=[robot(np.random.rand(1,2)*6+3,dynamics,eta,sigma,Z_M,Z_R) for i in range(number)]
        self.A=None
        self.size=number
        self.M=0
        self.Z_M=None
        self.dataM=None
        self.X=None
        self.Y=None
        self.data=dict()
        self.data_keys=[]
        self.sigma=sigma
        self.gpr=None
        for z in Z_R:
            self.data_keys.append(z)
    def distributed_gpr(self,Z_M,A):
        self.M=len(Z_M)
        self.Z_M=Z_M
        self.A=A
        for index,robot in enumerate(self.network):
            
            mu_M=np.array([[robot.data[str(z)][0] for z in Z_M]]).T
            cov_M=np.array([[robot.data[str(z)][1] for z in Z_M]]).T
            
            r_theta=mu_M/cov_M
            theta=sum(self.A[index,ind]*(robo.theta_-robot.theta_) for ind,robo in enumerate(self.network))
            robot.theta=robot.theta_+theta+r_theta-robot.r_theta
            robot.r_theta=r_theta
            
            r_xi=1/cov_M
            xi=sum(self.A[index,ind]*(robo.xi_-robot.xi_)for ind,robo in enumerate(self.network))
            robot.xi=robot.xi_+xi+r_xi-robot.r_xi
            robot.r_xi=r_xi
                      
            
#            k_MM=np.reshape(np.diag(k(robot.gpr.kernel,Z_M,Z_M)),(self.M,1))+robot.sigma**2
            cov_M_hat=robot.xi**(-1)
            mu_M_hat=cov_M_hat*robot.theta
            
            muM=iter(mu_M_hat)
            covM=iter(cov_M_hat)
            for x in self.Z_M:
                robot.dataM[str(x)]=(next(muM),next(covM))
#                robot.dataM_keys.add(x)
        for robot in self.network:
            robot.theta_=robot.theta
            robot.xi_=robot.xi
  
        
    def fuse_gpr(self):
        for robot in self.network:
            Z_M=[z for z in robot.dataM_keys if robot.dataM[str(z)][1]<robot.data[str(z)][1]]
            Z_R=robot.data_keys
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
#                robot.dataR_keys.append(zr)
#            for zm in Z_M:
#                robot.dataR[str(zm)]=robot.dataM[zm]
    def run_and_sample(self,eta):
        for robot in self.network:
            robot.run_and_sample(eta)
    def gpr_setup(self, kernel,mean_fun):
       
        for robot in self.network:
            robot.gp_setup(kernel,mean_fun)
    def gpr_prep(self):
        for robot in self.network:
            robot.gp_prep_local()
    def gpr_local(self,Z_R):
#        Z=np.concatenate((Z_M,Z_R))
        for robot in self.network:
            robot.gp_full(Z_R)
        
#    def gp_prep_local(self):
#        self.gpr.gp_prep_local(self.X, self.Y)
#    def gp_full(self,Z_,X,Y):
#        self.mu_full,self.cov_full=self.gpr.predict(Z_,X,Y)
#    def gp_full(self, Z):k_star=np.array([k1(self.kernel,test[i],z_train[i]) for i in range(len(test))]).T
#        mu,cov=self.gpr.predict_local(Z)
#        mu=np.reshape(mu,(mu.size,))
#        cov=np.reshape(cov,(cov.size,))
#        mu_iter=iter(mu)
#        cov_iter=iter(cov)
#        for z in Z:
#            self.data[str(z)]=(next(mu_iter),next(cov_iter))
            
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
            
    def print_network_results(self,sumError,plot):
        
        ZZ=[]
        
#        x=Z_R[i,0]
#        y=Z_R[i,1]
        N=len(self.data_keys)
        for x in self.data_keys:
            truth=eta1(x)
#            ZZ.append(self.data[str(x)][0])
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
                
def print_result(robo,T,M,sumError_local,sumError_fuse,plot):
    
    r1=robo_network.network[robo]
#    Z=np.vstack((Z_M,Z_R))
    
    ZZ=[]
    
#    Z=np.zeros((R-1,R-1))
    N=len(r1.dataR_keys)
    for x in r1.dataR_keys:    
        truth=eta1(x)
#        ZZ.append(r1.data[str(x)][0])
        ZZ.append(np.linalg.norm(r1.data[str(x)][0]-truth))
#    plt.plot(r1.X[:,0],r1.X[:,1],label='robot'+str(robo+1))
#    plt.scatter(r1.X[0,0],r1.X[0,1])
#    plt.legend(prop={'size': 15})
    if plot:
        fontsize=14
        legendsize=6
        cbarsize=14
        n=int(sqrt(len(r1.dataR_keys)))
        size=n
        X=np.linspace(0,10,size)
        Y=np.linspace(0,10,size)
        XX,YY=np.meshgrid(X,Y)
        fig2=plt.figure()
        plt.contourf(XX.T,YY.T,np.reshape(ZZ,(n,n)),cmap='magma',labelsize=fontsize)
    
        cbar=plt.colorbar()
        cbar.ax.tick_params(labelsize=cbarsize) 
    #    if M*100%10==0:
    #        M=M/10
        fig2.savefig('localerror'+str(T)+'M0_'+str(M)+'robot'+str(robo+1)+'.png')
    sumError_local.append(sum(ZZ)/N)
#plot of variance of local GPR
    
    ZZ=[]
    for x in r1.dataR_keys:
        ZZ.append(r1.data[str(x)][1])
    if plot:
        fig3=plt.figure()
        plt.contourf(XX.T,YY.T,np.reshape(ZZ,(n,n)),cmap='magma',labelsize=fontsize)
        cbar=plt.colorbar()
        cbar.ax.tick_params(labelsize=cbarsize) 
    #    plt.title('Variance of local GPR for robot'+str(robo),fontsize=16)
    #    plt.xlabel('consensus on '+str((M*20)**2*100/361)+'% of test data',fontsize=18)
    #    plt.plot(r1.X[:,0],r1.X[:,1],label='robot'+str(robo+1))
    #    plt.scatter(r1.X[0,0],r1.X[0,1])
    #    plt.legend(prop={'size': 15})
    #    if M*100%10==0:
    #        M=M/10
        fig3.savefig('localvar'+str(T)+'M0_'+str(M)+'robot'+str(robo+1)+'.png')
#plot of error of fusion  
    
    ZZ=[]
    for x in r1.dataR_keys:
#        x=Z_R[i,0]
#        y=Z_R[i,1]
        truth=eta1(x)
        ZZ.append(np.linalg.norm(r1.dataR[str(x)][0]-truth))
#    plt.plot(r1.X[:,0],r1.X[:,1],label='robot'+str(robo+1))
#    plt.scatter(r1.X[0,0],r1.X[0,1])
#    plt.legend(prop={'size': 15})
    sumError_fuse.append(sum(ZZ)/N)
    if plot:
        fig4=plt.figure()
        plt.contourf(XX.T,YY.T,np.reshape(ZZ,(n,n)),cmap='magma',labelsize=fontsize)
        cbar=plt.colorbar()
        cbar.ax.tick_params(labelsize=cbarsize) 
#    plt.title('Error of collaborative GPR for robot'+str(robo),fontsize=16)
        if M*100%10==0:
            M=M/10
    #    plt.xlabel('consensus on '+str((M*20)**2*100/361)+'% of test data',fontsize=18)
        fig4.savefig('fusionerror'+str(T)+'M0_'+str(M)+'robot'+str(robo+1)+'.png')
#plot of variance of fusion GPR
    
    ZZ=[]
    for x in r1.dataR_keys:
        ZZ.append(r1.dataR[str(x)][1])
    
    if plot:
        fig5=plt.figure()
        plt.contourf(XX.T,YY.T,np.reshape(ZZ,(n,n)),cmap='magma',labelsize=fontsize)
        cbar=plt.colorbar()
        cbar.ax.tick_params(labelsize=cbarsize) 
    #    plt.title('Variance of collaborative GPR for robot'+str(robo),fontsize=16)
    #    plt.xlabel('consensus on '+str((M*20)**2*100/361)+'% of test data',fontsize=18)
    #    plt.plot(r1.X[:,0],r1.X[:,1],label='robot'+str(robo+1))
    #    plt.scatter(r1.X[0,0],r1.X[0,1])
    #    plt.legend(prop={'size': 15})
    #    if M*100%10==0:
    #        M=M/10
        fig5.savefig('fusionvar'+str(T)+'M0_'+str(M)+'robot'+str(robo+1)+'.png')
    return sumError_local,sumError_fuse
  
if __name__=='__main__':
    fontsize=14
    legendsize=10
    cbarsize=14
    np.random.seed(10) 
    dynamics=lambda x:np.random.multivariate_normal([0,0], np.eye(2), 1)
    eta=lambda x:sin(x[0,0]+x[0,1])+cos(x[0,0]-x[0,1])
    eta1=lambda x:sin(x[0]+x[1])+cos(x[0]-x[1])
    A=lambda t: 0.5*(1-(-1)**t)*np.array([[0.5,0,0.5,0],[0,0.5,0,0.5],[0.5,0,0.5,0],[0,0.5,0,0.5]])+0.5*(1+(-1)**t)*np.array([[0.5,0.25,0.,0.25],[0.25,0.5,0.25,0.],[0.,0.25,0.5,0.25],[0.25,0.,0.25,0.5]])
    robo=0
    l=0.5
    sigma_f=1
    sigma=0.1
    kernel=lambda x,y:np.exp(-np.linalg.norm(x-y)**2/l)*sigma_f
    number=4
    T=100
    R=40
    MM=[0.5]
    sumError=[]
    sumError_local=[]
    sumError_fuse=[]
    plot=False
    
    for M in MM:
#    M=0.4
        np.random.seed(10) 
        print('T = ',T,' M = ',M,' R = ',R)
        Z_M=get_Z_M(np.linspace(0,10,int(M*R+1)))
        Z_R=get_Z_R(np.linspace(0,10,int(R+1)))
        robo_network=robot_network(number,dynamics,eta,sigma,Z_M,Z_R)
        robo_network.gpr_setup(kernel,mean_fun=None)
        ZZ=[]
    
#    Z=np.zeros((R-1,R-1))
        N=len(Z_R)
        for x in Z_R:    
            truth=eta1(x)
#            ZZ.append(r1.data[str(x)][0])
            ZZ.append(np.linalg.norm(truth))
        SE=sum(ZZ)/N
        sumError.append(SE)
        for i in range(number):
            sumError_local.append([SE])
            sumError_fuse.append([SE])
        for t in range(T):
            print('t=',t, 'sampling')
            robo_network.run_and_sample(eta)
            print('t=',t, 'gpr training')
    #        if t%10==0:
            robo_network.gpr_prep()
            print('t=',t, 'gpr local')
            robo_network.gpr_local(Z_R)      
            print('t=',t, 'gpr Z_M')
            robo_network.distributed_gpr(Z_M,A(t))  
            print('t=',t,'fusing gpr')
            robo_network.fuse_gpr()
            robo_network.full_gpr(Z_R,kernel,mean_fun=None)
            for robo in range(len(robo_network.network)):
                sumError_local[robo],sumError_fuse[robo]=print_result(robo,T,M,sumError_local[robo],sumError_fuse[robo],plot)
            sumError=robo_network.print_network_results(sumError,plot)
        
    #    r1=robot(np.array([[5,5]]).T,dynamics,eta,sigma)
    #    for t in range(T*10):
    #        r1.run_and_sample(eta)
    #    r1.gp_prep()
    #    r1.gp_full(Z_R)
    #    plt.contourf(XX,YY,np.reshape(r1.mu_R,(n,n)))   
            
    #visualization of function to learn
        lw=4
        if plot:
            fig0=plt.figure()
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
            fig0.savefig('ground truth of eta.png')
    #        plt.title('ground truth of eta',fontsize=15)
        #print trajectory of robots
            fig1=plt.figure(figsize=[4.8,4.0])
            for i,robo in enumerate(robo_network.network):
                if i ==0:
                    ln='o'
                elif i==1:
                    ln='s'
                elif i==2:
                    ln='*'
                else:
                    ln='X'
                plt.plot(robo.X[:,0],robo.X[:,1])
                plt.scatter(robo.X[0,0],robo.X[0,1],marker=ln,s=lw*30,label='robot '+str(i+1))
                plt.legend(prop={'size':15})
    #            plt.xlabel('T = '+str(T),fontsize=18)
            plt.xlim(0,10)
            plt.ylim(0,10)
            fig1.savefig('trajectory_T'+str(T)+'M'+str(M*100)+'.png')
    #plot of error of full gp  
#        for i in range(number):
        lw=4
        
        for i in range(4):
            figerror=plt.figure()
            plt.plot(sumError,'-',linewidth=lw,label='cNN-GPR')
#            if i==0:
            lb1='local GPR'
            lb2='RaDGPR'
#            else:
#                lb1=None
#                lb2=None
            plt.plot(sumError_local[i],'--', linewidth=lw,label=lb1)
            plt.plot(sumError_fuse[i],':', linewidth=lw,label=lb2)
#            plt.plot(0.8,0.8,color=[1-0.25*i, 0, 1-0.25*i],label='robot'+str(i+1))
#            plt.legend(prop={'size':15},loc=9, bbox_to_anchor=(0.8,1.2), ncol=3)
            plt.xlabel('robot'+str(i+1), fontsize=18)
#            plt.ylabel('Average predictive mean errors',fontsize=18)
            figerror.savefig('temporalerror'+str(i+1)+'.png')
