# -*- coding: utf-8 -*-
"""
Spyder Editor
This is a temporary script file.
"""
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
#mu = 0
#x = np.linspace(-5,5,50)
#
#C = np.zeros((len(x),len(x)))
#mu = mu*np.ones(len(x))
#for ind,i in enumerate(x):
#    for jnd,j in enumerate(x):
#        C[ind,jnd] = np.exp(-0.5*(i-j)**2)
#
#f = np.random.multivariate_normal(mu,C)    
#
#plt.plot(x,f)
#plt.plot(x,[2]*len(x))
#plt.plot(x,[-2]*len(x))
#plt.ylim(-3,3)


sigma_f = 1
l = 0.1
#def kernel(x,y):
#    return np.exp(-l*(x-y)**2)*sigma_f

#def kernel(x,y):
#    return np.exp(-l*np.linalg.norm(x-y))*sigma_f






#def k(x,y):
#        C = np.zeros((len(x),len(y)))
#        for ind,i in enumerate(x):
#            for jnd,j in enumerate(y):
#                C[ind,jnd] = kernel(i,j)   
#        return C



#def gp(kernel,test,train, train_f):
#    
#    def k(x,y):
#        C = np.zeros((len(x),len(y)))
#        for ind,i in enumerate(x):
#            for jnd,j in enumerate(y):
#                C[ind,jnd] = kernel(i,j)   
#        return C
#
#    mu = np.matmul(np.matmul(k(test,train),np.linalg.inv(k(train,train))),train_f)
#    cov = k(test,test)-np.matmul(np.matmul(k(test,train),np.linalg.inv(k(train,train)+np.eye(num)*sigma**2)),k(train,test))
#    return mu,cov
def k(x,y):
        C = np.zeros((len(x),len(y)))
        for ind,i in enumerate(x):
            for jnd,j in enumerate(y):
                C[ind,jnd] = kernel(i,j)   
        return C

def gp_prep(kernel,train, train_f,sigma,mean_fun=None):
    
    
    K_inv=np.linalg.inv(k(train,train)+np.eye(len(train))*sigma**2)
    if mean_fun is not None:
        mu_prep = np.matmul(K_inv,train_f-mean_fun(train))
    else:
#        print(K_inv)
        mu_prep = np.matmul(K_inv,train_f)
    return mu_prep,K_inv

def gp(mu_prep,cov_prep,test,train,kernel,mean_fun=None):
    
   
    k_star=k(test,train)
    if mean_fun is not None:
        mu = np.matmul(k_star,mu_prep)+mean_fun(test)
    else:
        mu = np.matmul(k_star,mu_prep)
    cov = k(test,test)-np.matmul(np.matmul(k_star,cov_prep),k_star.T)
    return mu,cov


def gp_(mu_prep,cov_prep,test,train,kernel):
    def k(x,y):
        C = np.zeros((len(x),len(y)))
        for ind,i in enumerate(x):
            for jnd,j in enumerate(y):
                C[ind,jnd] = kernel(i,j)   
        return C
   
    k_star=k(test,train)
    mu = np.matmul(k_star,mu_prep)
    cov = k(test,test)-np.matmul(np.matmul(k_star,cov_prep),k_star.T)
    return mu,cov

def square3(x):
    return x[0]**2+x[1]**2
if __name__=='__main__':
    
    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d')
    num = 5
    train1  = np.linspace(0,2,10)
    train2 = np.linspace(3,5,10)
    
    #train[1] = train[1]-1
    #train  = np. random.uniform(-5,5,5)
    test_old = np.linspace(0,5,61)
    truth=np.sin(test_old)
    M=np.linspace(0,5,16)
    R=np.array(list(set(test_old).difference(set(M))))
    test=np.concatenate((M,R))
    vsquare=np.vectorize(square3)
    train_f1 = np.sin(train1)
    train_f2=np.sin(train2)
    #train_f = np.random.uniform(-2,2,num)

    sigma = 0.01
    sigma_f = 1
    l = 0.3
    kernel=lambda x,y:np.exp(-l*np.linalg.norm(x-y))*sigma_f
    mu_prep1,cov_prep1=gp_prep(kernel,train1, train_f1,sigma)
    mu1,cov1=gp(mu_prep1,cov_prep1,test,train1,kernel)
    
    mu_prep2,cov_prep2=gp_prep(kernel,train2, train_f2,sigma)
    mu2,cov2=gp(mu_prep2,cov_prep2,test,train2,kernel)
#    mu,cov = gp(kernel,test,train,train_f)

#    K_inv = np.linalg.inv(k(train,train)+np.eye(num)*sigma**2)
#    K = k(train,train)+np.eye(num)*sigma**2
#    d = np.zeros(len(test))
#    for ind,t in enumerate(test):
#        distances = [abs(t-x) for x in train]
#        d[ind]=max(distances)
    
#    var_est = 2*(sigma_f**2 - sigma_f**2*np.exp(-l*(d))**2)
    
    #for i in range(3):
    #    f = np.random.multivariate_normal(mu,cov)    
    #    plt.plot(test,f)
#    plt.ylim(-10,20)
#    ax.scatter(train1[:,0],train1[:,1],train_f)
    
    plt.scatter(test,mu1)
    plt.plot(test_old,truth,'red')
    
    std2 = np.sqrt(np.diag(cov1))*2
    
    plt.scatter(test,mu1+std2,c='g')
    plt.scatter(test,mu1-std2,c='g')
    plt.plot(test_old,truth,'red')
    plt.figure()
    plt.scatter(test,mu2)
    
    std2 = np.sqrt(np.diag(cov2))*2
    
    plt.scatter(test,mu2+std2,c='g')
    plt.scatter(test,mu2-std2,c='g')
    plt.plot(test_old,truth,'red')
    plt.figure()
    train=np.concatenate((train1,train2))
    train_f=np.sin(train)
    mu_prep,cov_prep=gp_prep(kernel,train, train_f,sigma)
    mu,cov=gp(mu_prep,cov_prep,test,train,kernel)
    plt.scatter(test,mu)
    
    std2 = np.sqrt(np.diag(cov))*2
    
    plt.scatter(test,mu+std2,c='g')
    plt.scatter(test,mu-std2,c='g')
    plt.plot(test_old,truth,'red')
    
    
    
    plt.figure()
    SigmaMM=cov1[:16,:16]
    SigmaMR=cov1[16:,:16]
    SigmaRM=cov1[:16,16:]
    SigmaRR=cov1[16:,16:]
    
    var1=np.diag(SigmaMM)
    var2=np.diag(cov2[:16,:16])
    
    sigma_bcm=1/var1+1/var2-np.linalg.inv(k(M,M))
    var_bcm=(np.diag(sigma_bcm))**(-1)
    mu_bcm=var_bcm*(mu1[:16]/var1+mu2[:16]/var2)
    
    mu_prep=np.matmul(np.linalg.inv(SigmaMM+var_bcm*np.eye(16)),mu_bcm-mu1[:16])
    mu_R = np.matmul(SigmaMR,mu_prep)+mu1[16:]
    cov_R = SigmaRR-np.matmul(np.matmul(SigmaMR,np.linalg.inv(SigmaMM+var_bcm*np.eye(16))),SigmaRM)
    
    std2R = np.sqrt(np.diag(cov_R))*2
    std2M = np.sqrt(var_bcm)*2
    std2=np.concatenate((std2M,std2R))
    mu_=np.concatenate((mu_bcm,mu_R))
    
    
    plt.scatter(test,mu_)
    plt.plot(test_old,truth,'red')

    
    plt.scatter(test,mu_+std2,c='g')
    plt.scatter(test,mu_-std2,c='g')
    
#    mu_bcm=
#    plt.plot(test,np.sin(test)*2+np.cos(test)*2+test,'--r')
#    
#    plt.plot(test, mu+var_est,'--g')
#    plt.plot(test, mu-var_est,'--g')

#print(max(std2))