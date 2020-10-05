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
sys.setrecursionlimit(2500000)
from ACC2019_4robots_NNGPR import robot_network, robot,learning_MultiD_no_communication,learning_MultiD_no_communication_no_sample
import numpy as np
import matplotlib.pyplot as plt

import matplotlib


import datetime

import time
from sklearn.neighbors import KDTree

#from gp import gp,gp_prep
from state_space import ball
from car_dynamics import car,car_disturbance
#from safe_plot import progression_plot
#from safe_plot import progression
from state_space import State_Space,grid,d,gridsearch_construct,to_string
from car_dynamics import car_execution_disturbance
from safety_control import safety_iteration_GP, SI, SI_GP
from sklearn.gaussian_process.kernels import RBF

import pickle
from wind import wind
from decimal import *
getcontext().prec = 3
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 


def computation_time(network):
    for r in network:
        Total=np.array(r.oca_time)+np.array(r.ica_time)+np.array(r.Learning_time)+np.array(r.AL_time[:-1])
        mean_total=np.mean(Total)
        std_total=np.std(Total)
        
        mean_oca=sum(r.oca_time)/len(r.oca_time)
        std_oca=np.std(r.oca_time)
        ratio_oca=np.array(r.oca_time)/Total
        mean_ratio_oca=np.mean(ratio_oca)
        std_ratio_oca=np.std(ratio_oca)
        
        mean_ica=np.mean(r.ica_time)
        std_ica=np.std(r.ica_time)
        ratio_ica=np.array(r.ica_time)/Total
        mean_ratio_ica=np.mean(ratio_ica)
        std_ratio_ica=np.std(ratio_ica)
        

        
        mean_learning=np.mean(r.Learning_time)
        std_learning=np.std(r.Learning_time)
        ratio_learning=np.array(r.Learning_time)/Total
        mean_ratio_learning=np.mean(ratio_learning)
        std_ratio_learning=np.std(ratio_learning)
        
        
        mean_al=np.mean(r.AL_time[:-1])
        std_al=np.std(r.AL_time[:-1])
        ratio_al=np.array(r.AL_time[:-1])/Total
        mean_ratio_al=np.mean(ratio_al)
        std_ratio_al=np.std(ratio_al)
        
       
        
        
        print('Robot '+str(r.id)+'oca: ', mean_oca,std_oca,mean_ratio_oca,std_ratio_oca)
        print('Robot '+str(r.id)+'ica: ', mean_ica,std_ica,mean_ratio_ica,std_ratio_ica)
        print('Robot '+str(r.id)+'learning: ', mean_learning,std_learning,mean_ratio_learning,std_ratio_learning)
        print('Robot '+str(r.id)+'al: ', mean_al,std_al,mean_ratio_al,std_ratio_al)
        print('Robot '+str(r.id)+'total: ', mean_total,std_total)
        
        
                
def plot_safe(r,angle,k,obs):
    plt.figure()
#    angle = theta[0]
    state_temp=[x for x in r.state_temp if x.state[2]==angle]

    T = [float(x.unsafe) for x in state_temp]
    X1 = [float(x.state[0])*100 for x in state_temp]
    X2 = [float(x.state[1])*100 for x in state_temp]
#    center=initial
    
        
    left = obs.parameter[0]*100
    down = obs.parameter[2]*100
    width = obs.parameter[1]*100-obs.parameter[0]*100
    height = obs.parameter[3]*100-obs.parameter[2]*100  
    rectangle=plt.Rectangle((left,down),width,height,fc='red')
    plt.gca().add_patch(rectangle)
    plt.scatter(X1,X2, c=T)
    plt.plot(r6.X_[:,0]*100,r6.X_[:,1]*100,color='black',label='robot 6')
    plt.scatter(r6.X_[0,0]*100,r6.X_[0,1]*100,marker='>',color='black',s=60)
#    plt.title('safe states at the angle of '+str(angle)+' at iteration '+str(k))
    plt.legend(loc='upper left')

def plot_learning(r,angle):
    MU=[]
    VAR=[]
    state_temp=[x for x in r.state_temp if x.state[2]==angle]
    for x in state_temp:
        mu_E=0
        Var=0
        for u in x.mu:
              mu_e=np.linalg.norm(x.mu[u]-np.array([wind(x.state[0]*100,x.state[1]*100)*0.004,0,0]))
              var=x.cov[u]
              if mu_e>mu_E:
                  mu_E=mu_e
              if var>Var:
                  Var=var
        MU.append(mu_E)
        VAR.append(Var)
#    MU = [max(x.mu_) for x in r.state_temp]
#    VAR= [max(x.cov_) for x in r.state_temp]
    X1 = [float(x.state[0]) for x in state_temp]
    X2 = [float(x.state[1]) for x in state_temp]
    
    figure1=plt.figure()
    plt.scatter(X1,X2,c=MU,marker='s')
    plt.colorbar()
    figure2=plt.figure()
    plt.scatter(X1,X2,c=VAR,marker='s')
    plt.colorbar()

if __name__=='__main__':
    #learning setup  
    l=100
    sigma_f=0.000004
#    kernel=lambda x,y:np.exp(-np.linalg.norm(x-y)**2/l)*sigma_f
    kernel = sigma_f * RBF(0.1)#0.01
    sigma=0
    
#    Z_M=get_Z_M(np.linspace(0,10,int(M*R+1)), np.linspace(-1,1,int(M*R+1)))
#    Z_R=get_Z_R(np.linspace(0,10,int(R+1)))

    robot_size=0.015
    robot_M=0.025
    robot_l=0.025
    #network setup
    number=48

    
    A=lambda t: 0.5*(1-(-1)**t)*np.array([[0.5,0,0.5,0],[0,0.5,0,0.5],[0.5,0,0.5,0],[0,0.5,0,0.5]])+0.5*(1+(-1)**t)*np.array([[0.5,0.25,0.,0.25],[0.25,0.5,0.25,0.],[0.,0.25,0.5,0.25],[0.25,0.,0.25,0.5]])
    
    robo_network=robot_network(A)
    
    
    r1=robot(np.array([[0.20,0.80,0.5]]),car_disturbance, car, robot_M, robot_l,sigma,1,robot_size)
    r2=robot(np.array([[0.80,0.80,0.5]]),car_disturbance, car,  robot_M, robot_l,sigma,2,robot_size)
    r3=robot(np.array([[0.80,0.50,-0.5]]),car_disturbance, car,  robot_M, robot_l,sigma,3,robot_size)
    r4=robot(np.array([[0.80,0.2,-0.5]]),car_disturbance, car,  robot_M, robot_l,sigma,4,robot_size)
    r5=robot(np.array([[0.20,0.20,-0.5]]),car_disturbance, car,  robot_M, robot_l,sigma,5,robot_size)
    r6=robot(np.array([[0.20,0.50,0.5]]),car_disturbance, car,  robot_M, robot_l,sigma,6,robot_size)
    r7=robot(np.array([[0.5,0.85,-0.5]]),car_disturbance, car,  robot_M, robot_l,sigma,7,robot_size)
    r8=robot(np.array([[0.5,0.15,0.5]]),car_disturbance, car,  robot_M, robot_l,sigma,8,robot_size)
    
    
    r1.goal=np.array([[0.80,0.20,1]])
    r2.goal=np.array([[0.20,0.20,1]])
    r3.goal=np.array([[0.20,0.50,0]])
    r4.goal=np.array([[0.20,0.80,0]])
    r5.goal=np.array([[0.80,0.80,0]])
    r6.goal=np.array([[0.80,0.50,1]])
    r7.goal=np.array([[0.5,0.15,0]])
    r8.goal=np.array([[0.5,0.85,0]])
    
    robo_network.add_robot(r1)
    robo_network.add_robot(r2)
    robo_network.add_robot(r3)
    robo_network.add_robot(r4)
    robo_network.add_robot(r5)
    robo_network.add_robot(r6)
    robo_network.add_robot(r7)
    robo_network.add_robot(r8)
#       
#    R8=[]


    P=5
    p=4
    save=True
    num=5
    size=50
    results=[]
    print(datetime.datetime.now())
#    print('save is', save)
    

    Up = np.linspace(-0.3,0.3,5)
    obs_1 = State_Space(3,[0.475,0.525,0.45,0.55,-1,1])
    obs_2 = State_Space(3,[0.175,0.225,0.125,0.175,-1,1])
    obs_3 = State_Space(3,[0.825,0.875,0.45,0.55,-1,1])
#    Xfree = grid([[-5,5],[-5,5],[-1,1]])  
    obs=obs_1
    
    Total=[]
    
#    ball_obs=[]
#    for ob in obs:
    ball_obs=ball(obs,0)
    
    left = obs_1.parameter[0]
    down = obs_1.parameter[2]
    width = obs_1.parameter[1]-obs_1.parameter[0]
    height = obs_1.parameter[3]-obs_1.parameter[2]  
    #                if final and s.to_goal<self.M*self.eps_p5+self.hp:
    
    
    
       
    theta = np.linspace(-1,1,2**(P)+1)
    angle = theta[0]
    initial=np.array([4,2.0, angle])        

    Y, X = np.meshgrid(np.linspace(-5, 5, size), np.linspace(-5, 5, size)) 
    _,Z = np.meshgrid(np.linspace(-5, 5, size), np.linspace(-5, 5, size)) 
       


    print('computing save initial controller...')
#    start=time.time()
    
#    robo_network.control_init(car,car_disturbance, [[0,1],[0,1],[-1,1]],robot_M,robot_l,Up,obs,P,kernel)
#    end=time.time()
#    print('time spent:', end-start)
    
 
#    plot_control=False
#    np.random.seed(10) 
#    final=False
    for psi in np.linspace(0,10,30):
        robo_network=robot_network(A)
        r1=robot(np.array([[0.20,0.80,0.5]]),car_disturbance, car, robot_M, robot_l,sigma,1,robot_size)
        r2=robot(np.array([[0.80,0.80,0.5]]),car_disturbance, car,  robot_M, robot_l,sigma,2,robot_size)
        r3=robot(np.array([[0.80,0.50,-0.5]]),car_disturbance, car,  robot_M, robot_l,sigma,3,robot_size)
        r4=robot(np.array([[0.80,0.2,-0.5]]),car_disturbance, car,  robot_M, robot_l,sigma,4,robot_size)
        r5=robot(np.array([[0.20,0.20,-0.5]]),car_disturbance, car,  robot_M, robot_l,sigma,5,robot_size)
        r6=robot(np.array([[0.20,0.50,0.5]]),car_disturbance, car,  robot_M, robot_l,sigma,6,robot_size)
        r7=robot(np.array([[0.5,0.85,-0.5]]),car_disturbance, car,  robot_M, robot_l,sigma,7,robot_size)
        r8=robot(np.array([[0.5,0.15,0.5]]),car_disturbance, car,  robot_M, robot_l,sigma,8,robot_size)
        
        
        r1.goal=np.array([[0.80,0.20,1]])
        r2.goal=np.array([[0.20,0.20,1]])
        r3.goal=np.array([[0.20,0.50,0]])
        r4.goal=np.array([[0.20,0.80,0]])
        r5.goal=np.array([[0.80,0.80,0]])
        r6.goal=np.array([[0.80,0.50,1]])
        r7.goal=np.array([[0.5,0.15,0]])
        r8.goal=np.array([[0.5,0.85,0]])
        
        robo_network.add_robot(r1)
        robo_network.add_robot(r2)
        robo_network.add_robot(r3)
        robo_network.add_robot(r4)
        robo_network.add_robot(r5)
        robo_network.add_robot(r6)
        robo_network.add_robot(r7)
        robo_network.add_robot(r8)
        robo_network.control_init(car,car_disturbance, [[0,1],[0,1],[-1,1]],robot_M,robot_l,Up,obs,P,kernel)
#        end=time.time()
    #    print('time spent:', end-start)
        
        
        plot_control=False
        np.random.seed(10) 
        final=False
        for t in range(15):
            print('psi=',psi,'t=',t, 'p=', p)
#            print('update controller...')
                    
                    
            learning_MultiD_no_communication(robo_network,  kernel,t,ball_obs,T=80,psi=psi) #only run local GPR and distributed GPR
    #        print('safe planning...')
    #        start=time.time()
            if t>5:
                final=True
    #            p=P-1
    #            if t>10:
                p=P
            robo_network.safe_planner(obs,p,GP=True,final=final)
            
            end=time.time()
        total=0
        for r in robo_network.network:
            total_=len(r.X_)
            if total_>total:
                total=total_
        Total.append(total)
        
#        print('time spent:', end-start)
#        robo_network.count_inclusion()


#        p=min(p+1,P) self.
#        robo_network.distance_to_goal()
#        plot_safe(r6,angle,t,obs)

#    
    plt.plot(np.linspace(0,10,30),np.array(Total)*0.2)
    plt.xlabel('\psi',fontsize=20)
    plt.ylabel('Arrival time',fontsize=20)
    if plot_control:
            figure1=plt.figure(1)
            angle = theta[0]
            state_temp=[x for x in r1.state_temp if x.state[2]==angle]

            T = [float(x.unsafe) for x in state_temp]
            X1 = [float(x.state[0])*100 for x in state_temp]
            X2 = [float(x.state[1])*100 for x in state_temp]
            center=initial
            plt.scatter(X1,X2, c=T)
            left = obs.parameter[0]*100
            down = obs.parameter[2]*100
            width = (obs.parameter[1]-obs.parameter[0])*100
            height = (obs.parameter[3]-obs.parameter[2])*100  
            rectangle=plt.Rectangle((left*100,down*100),width*100,height*100,fc='red',label='obstacle')
            plt.gca().add_patch(rectangle)
            plt.title('safe states at an angle of '+str(angle))
            plt.legend(loc='upper left')
            
            
#            plt.plot(r1.X_[:,0],r1.X_[:,1],color='g',label='robot 2')
#            plt.s>10:
#            final=True
#            p=Pcatter(r1.X_[0,0],r1.X_[0,1],marker='*',color='g',s=60)
            
            figure3=plt.figure(3)
            
            
            
            plt.plot(r1.X_[:,0]*100,r1.X_[:,1]*100,color='g',label='robot 1')
            plt.scatter(r1.X_[0,0]*100,r1.X_[0,1]*100,marker='*',color='g',s=60)
            plt.plot(r2.X_[:,0]*100,r2.X_[:,1]*100, color='blue',label='robot 2')
            plt.scatter(r2.X_[0,0]*100,r2.X_[0,1]*100,marker='o',color='blue',s=60)
            plt.plot(r3.X_[:,0]*100,r3.X_[:,1]*100,color='orange',label='robot 3')
            plt.scatter(r3.X_[0,0]*100,r3.X_[0,1]*100,marker='x',color='orange',s=60)
            plt.plot(r4.X_[:,0]*100,r4.X_[:,1]*100,color='cyan',label='robot 4')
            plt.scatter(r4.X_[0,0]*100,r4.X_[0,1]*100,marker='v',color='cyan',s=60)
            plt.plot(r5.X_[:,0]*100,r5.X_[:,1]*100,color='purple',label='robot 5')
            plt.scatter(r5.X_[0,0]*100,r5.X_[0,1]*100,marker='H',color='purple',s=60)
            plt.plot(r6.X_[:,0]*100,r6.X_[:,1]*100,color='black',label='robot 6')
            plt.scatter(r6.X_[0,0]*100,r6.X_[0,1]*100,marker='>',color='black',s=60)
            plt.plot(r7.X_[:,0]*100,r7.X_[:,1]*100,color='gold',label='robot 7')
            plt.scatter(r7.X_[0,0]*100,r7.X_[0,1]*100,marker='<',color='gold',s=60)
            plt.plot(r8.X_[:,0]*100,r8.X_[:,1]*100,color='magenta',label='robot 8')
            plt.scatter(r8.X_[0,0]*100,r8.X_[0,1]*100,marker='>',color='magenta',s=60)
            
#            plt.colorbar() 
            
            
            circle1 = plt.Circle((r1.goal[0,0]*100, r1.goal[0,1]*100), 0.06*100, color='g',label='goal for robot 1')
            circle2 = plt.Circle((r2.goal[0,0]*100, r2.goal[0,1]*100), 0.06*100, color='blue',label='goal for robot 2')
            circle3 = plt.Circle((r3.goal[0,0]*100, r3.goal[0,1]*100), 0.06*100, color='orange',label='goal for robot 3')
            circle4 = plt.Circle((r4.goal[0,0]*100, r4.goal[0,1]*100), 0.06*100, color='cyan',label='goal for robot 4')
            circle5 = plt.Circle((r5.goal[0,0]*100, r5.goal[0,1]*100), 0.06*100, color='purple',label='goal for robot 5')
            circle6 = plt.Circle((r6.goal[0,0]*100, r6.goal[0,1]*100), 0.06*100, color='black',label='goal for robot 6')
            circle7 = plt.Circle((r7.goal[0,0]*100, r7.goal[0,1]*100), 0.06*100, color='gold',label='goal for robot 7')
            circle8 = plt.Circle((r8.goal[0,0]*100, r8.goal[0,1]*100), 0.06*100, color='magenta',label='goal for robot 8')
            rectangle1=plt.Rectangle((left,down),width,height,fc='red',label='obstacle')
            
            
            plt.gca().add_patch(circle2)
            plt.gca().add_patch(circle1)
            plt.gca().add_patch(circle3)
            plt.gca().add_patch(circle4)
            plt.gca().add_patch(circle5)
            plt.gca().add_patch(circle6)
            plt.gca().add_patch(circle7)
            plt.gca().add_patch(circle8)
            plt.gca().add_patch(rectangle1)
            plt.title('robot trajectories')
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), shadow=True, ncol=3)
            
            figure2=plt.figure(2)
#            for robo in robo_network.network:
            min_dist=100
            length=0
            
            for r in robo_network.network:
                
                if min(r.dist_to_collision)*100<min_dist:
                    
                    min_dist=min(r.dist_to_collision)*100
#                    print(r.id,min_dist)
                if len(r.X)>length:
                    length=len(r.X)
            xx=np.linspace(0,int(length*0.2),length)
            plt.plot(xx[:len(r1.dist_to_collision)],np.array(r1.dist_to_collision)*100,label='robot '+str(r1.id),color='g')
            plt.plot(xx[:len(r2.dist_to_collision)],np.array(r2.dist_to_collision)*100,label='robot '+str(r2.id),color='blue')
            
            plt.plot(xx[:len(r3.dist_to_collision)],np.array(r3.dist_to_collision)*100,label='robot '+str(r3.id),color='orange')
            plt.plot(xx[:len(r4.dist_to_collision)],np.array(r4.dist_to_collision)*100,label='robot '+str(r4.id),color='cyan')
            plt.plot(xx[:len(r5.dist_to_collision)],np.array(r5.dist_to_collision)*100,label='robot '+str(r5.id),color='purple')
            plt.plot(xx[:len(r6.dist_to_collision)],np.array(r6.dist_to_collision)*100,label='robot '+str(r6.id),color='black')
            plt.plot(xx[:len(r7.dist_to_collision)],np.array(r7.dist_to_collision)*100,label='robot '+str(r7.id),color='gold')
            plt.plot(xx[:len(r8.dist_to_collision)],np.array(r8.dist_to_collision)*100,label='robot '+str(r8.id),color='magenta')
            
                    
#            min_dist=min((min(r1.dist_to_collision)),min(r2.dist_to_collision),min(r3.dist_to_collision),min(r4.dist_to_collision))
            min_dist_=np.ones((length,))*min_dist
            print(length*0.2)
            
            plt.plot(xx,min_dist_,'--',color='red')
#            plt.xticks(range(0, (length+1)*0.2))
            plt.xlabel('time (seconds)')
            plt.legend(loc='upper right')
            plt.title('minimum distance: '+str(round(min_dist,3)))


            figure3=plt.figure(4)
            plt.plot(r1.R, color='g', label='robot '+str(r1.id))
            plt.plot(r2.R, color='blue', label='robot '+str(r2.id))
            plt.plot(r3.R, color='orange', label='robot '+str(r3.id))
            plt.plot(r4.R, color='cyan', label='robot '+str(r4.id))
            plt.plot(r5.R, color='purple', label='robot '+str(r5.id))
            plt.plot(r6.R, color='black', label='robot '+str(r6.id))
            plt.plot(r7.R, color='gold', label='robot '+str(r7.id))
            plt.plot(r8.R, color='magenta', label='robot '+str(r8.id))
            plt.xlabel('Iteration',fontsize=20)
            plt.ylabel('Ratio',fontsize=20)
            plt.legend()
            
            figure4=plt.figure(5)
            plt.plot(r1.ica_time,color='g', label='robot '+str(r1.id))
            plt.plot(r2.ica_time,color='blue',label='robot '+str(r2.id))
            plt.plot(r3.ica_time,color='orange',label='robot '+str(r3.id))
            plt.plot(r4.ica_time,color='cyan',label='robot '+str(r4.id))
            plt.plot(r5.ica_time,color='purple',label='robot '+str(r5.id))
            plt.plot(r6.ica_time,color='black',label='robot '+str(r6.id))
            plt.plot(r7.ica_time,color='gold',label='robot '+str(r7.id))
            plt.plot(r8.ica_time,color='magenta',label='robot '+str(r8.id))
            plt.xlabel('Iteration',fontsize=20)
            plt.ylabel('Time (second)',fontsize=20)
            plt.legend()
        
    
      
        
        
     