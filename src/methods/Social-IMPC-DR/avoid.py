import cvxpy as cp
import numpy as np
from scipy import linalg as lg
from uav import *



def GET_cons(agent,obstacle_list, wall_collision_multiplier=2.0):
    
    P=agent.pre_traj 
    target=agent.target 
    K=agent.K 
    D=agent.D
    eta=agent.eta
    r_min=agent.r_min
    index=agent.index
    term_overlap=agent.term_overlap

    cons_A=np.zeros((1,D*K))
    cons_B=np.array([-1.0])
    cons_C=np.zeros((1,1))
    RHO=np.array([0.0])

    for j in range(0,len(obstacle_list)): # j is the index of other agent

        if(j==index): 
            continue
        
        P_j=obstacle_list[j]

        for t in range(1,len(P)):
            p=P[t]
            l=len(P_j)
            if(t>=l):
                p_j=P_j[-1]
            else:
                p_j=P_j[t]

            # Determine if this is a collision with a stationary robot (wall)
            # Stationary robots have zero velocity and their position doesn't change
            is_stationary_robot = False
            if len(P_j) > 1:
                # Check if the robot's position is constant (indicating it's stationary)
                position_variance = np.var(P_j, axis=0)
                is_stationary_robot = np.all(position_variance < 1e-6)  # Very small variance indicates stationary
            
            # Use larger collision distance for stationary robots (walls)
            if is_stationary_robot:
                effective_r_min = r_min * wall_collision_multiplier  # Use configurable multiplier for walls
            else:
                effective_r_min = r_min

            if(t==len(P)-1):
                a,b,rho = MBVC_WB(p,p_j,target,effective_r_min,eta,term_overlap)
            else:
                a,b = MBVC(p,p_j,effective_r_min)
            
            # add constraints 
            cons_a=[]
            for i in range(0,len(P)-1):
                if(i==t-1):
                    cons_a=np.append(cons_a,a) 
                else:
                    cons_a=np.append(cons_a,np.zeros(D))
            
            cons_A=np.row_stack((cons_A,cons_a))
            cons_B=np.append(cons_B,b)
            cons_C=np.row_stack(( cons_C , np.zeros(cons_C.shape[1]) ))

            if(t==len(P)-1):
                cons_c=np.zeros((len(cons_B),1))
                cons_c[-1]=1.0
                cons_C=np.column_stack((cons_C,cons_c))
            
                RHO=np.append(RHO,rho)

     
    return [cons_A,cons_B,cons_C,RHO]



def MBVC(agent_i,agent_j,r_min):

    p=(agent_i+agent_j)/2
    a=(agent_i-agent_j)/np.linalg.norm(agent_i-agent_j)

    b=a @ p + r_min/2

    return a,b



def MBVC_WB(agent_i,agent_j,target,r_min,eta,term_overlap):

    p=(agent_i+agent_j)/2
    a=(agent_i-agent_j)/np.linalg.norm(agent_i-agent_j)

    b=a @ p + r_min/2
    n_j=np.zeros(2)
    n_target=np.zeros(2)

    for i in range(2):
        n_j[i]=(agent_j-agent_i)[i]
        n_target[i]=(target-agent_i)[i]

    if np.linalg.norm(n_j) > 1e-10 and np.linalg.norm(n_target) > 1e-10:

        n_j=n_j/np.linalg.norm(n_j)

        
        n_target=n_target/np.linalg.norm(n_target)

        
        if(term_overlap):
            rho_i_j=0.0
            #rho_i_j=0.2*2.71828**(eta*(n_j[1]*n_target[0]-n_j[0]*n_target[1]))
        else:
            rho_i_j=0.0
            
    else:
        rho_i_j=0.0

    return a,b,rho_i_j