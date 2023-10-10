import math
import numpy as np
import scipy.special

inf  = float("inf")

def soft_bellman_operation(env, reward):
    
    # Input:    env    :  environment object
    #           reward :  SxA dimensional vector 
    #           horizon:  finite horizon
        
    discount = env.discount
    horizon = env.horizon
    
    if horizon is None or math.isinf(horizon):
        raise ValueError("Only finite horizon environments are supported")
    
    n_states  = env.n_states
    n_actions = env.n_actions
    
    T = env.transition_matrix
    
    V = np.zeros(shape = (horizon, n_states))
    Q = np.zeros(shape = (horizon, n_states,n_actions))
        
    broad_R = reward

    # Base case: final timestep
    # final Q(s,a) is just reward
    Q[horizon - 1, :, :] = broad_R
    # V(s) is always normalising constant
    V[horizon - 1, :] = scipy.special.logsumexp(Q[horizon - 1, :, :], axis=1)

    # Recursive case
    for t in reversed(range(horizon - 1)):
#         next_values_s_a = T @ V[t + 1, :]
#         next_values_s_a = next_values_s_a.reshape(n_states,n_actions)
        for a in range(n_actions):
            Ta = env.mdp_transition_matrices[a]
            next_values_s_a = Ta@V[t + 1, :]
            Q[t, :, a] = broad_R[:,a] + discount * next_values_s_a
            
        V[t, :] = scipy.special.logsumexp(Q[t, :, :], axis=1)

    pi = np.exp(Q - V[:, :, None])

    return V, Q, pi

def discounted_reward_sum(reward,trajectory,env):
    gamma = env.discount
    discounted_reward = 0
    reward_arr = []
    for t, (s,a,sprime) in enumerate(trajectory):
        # discounted_reward += (gamma**t)*reward[s][a]
        reward_arr.append(reward[s][a])   

    num_stable_discounted_reward = np.polynomial.polynomial.polyval(gamma, np.array(reward_arr) )  
    
    return num_stable_discounted_reward
    

def confirm_boltzman_distribution(reward, traj, env,P):
    V_1, Q_1, pi = soft_bellman_operation(env,reward)
    
    # P = compute_trajectory_probability(traj, pi,env)
 
    P_ziebart = []
    for t in traj:
        rr = discounted_reward_sum(reward,t,env)

        P_ziebart.append(rr)
    
    Z = scipy.special.logsumexp(P_ziebart)
    P_z = np.exp(P_ziebart - Z)
#     print(P_z)
    return True if np.linalg.norm(P- P_z) <= 5e-2 else False