import numpy as np
import math
import scipy.special
from utils import confirm_boltzman_distribution, soft_bellman_operation, discounted_reward_sum

inf  = float("inf")


class Env(object):
    def __init__(self, n_states, n_actions, horizon, discount = 0.99, unique_start = True, deterministic = False):
        self.n_states = n_states
        self.n_actions= n_actions
        self.horizon = horizon
        self.discount = discount
        self.unique_start = unique_start
        self.deterministic = deterministic
        
        if self.unique_start:
            self.start_state = np.random.choice(np.arange(self.n_states))
            self.start_distribution = np.zeros(self.n_states)
            self.start_distribution[self.start_state] = 1
            
        else:
            self.start_distribution = np.random.rand(num_states)
            self.start_distribution /= np.sum(self.start_distribution)
#         self.initial_state_distribution = (1/self.n_states)*np.ones(self.n_states)
        self.mdp_transition_matrices = []
        
        self.get_random_transition_matrices()
        
    def get_random_transition_matrices(self):
        k = self.n_states
        T = np.full(shape = (self.n_states*self.n_actions, self.n_states) , fill_value=0)
        T = np.zeros((self.n_states*self.n_actions, self.n_states))
        
        for i in range(self.n_actions):
            
            Pa = np.identity(k) + \
                    np.random.uniform(low=0., high=.25, size=(k, k))
            Pa /= Pa.sum(axis=1, keepdims=1)
            
            if self.deterministic:
                Pa = np.random.permutation(np.eye(self.n_states))
                
            self.mdp_transition_matrices.append(Pa)
            T[i*k:(i+1)*k,:] = Pa
        
        assert (np.linalg.norm(T.sum(axis = 1)- np.ones(T.shape[0]))) <= 1e-5
        
        self.transition_matrix = T
    
    def generate_trajectories(self):
        # Memoization dictionary to store already computed results
        horizon = self.horizon
        memo_dict = {}

        def generate_recursive(current_time, current_state):
            # Check if the result is already computed
            if (current_time, current_state) in memo_dict:
                return memo_dict[(current_time, current_state)]

            # Base case: when we reach the horizon, return an empty trajectory
            if current_time == horizon:
                return [[]]

            trajectories = [] 

            # Recursive case: try all actions and next states
            for action in range(self.n_actions):
                next_state_probs = self.mdp_transition_matrices[action][current_state] if current_state is not None else self.start_distribution
                
                for next_state in range(self.n_states):
                    # Recursive call
                    if next_state_probs[next_state] != 0:
                        next_trajectories = generate_recursive(current_time + 1, next_state)

                        # Extend the current trajectory with the current action and next state
                        if current_state == None:
                            
                            trajectories.extend([(next_state, action )] + traj for traj in next_trajectories)
                        else:
                            trajectories.extend([(current_state, action, next_state)] + traj for traj in next_trajectories)

            # Memoize the result before returning
            memo_dict[(current_time, current_state)] = trajectories
            return trajectories

        # Start the recursion with time = 0 and no initial state
        # For a unique starting state
        return generate_recursive(0, self.start_state)

def compute_single_trajectory_probability(traj, policy, env):
    # traj  : single trajectory 
    # policy: is of shape (horizon, n_states, n_actions)
    # env   : environment class
        
    p = 1.0
    
    for t, (s,a,s_prime) in enumerate(traj):
        
        if t == 0:
            p *= env.start_distribution[s]
        
        # policy probability
        p_policy = policy[t][s][a]
        # tranisition porbability
        p_trans = env.mdp_transition_matrices[a][s][s_prime]
        
        p*= p_policy*p_trans 
    
    return p


def compute_all_trajectory_probability(traj, pi,env):
    P = []
    for t in traj:
        p = compute_single_trajectory_probability(t,pi, env)
        P.append(p)
    # assert that we indeed have a distribution
    assert np.abs(1 - sum(P)) < 1e-4
    return P



def check_reward_equivalence(r_1,r_2,T):
    # check weak identifiability equivalence
    diff = []
    for trajectory in T:
        discounted_r_1 = discounted_reward_sum(r_1,trajectory, env)
        discounted_r_2 = discounted_reward_sum(r_2,trajectory, env)
        c = discounted_r_1 - discounted_r_2
        diff.append(c)
        
    diff = np.array(diff)
    # need to round for floating-point precision
    
#     print("Unique in d: ",np.unique(diff.round(decimals=4)))

    if np.unique(diff.round(decimals=4)).shape[0] == 1:
#         print("Rewards are equivalent")
        return True
    else:
#         print("Rewards are not equivalent")
        return False


def check_distribution_equivalence(P1,P2):
    # print("norm is:",np.linalg.norm(np.array(P1)-np.array(P2)) )
    return True if np.linalg.norm(np.array(P1)-np.array(P2)) <= 1e-3 else False


def check_weak_identifiability(reward_set, trajectories, env):
    pi = []

    # for each reward, find the resulting policy
    for reward in reward_set:
        _,_, pi_r = soft_bellman_operation(env, reward)
        pi.append(pi_r)

    # for each policy,  find the resulting distribution over trajectories
    dist = []
    for i,pi_r in enumerate(pi):
        P_r = compute_all_trajectory_probability(trajectories, pi_r,env)
        dist.append(P_r)

        if env.deterministic:
            assert confirm_boltzman_distribution(reward_set[i], trajectories, env, P_r) == True

    non_unique_dist_pairs = []

    for i,p_1 in enumerate(dist):
        for j,p_2 in enumerate(dist):

            if i < j:  
                if check_distribution_equivalence(p_1,p_2):
                    non_unique_dist_pairs.append([i,j])

    

    if len(non_unique_dist_pairs) == 0: 
        
        # this means that all pairs are unique and we are on the left branch of the tree
        equivalent_rewards = []

        for i,r_1 in enumerate(reward_set):
            for j,r_2 in enumerate(reward_set, start = i):
                if i != j:
                    if check_reward_equivalence(r_1,r_2,trajectories):
                        equivalent_rewards.append([i,j])
        
        if len(equivalent_rewards) == 0:
            print("Weakly Identifiable -- Left Branch Of the Tree")
            return

        else:
            print("Not Weakly Identifiable -- Left Branch Of the Tree")
            return

    # Now we are at the right side of the tree, which means there are equivalent distributions
    # print("Non-uniqur pairs: ", non_unique_dist_pairs)
    r_e = 0
    for [i,j] in non_unique_dist_pairs:
        # check if there rewards are equivalent  
        r_1 = reward_set[i]
        r_2 = reward_set[j]

        if not check_reward_equivalence(r_1,r_2,trajectories):
            r_e +=1

    if r_e >= 1:
        # this means that we are on the furthest right branch of the tree
        print("Not Weakly Identifiable -- Most Right Branch Of the Tree")
        return 

    # now we need to check if we have a proper model
    proper_model = True
    for i,r_1 in enumerate(reward_set):
            for j,r_2 in enumerate(reward_set):
                if i != j:
                    if check_reward_equivalence(r_1,r_2,trajectories):
                       if not check_distribution_equivalence(dist[i], dist[j]):
                        proper_model = False
    if proper_model:
        print("Weakly Identifiable -- Right Branch Of the Tree")
        return
    else:
        print("Not Weakly Identifiable -- Middle Right Branch Of the Tree")
        return 


def construct_single_M(traj, env):
    n_states = env.n_states
    n_actions = env.n_actions
    horizon = env.horizon
    discount = env.discount

    M = np.zeros((horizon, n_states*n_actions))

    for i in range(horizon):
        cur_state = traj[i][0]
        cur_action = traj[i][1]

        M[i][cur_state + n_states*cur_action] = discount**i
    
    return M

def construct_entire_M(trajectories, env):
    n_states = env.n_states
    n_actions = env.n_actions
    horizon = env.horizon
    M = construct_single_M(trajectories[0], env)

    for i, traj in enumerate(trajectories):
        if i == 0:
            continue
        cur_M = construct_single_M(traj,env)
        # print(cur_M.shape)
        cur_M = np.ones((1,horizon))@cur_M

        M = np.vstack((M,cur_M))

    return M


def shaped_reward(R, env):
    n_states = env.n_states
    n_actions = env.n_actions
    discount = env.discount 

    phi = 10*np.random.randn(n_states, 1)


    shaped_reward = np.zeros_like(R)

  
    for a in range(n_actions):
        
        shaped_reward[:,a][:,None] = R[:,a][:,None] + discount*env.mdp_transition_matrices[a]@phi - phi

    return shaped_reward

if __name__ == '__main__':

    n_states = 4
    n_actions = 2
    horizon = 5
    
    env = Env(n_states,n_actions,horizon, unique_start=True, deterministic = True)
    reward = np.zeros(shape = (n_states,n_actions))
    reward[-1,:] = 10
    V, Q, pi = soft_bellman_operation(env,reward )

    traj = env.generate_trajectories()
    # print(traj[0])
    # print(len(traj))
    num_rewards = 10

    R = [np.random.randn(env.n_states, env.n_actions) for i in range(num_rewards)]

    R = np.array([[1.0,1],
                  [-1,-1],
                  [-2,-2],
                  [-3,-3]])

    s_R = shaped_reward(R,env)
   
    # V_R, Q_R, pi_R = soft_bellman_operation(env,R )
    # V_sR, Q_sR, pi_sR = soft_bellman_operation(env,s_R )

    # print(traj[25])
    # print(construct_single_M(traj[25], env))
    M = construct_entire_M(traj, env)
    Z = scipy.linalg.null_space(M)

    print("hello")
    # print(Z)


    # R[-1] = R[-2] + 2
    # R[-10] = R[-2] + 10
    # check_weak_identifiability(R,traj,env)   