#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 13:09:52 2017

@author: Emmanouil Theofanis Chourdakis
"""

import numpy as np
import pprint
import gym

from lib.envs.gridworld import GridworldEnv
from lib.envs.blackjack import BlackjackEnv
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
from collections import defaultdict
from lib import plotting


pp = pprint.PrettyPrinter(indent=2)
env = GridworldEnv()

def policy_eval(policy, env, discount_factor=1.0, theta=0.000000000001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a (prob, next_state, reward, done) tuple.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with a random (all 0) value function
    V = np.zeros(env.nS)
    
    Vold = V
        
    k=0
    while True:
        k+=1
        delta = 0
        # TODO: implement!
        for s in range(env.nS):
            v = 0
            for a, Pa in enumerate(policy[s]):
                
                for prob, sp, r, done in env.P[s][a]:
                    v += Pa*prob*(r+discount_factor*V[sp])
            
                
            delta = max(delta, np.abs(v-V[s]))
            V[s] = v
            
        if delta < theta:
            break
        else:
            Vold = V
    print("Policy evaluation converged after {} steps".format(k))
    return np.array(V)
    
    
def value_iteration(env, discount_factor=1.0, theta=0.00001):
    # Start with random (all 0 ) value function
    V = np.zeros(env.nS)
    
    k=0
    while True:
        k+=1
        delta = 0
        for s in range(env.nS):
            v = V[s]
            action_values = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, sp, r, done in env.P[s][a]:
                    action_values[a] += prob*(r+discount_factor*V[sp])
            V[s] = np.max(action_values)
            delta = max(delta, abs(v-V[s]))
        if delta < theta:
            break
            
    policy = np.zeros([env.nS, env.nA])
    
    
    print("Value iteration converged after {} steps.".format(k))
    
    for s in range(env.nS):
        action_values = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, sp, r, done in env.P[s][a]:
                action_values[a] += prob*(r+discount_factor*V[sp])
        policy[s] = np.eye(env.nA)[np.argmax(action_values)]
        
    
    return policy, V
                
                
    
    
    
    
                
                
    
def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: Lambda discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    
    k=0
    while True:
        k+=1
        V = policy_eval_fn(policy, env, discount_factor)
        
        policy_stable = True
        for s in range(env.nS):
            # Old action
            old_action = np.argmax(policy[s])
            action_values = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, sp, r, done in env.P[s][a]:
                    action_values[a] += prob*(r+discount_factor*V[sp])
            new_action = np.argmax(action_values)
            
            if new_action != old_action:
                policy_stable = False
            else:
                policy_stable = True
            
            policy[s] = np.eye(env.nA)[new_action]
        if policy_stable:
            break
    print("Policy iteration converged after {} steps".format(k))
            
    V = policy_eval_fn(policy, env, discount_factor)
    
    return policy, V

def mc_prediction(policy, env, num_episodes, discount_factor=1.0):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given policy using sampling.
    
    Args:
        policy: A function that maps an observation to action probabilities.
        env: OpenAI gym environment.
        num_episodes: Nubmer of episodes to sample.
        discount_factor: Lambda discount factor.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    
    # The final value function
    V = defaultdict(float)
    
    # Implement this!
    

    for n in range(num_episodes):
        episode = []
        
        for t in range(100):
            s = env.reset()
            probs = policy(s)
            a = np.random.choice(np.arange(len(probs)), p=probs)
            sp, r, done, _ = env.step(a)
            episode.append((s, a, r))
            if done:
                break
            s = sp
        
        states_in_episode = set([tuple(x[0]) for x in episode])
        
        for s in states_in_episode:
            first_occurence_idx = next(i for i,x in enumerate(episode) if x[0] == s)
            G = sum([x[2]*(discount_factor**i) for i,x in enumerate(episode[first_occurence_idx:])])
            returns_sum[s] += G
            returns_count[s] += 1.0
            V[s] = returns_sum[s]/returns_count[s]
        
        

    return V
    
def sample_policy(observation):
    """
    A policy that sticks if the player score is > 20 and hits otherwise.
    """
    score, dealer_score, usable_ace = observation
    return np.array([1.0, 0.0]) if score >= 19 else np.array([0.0, 1.0])    


#
##policy, v = policy_improvement(env)
#
#policy, v = value_iteration(env)
#print("Policy Probability Distribution:")
#print(policy)
#print("")
#
#print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
#print(np.reshape(np.argmax(policy, axis=1), env.shape))
#print("")
#
#print("Value Function:")
#print(v)
#print("")
#
#print("Reshaped Grid Value Function:")
#print(v.reshape(env.shape))
#print("")
#
#fig = plt.figure()
#ax = fig.add_subplot(211, projection='3d')
#
#
#X,Y = np.meshgrid(np.arange(0,4),np.arange(0,4))
#
#Z = X**2+Y**2
#Z = np.argmax(policy,axis=1).reshape(env.shape)
#
#ax.plot_surface(Y,X,Z,rstride=1, cstride=1, cmap=cm.coolwarm)
#plt.title('Approximated Optimal Policy')
#plt.xlabel('state row')
#plt.ylabel('state column')
#
#ax = fig.add_subplot(212, projection='3d')
#
#ax.plot_surface(Y,X,v.reshape(env.shape),rstride=1, cstride=1, cmap=cm.coolwarm)
#plt.title('Value Function')
#plt.show()

plt.close('all')
env = BlackjackEnv()

def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
            
    return policy_fn


def mc_control_epsilon_greedy(env, num_episodes, discount_factor=1.0, epsilon=0.1):
    """
    Monte Carlo Control using Epsilon-Greedy policies.
    Finds an optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI gym environment.
        num_episodes: Nubmer of episodes to sample.
        discount_factor: Lambda discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
    
    Returns:
        A tuple (Q, policy).
        Q is a dictionary mapping state -> action values.
        policy is a function taht takes an observation as an argument and returns
        action probabilities
    """
    
    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    
    # Implement this!
    for n in range(num_episodes):
        s = env.reset()

        
        episode = []
        for t in range(100):
            
            prob = policy(s)
            a = np.random.choice(np.arange(len(prob)), p=prob)            
            sp, r, done, _ = env.step(a)
            episode.append((s,a,r))
            
            
            if done:
                break;
                
            s = sp
            
        state_action_pairs = set([(tuple(x[0]),x[1]) for x in episode])
        states = set([x[0] for x in episode])
        
        for s,a in state_action_pairs:
            first_occurence_idx = next(i for i,x in enumerate(episode) if x[0] == s and x[1] == a)
            G = sum([x[2]*(discount_factor**i) for i,x in enumerate(episode[first_occurence_idx:])])
            returns_sum[(s,a)] += G
            returns_count[(s,a)] += 1
            Q[s][a] = returns_sum[(s,a)]/returns_count[(s,a)]
            
            
                          
                          
        
            
        
        
    
    return Q, policy



Q, policy = mc_control_epsilon_greedy(env, num_episodes=500000, epsilon=0.1)

# For plotting: Create value function from action-value function
# by picking the best action at each state
V = defaultdict(float)
for state, actions in Q.items():
    action_value = np.max(actions)
    V[state] = action_value
plotting.plot_value_function(V, title="Optimal Value Function")

        