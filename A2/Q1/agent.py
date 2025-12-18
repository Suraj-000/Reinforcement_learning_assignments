import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import imageio
import numpy as np
import sys
import json
import math
import os
import time
import random
import matplotlib.pyplot as plt
import pandas as pd 
from dataclasses import dataclass
from collections import defaultdict
from frozenlake import DiagonalFrozenLake
from cliff import MultiGoalCliffWalkingEnv
from helper_fns import initialize_Q, plot_ep_rewards_vs_iterations,get_action_boltzman,get_action_epsilon,eval_Q,softmax,plot_avg_goal_visits,eval_Q_100,print_star,render_gif,get_action_epsilon_frozen

def SARSA(env):
    '''
    Implement the SARSA algorithm to find the optimal policy for the given environment.
    return: best Q table -> np.array of shape (num_states, num_actions)
    return average training_rewards across 10 seeds-> []
    return: average safe_visits across 10 seeds -> float
    return: average risky_visits across 10 seeds -> float
    '''
    start_alpha,end_apha=0.5,0.1
    start_epsilon,end_epsilon=0.9,0.05
    decay_rate=0.999
    gamma=0.9
    num_actions=4
    num_eps=10000
    max_ep_length=1000
    all_episode_rewards=[]
    all_safe_visits, all_risky_visits=[],[]

    best_Q_all=None
    max_reward=-float('inf')
    best_seed=-1
    for seed in range(10):
        s,*_=env.reset(seed=seed)
        random.seed(seed)
        np.random.seed(seed)

        Q=initialize_Q(env,num_actions)
        episode_rewards=[]
        safe_visits, risky_visits=0.0,0.0
        epsilon=start_epsilon
        alpha=start_alpha

        for itr in range(num_eps):
            s,*_=env.reset()
            a=get_action_epsilon(s,Q,epsilon)
            total_reward=0

            for i in range(max_ep_length):
                ns, r, done, _, _=env.step(a)
                na=get_action_epsilon(s,Q,epsilon)
                Q[s,a]+= alpha*(r + gamma*Q[ns,na] - Q[s,a])

                total_reward+=r
                s,a=ns,na

                if r==40: safe_visits+=1
                elif r==200: risky_visits+=1
                if done: break

            # print(f'iteration={itr+1} total_ep_reward={total_reward}')
            epsilon=max(end_epsilon,epsilon*decay_rate)
            alpha=max(end_apha,alpha*decay_rate)
            episode_rewards.append(total_reward)

        all_episode_rewards.append(episode_rewards)
        all_safe_visits.append(safe_visits/num_eps)
        all_risky_visits.append(risky_visits/num_eps)

        if np.mean(episode_rewards)>max_reward:
            max_reward=np.mean(episode_rewards)
            best_Q_all=Q.copy()
            best_seed=seed
        # plot_ep_rewards_vs_iterations(episode_rewards,'SARSA',seed)
        print(f'SARSA: seed:{seed} avg reward={np.mean(episode_rewards)} safe visits={safe_visits}, risky visits={risky_visits}')
    print(f'Best seed for SARSA={best_seed}')
    return best_Q_all, np.mean(all_episode_rewards,axis=0), np.mean(all_safe_visits), np.mean(all_risky_visits)

def q_learning_for_cliff(env):
    '''
    Implement the Q-learning algorithm to find the optimal policy for the given environment.
    return: best Q table -> np.array of shape (num_states, num_actions)
    return average training_rewards across 10 seeds-> []
    return: average safe_visits across 10 seeds -> float
    return: average risky_visits across 10 seeds -> float
    '''
    start_alpha,end_apha=0.9,0.1
    start_epsilon,end_epsilon=1.0,0.05
    decay_rate=0.9999
    gamma=0.9
    num_actions=4
    num_eps=10000
    max_ep_length=1000
    all_episode_rewards=[]
    all_safe_visits, all_risky_visits=[],[]

    best_Q_all=None
    max_reward=-float('inf')
    best_seed=-1
    for seed in range(10):
        s,*_=env.reset(seed=seed)
        random.seed(seed)
        np.random.seed(seed)

        Q=initialize_Q(env,num_actions)
        episode_rewards=[]
        safe_visits, risky_visits=0.0,0.0
        epsilon=start_epsilon
        alpha=start_alpha

        for itr in range(num_eps):
            s,*_=env.reset()
            total_reward=0

            for i in range(max_ep_length):
                a=get_action_epsilon(s,Q,epsilon)
                ns, r, done, _, _=env.step(a)
                # q-learning update
                Q[s,a]+= alpha*(r + gamma*np.max(Q[ns]) - Q[s,a])

                total_reward+=r
                s=ns
                
                if r==40: safe_visits+=1
                if r==200: risky_visits+=1
                if done: break

            # print(f'iteration={itr+1} total_ep_reward={total_reward}')
            epsilon=max(end_epsilon,epsilon*decay_rate)
            alpha=max(end_apha,alpha*decay_rate)
            episode_rewards.append(total_reward)

        all_episode_rewards.append(episode_rewards)
        all_safe_visits.append(safe_visits/num_eps)
        all_risky_visits.append(risky_visits/num_eps)

        if np.mean(episode_rewards)>max_reward:
            max_reward=np.mean(episode_rewards)
            best_Q_all=Q.copy()
            best_seed=seed
        print(f'Q-learning: seed:{seed} avg reward={np.mean(episode_rewards)} safe visits={safe_visits}, risky visits={risky_visits}')
    print(f'Best seed for Q-learning={best_seed}')
    return best_Q_all, np.mean(all_episode_rewards,axis=0), np.mean(all_safe_visits), np.mean(all_risky_visits)

def expected_SARSA(env):
    '''
    Implement the Expected SARSA algorithm to find the optimal policy for the given environment.
    return: best Q table -> np.array of shape (num_states, num_actions)
    return average training_rewards across 10 seeds-> []
    return: average safe_visits across 10 seeds -> float
    return: average risky_visits across 10 seeds -> float
    '''
    start_alpha,end_apha=0.9,0.1
    start_epsilon,end_epsilon=1.0,0.05
    decay_rate=0.9999
    gamma=0.9
    num_actions=4
    num_eps=10000
    max_ep_length=1000
    all_episode_rewards=[]
    all_safe_visits, all_risky_visits=[],[]

    best_Q_all=None
    max_reward=-float('inf')
    best_seed=-1

    for seed in range(10):
        s,*_=env.reset(seed=seed)
        random.seed(seed)
        np.random.seed(seed)

        Q=initialize_Q(env,num_actions)
        episode_rewards=[]
        safe_visits, risky_visits=0.0,0.0
        epsilon=start_epsilon
        alpha=start_alpha

        for itr in range(num_eps):
            s,*_=env.reset()
            total_reward=0

            for i in range(max_ep_length):
                a=get_action_epsilon(s,Q,epsilon)
                ns, r, done, _, _=env.step(a)
                # expected sarsa update
                prob=softmax(Q[ns],epsilon)
                expected_val=np.dot(Q[ns],prob)

                Q[s,a]+= alpha*(r + gamma*expected_val - Q[s,a])

                total_reward+=r
                s=ns

                if r==40: safe_visits+=1
                if r==200: risky_visits+=1
                if done: break

            # print(f'iteration={itr+1} total_ep_reward={total_reward}')
            epsilon=max(end_epsilon,epsilon*decay_rate)
            alpha=max(end_apha,alpha*decay_rate)
            episode_rewards.append(total_reward)

        all_episode_rewards.append(episode_rewards)
        all_safe_visits.append(safe_visits/num_eps)
        all_risky_visits.append(risky_visits/num_eps)

        if np.mean(episode_rewards)>max_reward:
            max_reward=np.mean(episode_rewards)
            best_Q_all=Q.copy()
            best_seed=seed
        print(f'expected-SARSA: seed:{seed} avg reward={np.mean(episode_rewards)} safe visits={safe_visits}, risky visits={risky_visits}')
    print(f'Best seed for expected SARSA={best_seed}')
    return best_Q_all, np.mean(all_episode_rewards,axis=0), np.mean(all_safe_visits), np.mean(all_risky_visits)

def q_learning_for_frozenlake(env):
    '''
    Implement the Q-learning algorithm to find the optimal policy for the given environment.
    return: Q table -> np.array of shape (num_states, num_actions)
    return episode_rewards_for_one_seed -> []
    '''
    start_alpha,end_apha=0.5,0.1
    start_epsilon,end_epsilon=0.9,0.05
    decay_rate=0.999
    gamma=0.9
    num_actions=3
    num_eps=2500
    max_ep_length=16
    episode_rewards=[]

    Q=np.zeros((env.map_size*env.map_size,num_actions))
    alpha=start_alpha
    epsilon=start_epsilon
    for itr in range(num_eps):
        s,*_=env.reset()
        total_reward=0

        for i in range(max_ep_length):
            a=get_action_epsilon_frozen(s,Q,epsilon)
            ns, r, done, _, _=env.step(a)
            # q-learning update
            Q[s,a]+= alpha*(r + gamma*np.max(Q[ns]) - Q[s,a])

            total_reward+=r
            s=ns

            if done: break

        # print(f'iteration={itr+1} total_ep_reward={total_reward}')
        epsilon=max(end_epsilon,epsilon*decay_rate)
        alpha=max(end_apha,alpha*decay_rate)
        episode_rewards.append(total_reward)    
    print(f'Q-learning for frozen lake: avg reward={np.mean(episode_rewards)}')

    return Q, episode_rewards, _, _

def monte_carlo(env):
    '''
    Implement the Monte Carlo algorithm to find the optimal policy for the given environment.
    Return Q table.
    return: Q table -> np.array of shape (num_states, num_actions)
    return: episode_rewards -> []
    return: _ 
    return: _ 
    '''
    # 0.4 2500
    start_epsilon,end_epsilon=0.4,0.05
    decay_rate=0.999
    gamma=1.0
    num_actions=3
    num_eps=2500
    max_ep_length=16
    episode_rewards=[]
    
    # Q=np.random.randn(env.map_size*env.map_size,num_actions)
    Q=np.zeros((env.map_size*env.map_size,num_actions))
    # Q=np.full((env.map_size*env.map_size,num_actions),0.5)
    for i in range(env.map_size):
        idx=env.state_to_index((15,i))
        Q[idx]=0.0

    N=np.zeros((env.map_size*env.map_size,num_actions))
    epsilon=start_epsilon
    for itr in range(num_eps):
        s,*_=env.reset()
        total_reward=0
        states, actions, rewards=[],[],[]

        for i in range(max_ep_length):
            a=get_action_epsilon_frozen(s,Q,epsilon)
            ns, r, done, _, _=env.step(a)
            states.append(s)
            actions.append(a)
            rewards.append(r)
            total_reward+=r
            s=ns
            if done: break

        G=0
        visited=set()
        # first-visit mc
        for t in reversed(range(len(states))):
            G=gamma*G + rewards[t]
            s,a=states[t],actions[t]
            if (s,a) not in visited:
                visited.add((s,a))
                N[s,a]+=1
                Q[s,a]+=(1.0/N[s,a])*(G-Q[s,a])

        # print(f'iteration={itr+1} total_ep_reward={total_reward}')
        epsilon=max(end_epsilon,epsilon*decay_rate)
        episode_rewards.append(total_reward)

    return Q, np.array(episode_rewards), _, _




