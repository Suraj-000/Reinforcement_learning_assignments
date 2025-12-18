import numpy as np
import sys
import time
from env import FootballSkillsEnv

# random and zero initialize 
def initialize_policy_and_value(num_states, num_actions, env):

    # zero initialization
    # value_function=np.zeros(num_states)
    # policy = np.zeros(num_states, dtype=int)

    # random initialization
    value_function=np.random.rand(num_states)
    policy=np.random.randint(0,num_actions,size=num_states) 

    goal_states=env.goal_positions
    # print(f"Goal states: {goal_states}")
    for goal in goal_states:
        goal_index=env.state_to_index((goal[0],goal[1],1))
        goal_index2=env.state_to_index((goal[0],goal[1],0))
        value_function[goal_index]=0
        value_function[goal_index2]=0

    return policy, value_function

# Policy iteration algorithm
def policy_iteration(envr=FootballSkillsEnv,get_gif=False):

    env=envr(render_mode='gif')
    num_states = env.grid_size**2 * 2
    num_actions = env.action_space.n
    get_transitions_count=0
    env.reset(seed=seed)

    #initialization
    policy, value_function = initialize_policy_and_value(num_states, num_actions,env)

    for i in range(num_iterations):
        #policy evaluation 
        while True:
            delta=0
            old_value_function = value_function.copy()

            for state_index in range(num_states):
                state=env.index_to_state(state_index)
                if env._is_terminal(state): continue
                
                action=policy[state_index]
                transitions=env.get_transitions_at_time(state,action)
                get_transitions_count+=1
                value_function[state_index] = 0
                for prob,next_state in transitions:
                    reward=env._get_reward(next_state[:2],action,state[:2])
                    value_function[state_index]+=prob*(reward + discount_factor*old_value_function[env.state_to_index(next_state)])
                delta=max(delta, np.abs(value_function[state_index]-old_value_function[state_index]))

            if delta < threshold: break

        #policy improvement
        policy_stable=True
        for state_index in range(num_states):
            state=env.index_to_state(state_index)
            if env._is_terminal(state):continue

            old_action=policy[state_index]
            action_values=np.zeros(num_actions)

            for action in range(num_actions):
                transitions=env.get_transitions_at_time(state,action)
                get_transitions_count+=1
                for prob,next_state in transitions:
                    reward=env._get_reward(next_state[:2],action,state[:2])
                    action_values[action] +=  prob * (reward + discount_factor*value_function[env.state_to_index(next_state)])
            best_action=np.argmax(action_values)
            policy[state_index]=best_action
            if old_action!=best_action:
                policy_stable=False

        if policy_stable: break

    print(f"Converged in {i+1} iterations")
    print(f"Number of calls to get_transitions_at_time: {get_transitions_count}")
    if get_gif:env.get_gif(policy, seed=20, filename="policy_iteration.gif")
    return policy, value_function, i+1  

# policy evaluation for 20 different seed values. 
# Prints total reward for each seed 
# Prints mean reward and standard deviation of all seed values
def policy_evaluation(policy, envr=FootballSkillsEnv):
    env=envr(render_mode='gif')
    rewards=[]
    #random seeds
    for i in range(20):
        ep_reward=0
        obs,_ = env.reset(seed=i)
        while True:
            state_index=env.state_to_index(obs)
            action=policy[state_index]  
            obs, reward, done, truncated, info = env.step(action) 
            # print(f'obs: {obs}, action: {action}, reward: {reward}, done: {done}, truncated: {truncated}')  
            ep_reward+=reward
            if done or truncated or env._is_terminal(obs): 
                break   
        rewards.append(ep_reward)
        env.close()
    print(f'Rewards: {rewards}, Average reward: {np.mean(rewards):.4f}, Std: {np.std(rewards):.4f}')

    return sum(rewards)


'''
To execute 1.2.5 change discount factor to 0.3 and 0.5.
'''
# parameters
discount_factor = 0.95
threshold = 1e-6
num_iterations = 1000
seed=42

'''
Runs policy iteration algorithm.
Prints  Number of calls to get_transitions_at_time: 
to render gif make get_gif=True
'''
print(f'Starting Policy iteration Gamma={discount_factor}')
pi_policy,pi_value_function,nof_iterations=policy_iteration(get_gif=False)

'''
Policy evaluation for 20 different seed values. 
Prints total reward for each seed 
Prints mean reward and standard deviation of all seed values
'''
print(f'Evaluating policy')
policy_evaluation(pi_policy)