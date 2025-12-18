from env import FootballSkillsEnv
import numpy as np
import sys
import time

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

# Value iteration algorithm
def value_iteration(envr=FootballSkillsEnv,get_gif=False):
    env=envr(render_mode='gif')
    num_states = env.grid_size**2 * 2
    num_actions = env.action_space.n
    get_transitions_count=0
    env.reset(seed=seed)

    policy, value_function = initialize_policy_and_value(num_states, num_actions,env)

    for i in range(num_iterations):
        delta=0
        for state_index in range(num_states):
            old_value=value_function[state_index]
            state=env.index_to_state(state_index)
            if env._is_terminal(state): continue

            action_values=np.zeros(num_actions)
            for action in range(num_actions):
                transitions=env.get_transitions_at_time(state,action)
                get_transitions_count+=1
                for prob, next_state in transitions:
                    reward=env._get_reward(next_state[:2],action,state[:2])
                    action_values[action] += prob * (reward + discount_factor*value_function[env.state_to_index(next_state)])
            best_value = np.max(action_values)
            value_function[state_index]=best_value
            delta=max(delta, np.abs(best_value - old_value))
        if delta<threshold: break

    #Optimal policy
    for state_index in range(num_states):
        state=env.index_to_state(state_index)
        if env._is_terminal(state): continue

        action_values=np.zeros(num_actions)
        for action in range(num_actions):
            transitions=env.get_transitions_at_time(state,action)
            get_transitions_count+=1
            for prob,next_state in transitions:
                reward=env._get_reward(next_state[:2],action,state[:2])
                action_values[action] +=  prob * (reward + discount_factor*value_function[env.state_to_index(next_state)])

        policy[state_index]=np.argmax(action_values)
    
    print(f"Value iteration converged in {i+1} iterations")
    print(f"Number of calls to get_transitions_at_time: {get_transitions_count}")
    if get_gif: env.get_gif(policy, seed=20,filename="Value_iteration.gif")
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
discount_factor = 0.95
threshold = 1e-6
num_iterations = 1000
seed=42

'''
Runs policy iteration algorithm.
Prints  Number of calls to get_transitions_at_time: 
to render gif make get_gif=True
'''
print(f'Starting Value iteration Gamma={discount_factor}')
vi_policy,vi_value_function,nof_iterations2=value_iteration(get_gif=False)

'''
Policy evaluation for 20 different seed values. 
Prints total reward for each seed 
Prints mean reward and standard deviation of all seed values
'''
print(f'Evaluating policy')
policy_evaluation(vi_policy)

