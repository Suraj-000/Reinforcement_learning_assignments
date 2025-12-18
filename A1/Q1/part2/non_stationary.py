from env import FootballSkillsEnv
import numpy as np
import sys
import time
import heapq 


# random and zero initialization
def initialize_policy_and_value(num_states, num_actions, env):

    # zero initialization
    value_function=np.zeros(num_states)
    policy = np.zeros(num_states, dtype=int)

    # random initialization
    # value_function=np.random.rand(num_states)
    # policy=np.random.randint(0,num_actions,size=num_states) 

    # goal_states=env.goal_positions
    # # print(f"Goal states: {goal_states}")
    # for goal in goal_states:
    #     goal_index=env.state_to_index((goal[0],goal[1],1))
    #     goal_index2=env.state_to_index((goal[0],goal[1],0))
    #     value_function[goal_index]=0
    #     value_function[goal_index2]=0

    return policy, value_function

# policy evaluation for 20 different seed values. 
# Prints total reward for each seed 
# Prints mean reward and standard deviation of all seed values
def policy_evaluation(policy,degrade_pitch=False ,envr=FootballSkillsEnv):
    env=envr(render_mode='gif',degrade_pitch=degrade_pitch)
    rewards=[]
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

#finite horizon value iteration for horizon length 40
def Finite_horizon_value_iteration(envr=FootballSkillsEnv,get_gif=False):
    env_degraded=envr(render_mode='gif',degrade_pitch=True)

    num_states = env_degraded.grid_size**2 * 2
    num_actions = env_degraded.action_space.n
    get_transitions_count=0
    env_degraded.reset(seed=seed)
    horizon_length=40
    time_value_function=np.zeros((horizon_length+1,num_states))
    policy=np.zeros((horizon_length,num_states),dtype=int)

    env_degraded.reset(seed=seed)
    
    for time_step in reversed(range(horizon_length)):
        for state_index in range(num_states):
            state=env_degraded.index_to_state(state_index)
            if env_degraded._is_terminal(state): continue

            action_values=np.zeros(num_actions)
            for action in range(num_actions):
                transitions=env_degraded.get_transitions_at_time(state,action,time_step)
                get_transitions_count+=1
                for prob,next_state in transitions:
                    reward=env_degraded._get_reward(next_state[:2],action,state[:2])
                    action_values[action] +=  prob * (reward + discount_factor*time_value_function[time_step+1,env_degraded.state_to_index(next_state)])

            best_action = np.argmax(action_values)
            policy[time_step,state_index]=best_action
            time_value_function[time_step,state_index]=action_values[best_action]

    print(f"Number of calls to get_transitions_at_time: {get_transitions_count}")
    if get_gif: env_degraded.get_gif(policy, seed=20, filename="finite_horizon_value_iteration.gif")
    return policy, time_value_function

#degraded value iteration where env.degraded_mode=True
def Degraded_value_iteration(envr=FootballSkillsEnv,get_gif=False):
    env=envr(render_mode='gif',degrade_mode=True)
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
    
    print(f"Degraded Value iteration converged in {i+1} iterations")
    print(f"Number of calls to get_transitions_at_time: {get_transitions_count}")
    if get_gif: env.get_gif(policy, seed=20,filename='degraded_value_iteration.gif')
    return policy, value_function, i+1  

# parameters
discount_factor = 0.95
threshold = 1e-6
num_iterations = 500
seed=42

'''
Runs Finite Horizon Value iteration.
Prints  Number of calls to get_transitions_at_time: 
Total exection time: 1.9153 seconds
to render gif make get_gif=True
'''
print(f'Starting Finite Horizon Value iteration')
policy,tvf=Finite_horizon_value_iteration(get_gif=False)

print()
# degraded value iteration where env.degraded_mode=True
print(f'Degraded Value iteration')
vi_policy,vi_value_function,nof_iterations2=Degraded_value_iteration(get_gif=False)

print()
'''
Evaluates the performance of each of the two resulting policies by running 20 episodes with different seeds
Prints total reward for each seed 
Prints mean reward and standard deviation of all seed values
'''
print(f'Evaluating policy FHVI')
policy_evaluation(policy[0],degrade_pitch=True)
print(f'Evaluating policy for Degraded value iteration')
policy_evaluation(vi_policy)
