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

def modified_value_iteration(envr=FootballSkillsEnv,get_gif=False):
    env=envr(render_mode='gif')
    num_states = env.grid_size**2 * 2
    num_actions = env.action_space.n
    get_transitions_count=0
    env.reset(seed=seed)

    policy, value_function = initialize_policy_and_value(num_states, num_actions,env)
    bellman_error_queue=[]

    # compute initial bellman error
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
        bellman_error=max(action_values)
        delta=np.abs(value_function[state_index] - bellman_error)
        heapq.heappush(bellman_error_queue,(-delta,state_index))

    while bellman_error_queue:
        neg_delta,state_index=heapq.heappop(bellman_error_queue)
        delta = -neg_delta

        if delta<threshold: continue

        state=env.index_to_state(state_index)
        if env._is_terminal(state): continue

        action_values=np.zeros(num_actions)
        for action in range(num_actions):
            transitions=env.get_transitions_at_time(state,action)
            get_transitions_count+=1
            for prob,next_state in transitions:
                reward=env._get_reward(next_state[:2],action,state[:2])
                action_values[action] +=  prob * (reward + discount_factor*value_function[env.state_to_index(next_state)])

        best_value=np.max(action_values)
        old_value=value_function[state_index]
        new_value=best_value
        best_action=np.argmax(action_values)
        new_delta=np.abs(best_value-old_value)
        value_function[state_index]=new_value
        policy[state_index]=best_action
        heapq.heappush(bellman_error_queue,(-new_delta,state_index))


    print(f"Modified Value iteration converged")
    print(f"Number of calls to get_transitions_at_time: {get_transitions_count}")
    if get_gif: env.get_gif(policy, seed=20,filename='modified_value_iteration.gif')
    return policy, value_function, 1  


discount_factor = 0.95
threshold = 1e-6
num_iterations = 500
seed=42

'''
Runs policy iteration algorithm.
Prints  Number of calls to get_transitions_at_time: 
to render gif make get_gif=True
'''
print()
print(f'Starting Modified Value iteration')
p,v,_=modified_value_iteration(get_gif=False)

'''
Evaluates the performance of each of the two resulting policies by running 20 episodes with different seeds
Prints total reward for each seed 
Prints mean reward and standard deviation of all seed values
'''

print(f'Evaluating policy modified value iteration')
policy_evaluation(p)