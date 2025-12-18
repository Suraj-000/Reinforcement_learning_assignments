import gymnasium
import gym_gridworlds
import numpy as np
import random
import imageio
import os
import json
import matplotlib.pyplot as plt 
from collections import defaultdict
from gymnasium.envs.registration import register
from behaviour_policies import create_behaviour

GRID_ROWS = 4
GRID_COLS = 5
SEED=42
# Register the custom environment
register(
    id="Gym-Gridworlds/Full-4x5-v0",
    entry_point="gym_gridworlds.gridworld:Gridworld",
    max_episode_steps=500,
    kwargs={"grid": "4x5_full"},
)

def set_global_seed(seed: int):
    """Set seed for reproducibility across modules."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def state_to_cord(state):
    """Convert state number to (row, col) coordinates."""
    return divmod(state, GRID_COLS)

def cord_to_state(row, col):
    """Convert (row, col) coordinates to state number."""
    return row * GRID_COLS + col

def reset_fn(env,seed=10):
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    env.np_random,_=gymnasium.utils.seeding.np_random(seed)

def monte_carlo_off_policy_control(env, num_episodes, seed, noise,gamma=0.99):

    max_ep_length=500
    num_actions=env.action_space.n
    start_epsilon,end_epsilon=1.0,0.05
    decay_rate=(start_epsilon-end_epsilon)/num_episodes
    if noise==0.01: gamma=0.9
    set_global_seed(SEED)
    behavior_Q = create_behaviour(noise)
    Q = defaultdict(lambda: np.ones(env.action_space.n)*1.0)
    C = defaultdict(lambda: np.zeros(env.action_space.n))

    def behavior_policy(state):
        return np.random.choice(num_actions,p=behavior_Q[state])

    def get_behavior_prob(state, action):
        return behavior_Q[state][action]

    def target_policy(state):
        return np.argmax(Q[state])
    
    def get_target_prob(state, action,epsilon):
        action_values=Q[state]
        weights=np.ones(num_actions)*epsilon/num_actions
        weights[np.flatnonzero(action_values==np.max(action_values))]+=1-epsilon
        return weights[action]
        
    # Main training loop
    episode_rewards = []
    epsilon=start_epsilon

    for episode in range(num_episodes):
        # Generate episode using behavior policy
        episode_data = []
        s,_ = env.reset()
        total_reward = 0.0
        done=False
        while not done:
            a  = behavior_policy(s)
            ns, r , done, truncated, _ = env.step(a)
            episode_data.append((s , a , r ))
            total_reward += r 
            s  = ns
            if done or truncated:break

        episode_rewards.append(total_reward)
        G = 0.0
        W = 1.0  

        for s , a , r in reversed(episode_data):
            G = gamma * G + r
            C[s][a] += W
            Q[s][a] += (W/C[s][a]) * (G-Q[s][a])
            
            b_prob = get_behavior_prob(s, a)
            t_prob = get_target_prob(s,a,epsilon)
            if b_prob == 0:break
            rho = t_prob / b_prob
            W *= rho
            if W == 0:break

        if noise ==0.0:epsilon = max(end_epsilon, epsilon-decay_rate)
        elif noise==0.01: epsilon= max( end_epsilon, epsilon * (0.9995 ** episode))
        else: epsilon= max( end_epsilon, epsilon * (0.9991 ** episode))

    n_states = env.observation_space.n
    final_policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        final_policy[s] = target_policy(s)
            
    return Q, final_policy,episode_rewards

def evaluate_policy(env, policy, n_episodes=100, max_steps=500):
    """
    Evaluate a given policy by running it for multiple episodes.
    
    Args:
        env: The environment
        policy: Policy to evaluate (array of actions for each state)
        n_episodes: Number of episodes to evaluate
        max_steps: Maximum steps per episode
        
    Returns:
        tuple: (mean_reward, min_reward, max_reward, std_reward, success_rate)
    """
    rewards = []
    success_rate = 0

    for episode in range(n_episodes):
        state, _ = env.reset(seed=episode)
        done = False
        episode_reward = 0
        steps = 0
        
        while not done and steps < max_steps:
            action = policy[state]
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            steps += 1
            
        rewards.append(episode_reward)
        
        # Consider episode successful if reward > 0.5
        if episode_reward > 0.5:
            success_rate += 1

    return (np.mean(rewards), np.min(rewards), np.max(rewards), 
            np.std(rewards), success_rate / n_episodes)

def generate_policy_gif(env, policy, filename, seed,max_steps=500):
    """Generate a GIF showing the policy in action."""
    frames = []
    print(f"\nGenerating GIF... saving to {filename}")
    env_render = gymnasium.make(env.spec.id, render_mode='rgb_array', random_action_prob=0.1)
    reset_fn(env_render,seed)
    state, _ = env_render.reset()
    done = False
    steps = 0
    while not done and steps < max_steps:
        action = policy[state]
        frames.append(env_render.render())
        state, reward, terminated, truncated, _ = env_render.step(action)
        done = terminated or truncated
        steps += 1
    env_render.close()
    imageio.mimsave(filename, frames, fps=7)

def plot_ep_rewards_vs_iterations(episode_rewards,algo_name,path):
    plt.figure(figsize=(10,5))
    plt.plot(episode_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title(f'{algo_name} Episode Rewards avg over 10 seeds')
    plt.savefig(f'{path}')
    # plt.show()

def json_update(filename, key, value):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
    else: data = {}
    if key not in data: data[key] = {}
    data[key]['mean'] = value[0]
    data[key]['std'] = value[1]
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"JSON file '{filename}' updated successfully.")


if __name__ == '__main__':
    num_seeds = 10  
    num_episodes = 10000  # TODO:CHANGE TO SUITABLE VALUE
    seed=10
    env = gymnasium.make('Gym-Gridworlds/Full-4x5-v0', random_action_prob=0.1)
    reset_fn(env,seed)

    noises=[0.0,0.1,0.01]

    print(f"--- Starting training across {num_seeds} seeds ---")
    for noise in noises:
        print(f"\n--- Training with noise {noise} ---")
        all_episode_rewards = []
        max_mean_reward=-float('inf')
        best_policy = None
        best_q_values = None
        best_seed=None
        for seed in range(num_seeds):
            print(f"\n-- Training Seed {seed + 1}/{num_seeds}-> ",end='')
            reset_fn(env,seed)
            # Train the policy
            q_values, policy, episode_rewards = monte_carlo_off_policy_control(env, num_episodes, seed,noise)
            all_episode_rewards.append(episode_rewards)

            mean_reward, min_reward, max_reward, std_reward, success_rate = evaluate_policy(env, policy)

            if mean_reward>max_mean_reward:
                max_mean_reward=mean_reward
                best_policy=policy
                best_q_values=q_values
                best_seed=seed
            print(f"Eval results: Mean={mean_reward:.3f}, Min={min_reward:.3f}, "
                f"Max={max_reward:.3f}, Std={std_reward:.3f}, Success Rate={success_rate:.3f}")

        path=f"plots/monte_carlo_reward_curve_{noise}.png" 
        plot_ep_rewards_vs_iterations(np.mean(all_episode_rewards,axis=0),f"monte_carlo_reward_curve{noise}",path)

        print('update json file ...')
        mean_reward, min_reward, max_reward, std_reward, success_rate = evaluate_policy(env, best_policy)
        key = f"MC_ImportanceSampling({noise})"
        value = (mean_reward, std_reward)

        json_path = 'evaluation/importance_sampling_evaluation_results.json'
        json_update(json_path, key, value)
        
        path=f"gifs/monte_carlo_{noise}.gif"
        generate_policy_gif(env, best_policy,path,best_seed)

        action_map = {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right', 4: 'Stay'}
        
        print("\nBest Optimal Policy (Action to take in each state):")
        if best_policy is not None:
            policy_grid = np.array([action_map.get(i, 'N/A') for i in best_policy]).reshape(GRID_ROWS, GRID_COLS)
            print(policy_grid)
            print(f"\nBest evaluation mean reward across seeds: {mean_reward:.3f}")
            print("\nQ-values for State 0 (top-left) from the best policy:")
            if 0 in best_q_values:
                for action, value in enumerate(best_q_values[0]):
                    print(f"  Action: {action_map[action]}, Q-value: {value:.3f}")
    env.close()
