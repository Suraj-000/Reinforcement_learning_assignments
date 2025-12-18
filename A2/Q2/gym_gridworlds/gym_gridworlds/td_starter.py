import gymnasium
import gym_gridworlds
import gridworld
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

def tdis(env, num_episodes, seed, noise,gamma=0.99, alpha=0.1):
    max_ep_length=500
    num_actions=env.action_space.n
    start_epsilon,end_epsilon=1.0,0.05
    decay_rate=(start_epsilon-end_epsilon)/num_episodes
    start_alpha,end_alpha=0.1,0.01
    decay_alpha=(start_alpha-end_alpha)/num_episodes

    set_global_seed(SEED)
    behavior_Q = create_behaviour(noise)
    Q = defaultdict(lambda: np.zeros(num_actions))

    def behavior_policy(state):
        return np.random.choice(num_actions,p=behavior_Q[state])

    def get_behavior_prob(state, action):
        return behavior_Q[state][action]

    def target_policy(state):
        return np.argmax(Q[state])

    def get_target_prob(state, action):
        action_values=Q[state]
        weights=np.ones(num_actions)*epsilon/num_actions
        weights[np.flatnonzero(action_values==np.max(action_values))]+=1-epsilon
        return weights[action]
    
    def get_target_distribution(state, epsilon):
        probs = np.ones(num_actions) * epsilon / num_actions
        ba = target_policy(state)
        probs[ba] += (1 - epsilon)
        return probs

    episode_rewards = []
    epsilon=start_epsilon
    alpha=start_alpha
    for ep in range(num_episodes):
        s, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            a = behavior_policy(s)
            ns, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated

            b_prob = get_behavior_prob(s, a)
            pi_prob= get_target_distribution(s,epsilon)[a]
            # rho = np.clip(pi_prob / (b_prob + 1e-8), 0, 5)
            rho = pi_prob / (b_prob)
            nd= get_target_distribution(ns,epsilon)
            td_target = r if done else r + gamma * np.dot(Q[ns], nd)

            td_error = td_target - Q[s][a]
            Q[s][a] += alpha * rho * td_error
            total_reward += r
            s=ns

        epsilon = max(end_epsilon, epsilon - decay_rate)
        alpha=max(end_alpha,alpha-decay_alpha)
        episode_rewards.append(total_reward)

    
    n_states = env.observation_space.n
    final_policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        final_policy[s] = target_policy(s)
            
    return Q, final_policy, episode_rewards

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

    for ep in range(n_episodes):
        state, _ = env.reset(seed=ep)
        # reset_fn(env,ep)
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

def generate_policy_gif(env, policy, filename, max_steps=500):
    """Generate a GIF showing the policy in action."""
    frames = []
    print(f"\nGenerating GIF... saving to {filename}")
    env_render = gymnasium.make(env.spec.id, render_mode='rgb_array', random_action_prob=0.1)
    state, _ = env_render.reset(seed=504)
    reset_fn(env_render,504)
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
    # Training parameters
    num_seeds = 10  
    num_episodes = 5000 # TODO:CHANGE TO SUITABLE VALUE
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

        for seed in range(num_seeds):
            print(f"\n-- Training Seed {seed + 1}/{num_seeds}-> ",end='')
            reset_fn(env,seed)
            # Train the policy
            q_values, policy, episode_rewards = tdis(env, num_episodes, seed,noise)
            all_episode_rewards.append(episode_rewards)

            # Evaluate the trained policy
            
            mean_reward, min_reward, max_reward, std_reward, success_rate = evaluate_policy(env, policy)

            if mean_reward>max_mean_reward:
                max_mean_reward=mean_reward
                best_policy=policy
                best_q_values=q_values

            print(f"Eval results: Mean={mean_reward:.3f}, Min={min_reward:.3f}, "
                f"Max={max_reward:.3f}, Std={std_reward:.3f}, Success Rate={success_rate:.3f}")

        path=f"plots/temporal_differene_reward_curve_{noise}.png" 
        plot_ep_rewards_vs_iterations(np.mean(all_episode_rewards,axis=0),f"temporal_differene_reward_curve{noise}",path)

        print('update json file ...')
        mean_reward, min_reward, max_reward, std_reward, success_rate = evaluate_policy(env, best_policy)
        key = f"TD0_ImportanceSampling({noise})"
        value = (mean_reward, std_reward)

        json_path = 'evaluation/importance_sampling_evaluation_results.json'
        json_update(json_path, key, value)

        path=f"gifs/temporal_differene_{noise}.gif"
        generate_policy_gif(env, best_policy, filename=path)

        action_map = {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right', 4: 'Stay'}
        
        print("\nBest Optimal Policy (Action to take in each state):")
        if best_policy is not None:
            policy_grid = np.array([action_map.get(i, 'N/A') for i in best_policy]).reshape(GRID_ROWS, GRID_COLS)
            print(policy_grid)
            print(f"\nBest evaluation mean reward across seeds: {max_mean_reward:.3f}")
            print("\nQ-values for State 0 (top-left) from the best policy:")
            if 0 in best_q_values:
                for action, value in enumerate(best_q_values[0]):
                    print(f"  Action: {action_map[action]}, Q-value: {value:.3f}")
    env.close()
