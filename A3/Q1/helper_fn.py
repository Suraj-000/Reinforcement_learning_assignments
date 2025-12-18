import numpy as np
import imageio
import os
import numpy as np
import sys
import time
import matplotlib.pyplot as plt
import torch

from datetime import datetime
import matplotlib.pyplot as plt

def state_to_tensor(state, env):
    """Converts the environment's discrete state into a feature tensor."""
    grid_size = env.height * env.width
    checkpoint_status = state // grid_size
    position_index = state % grid_size
    
    y = position_index // env.width
    x = position_index % env.width

    checkpoints_binary = [(checkpoint_status >> i) & 1 for i in range(2)]
    
    features = [y / env.height, x / env.width] + checkpoints_binary
    return torch.tensor(features, dtype=torch.float32).unsqueeze(0)


def render_gif(env,Q,filename="cliffwalk.gif",device=None):
    Q.eval()
    if device is None: device = next(Q.parameters()).device
    Q.to(device)
    s,*_=env.reset()
    done=False
    total_reward=0
    frames = []
    for i in range(1000):
        s_tensor=state_to_tensor(s,env).to(device)
        with torch.no_grad():
            q_val=Q(s_tensor)
            a=q_val.max(1)[1].item()

        ns, r, done, _, _=env.step(a)
        # print(a,r)
        total_reward+=r
        frame = env.render()
        frames.append(frame)
        s=ns    
        if done: break
    env.close()
    print(f'nof frames = {len(frames)}')
    print(f'Total eval reward = {total_reward}')
    imageio.mimsave(filename, frames, fps=6)
    print(f"Saved gif to {filename}")

def get_action_epsilon(s,Q,epsilon):
    if np.random.rand()<epsilon:
        return np.random.randint(4)
    else:
        action_values=Q[s]
        max_q=np.max(action_values)
        max_action_values=np.flatnonzero(action_values==max_q)
        return np.random.choice(max_action_values)

def get_action_boltzman(s,Q, tau=1.0):
    action_values = Q[s]
    # For numerical stability subtract max
    preferences = action_values - np.max(action_values)  
    exp_preferences = np.exp(preferences / tau)
    probs = exp_preferences / np.sum(exp_preferences)
    return np.random.choice(np.arange(4), p=probs) 


class Logger:
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def plot_ep_rewards_vs_iterations(episode_rewards,algo_name,path):
    plt.figure(figsize=(10,5))
    plt.plot(episode_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title(f'{algo_name} Episode Rewards ')
    plt.savefig(f'{path}')
    # plt.show()