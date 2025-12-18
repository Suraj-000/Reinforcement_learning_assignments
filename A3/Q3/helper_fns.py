import numpy as np
import imageio
import os
import numpy as np
import sys
import time
import matplotlib.pyplot as plt
import torch
import random

from datetime import datetime
import matplotlib.pyplot as plt

def vis_agent(env,net,path='lander.gif',device='cpu'):
    s,_=env.reset()
    total_r=0.0
    frames=[]
    while True:
        state=torch.tensor(s,dtype=torch.float32,device=device).unsqueeze(0)
        with torch.no_grad():
            q_values=net(state).cpu().numpy().squeeze(0)
            a=int(np.argmax(q_values))

        s,r,terminated,truncated,info=env.step(a)
        total_r+=r
        frames.append(env.render())
        if terminated or truncated:break
    env.close()
    print(f'nof frames = {len(frames)}')
    print(f'Total eval reward = {total_r}')
    imageio.mimsave(path, frames, fps=25)
    print(f"Saved gif to {path}")

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
    plt.figure(figsize=(10,8))
    plt.plot(episode_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title(f'{algo_name} Episode Rewards ')
    plt.savefig(f'{path}')

def plot_ep_rewards_vs_iterations2(mean_r,std_r,algo_name,path):
    plt.figure(figsize=(10,8))
    plt.plot(mean_r, label='Mean Reward',linewidth=2.5, marker='o', markersize=5)
    plt.plot(std_r, label='Std Reward',linewidth=2.5, marker='s', markersize=5)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title(f'{algo_name} Episode Rewards')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def print_star(n=100):
    print("*"*n)

def network_details(net):
    # print(net)
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    # print(f"Trainable parameters: {trainable_params}")

def set_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available(): 
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device