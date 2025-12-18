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
    print(f'Total reward = {total_r}')
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
    plt.figure(figsize=(14,12))
    plt.plot(episode_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title(f'{algo_name} Episode Rewards ')
    plt.savefig(f'{path}')
    # plt.show()

def plot_ep_rewards_vs_iterations2(dqn, ddqn, path, window=50):
    plt.figure(figsize=(14,12))

    plt.plot(dqn, alpha=0.5, label='DQN (raw)')
    plt.plot(ddqn, alpha=0.5, label='DDQN (raw)')

    dqn_mean = np.convolve(dqn, np.ones(window)/window, mode='valid')
    ddqn_mean = np.convolve(ddqn, np.ones(window)/window, mode='valid')

    plt.plot(dqn_mean, linewidth=2, label=f'DQN {window}-episode mean')
    plt.plot(ddqn_mean, linewidth=2, label=f'DDQN {window}-episode mean')

    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('Episode Rewards Comparison: DQN vs DDQN')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    # plt.show()

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


def network_details(network,input_dim,output_dim):
    net=network(input_dim,output_dim)
    print(net)
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    # print(f"Trainable parameters: {trainable_params}")
