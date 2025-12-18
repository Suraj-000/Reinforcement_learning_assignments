import numpy as np
import itertools
from collections import deque

# from scipy.stats import mode
import sys
import copy
import time
import random

import json

from or_gym.envs.finance.discrete_portfolio_opt import DiscretePortfolioOptEnv,RewardShapedDiscretePortfolioOptEnv

import math
import random
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from helper_fns import Logger,print_star,plot_ep_rewards_vs_iterations,plot_ep_rewards_vs_iterations2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available(): 
    torch.cuda.manual_seed(seed)

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
device="cpu"
os.makedirs("logs",exist_ok=True)
os.makedirs("models",exist_ok=True)
os.makedirs("plots",exist_ok=True)

Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))
original_stdout = sys.stdout

def logging(s='log'):
    global current_logger
    log_path = os.path.join('logs', f'output_{s}.txt')
    current_logger = Logger(log_path)
    sys.stdout = current_logger
    print("Logging started — all terminal output will also go to:", log_path)
    print("Using device:", device)

def network_details(network,input_dim,output_dim):
    net=network(input_dim,output_dim)
    print(net)
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    # print(f"Trainable parameters: {trainable_params}")

class NonLinearDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NonLinearDQN, self).__init__()
        self.layers=nn.Sequential(
            nn.Linear(input_dim,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,output_dim)
        )
    def forward(self, x):
        return self.layers(x)

class ReplayBuffer(object):
    def __init__(self,capacity):
        self.buffer=deque(maxlen=capacity)
    
    def push(self,s,a,r,ns,done):
        self.buffer.append((s,a,r,ns,done))
    
    def get_sample(self,batch_size,clear=False):
        t=random.sample(self.buffer,batch_size)
        s,a,r,ns,d=zip(*t)

        s=torch.tensor(np.stack([i for i in s]),dtype=torch.float32, device=device)
        ns=torch.tensor(np.stack([i for i in ns]),dtype=torch.float32, device=device)
        a=torch.tensor(a,dtype=torch.int64,device=device).unsqueeze(1)
        r=torch.tensor(r,dtype=torch.float32,device=device).unsqueeze(1)
        d=torch.tensor(d,dtype=torch.float32,device=device).unsqueeze(1)

        if clear:self.empty()
        return s,a,r,ns,d

    def empty(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)

class DQNTrainer:
    def __init__(self, env, network,hp, action_list,optimizer=optim.Adam):
        self.env = env
        self.net = network
        self.optimizer = optimizer
        
        # TODO: Initialize target network, replay buffer, epsilon, etc.
        self.input_dim=hp['input_dim']
        self.output_dim=hp['output_dim']
        self.batch_size=hp['batch_size']
        self.buffer_size=hp['buffer_size']
        self.gamma=hp['gamma']
        self.lr=hp['lr']
        self.ep=hp['ep_start']
        self.ep_end=hp['ep_end']
        self.ep_decay=hp['ep_decay']
        self.num_episodes=hp['num_episodes']
        self.max_ep_length=hp['max_ep_length']
        self.target_update=hp['target_update']
        self.num_episodes=hp['num_episodes']
        self.max_ep_length=hp['max_ep_length']

        self.action_list=action_list
    
        self.target_network=self.net(self.input_dim,self.output_dim).to(device)
        self.Q_network=self.net(self.input_dim,self.output_dim).to(device)
        self.target_network.load_state_dict(self.Q_network.state_dict())

        self.buffer=ReplayBuffer(self.buffer_size)
        self.opt=self.optimizer(self.Q_network.parameters(),lr=self.lr)

    def select_action(self, state):
        if np.random.rand()<self.ep:
            return np.random.randint(self.output_dim)
        else:
            s=torch.tensor(state,dtype=torch.float32,device=device).unsqueeze(0)
            with torch.no_grad():
                q_values=self.Q_network(s)
                return int(torch.argmax(q_values,dim=1).item())

    def greedy_action(self,state):
        s=torch.tensor(state,dtype=torch.float32,device=device).unsqueeze(0)
        with torch.no_grad():
            q_values=self.Q_network(s)
            return int(torch.argmax(q_values,dim=1).item())

    def update_epsilon(self):
        self.ep=max(self.ep_end,self.ep*self.ep_decay)
    
    def optimize_model(self):
        if (len(self.buffer))<self.batch_size*10:return None

        s,a,r,ns,done=self.buffer.get_sample(self.batch_size,clear=False)

        cur_q_values=self.Q_network(s).gather(1,a)
        with torch.no_grad():
            # might have to squeeze
            next_q_values=self.target_network(ns).max(1)[0].unsqueeze(1)
            target_q_values=r + self.gamma*next_q_values*(1-done)

        loss=F.mse_loss(cur_q_values,target_q_values)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.Q_network.parameters(),10.0)
        self.opt.step()
        return loss.item()
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.Q_network.state_dict())

    def train(self):
        all_ep_rewards=[]
        all_ep_loss=[]
        for ep in range(self.num_episodes):
            s=self.env.reset()
            ep_reward=0
            done=False

            for i in range(self.max_ep_length):
                a_idx=self.select_action(s)
                action_vector=np.array(self.action_list[a_idx],dtype=np.int32)
                ns,r,done,_=self.env.step(action_vector)
                self.buffer.push(s,a_idx,r,ns,done)
                s=ns
                ep_reward+=r
                loss=self.optimize_model()
                if loss is not None: all_ep_loss.append(loss)
                if done:break
                
            if (ep+1)% self.target_update==0: self.update_target_network()
            self.update_epsilon()
            all_ep_rewards.append(ep_reward)
            k=100
            if (ep+1)%k==0:
                avg_reward=np.mean(all_ep_rewards[-k:])
                # avg_loss=np.mean(all_ep_loss[-k:])
                print(f'--> '
                    f'Ep {ep+1}/{self.num_episodes} | '
                    f'Avg Reward: {avg_reward:7.2f} | '
                    # f'Avg Loss: {avg_loss:7.2f} | '
                    f'Max Reward = {np.max(all_ep_rewards[-k:]):4.0f} | '
                    # f'Safe visits ={int(safe_visits):3d} | '
                        # f'Risky visits ={int(risky_visits):3d} | '
                    f'Epsilon: {self.ep:.3f}'
                    )
                

        return all_ep_rewards,all_ep_loss
        
    def evaluate_100(self, num_episodes=100):
        # TODO: Implement evaluation without exploration
        # Return mean and std of evaluation rewards
        self.Q_network.eval()
        all_rewards=[[] for _ in range(self.max_ep_length)]
        total_rewards=[]
        for seed in range(num_episodes):
            s=self.env.reset()
            done=False
            ep_reward=0
            for i in range(self.max_ep_length):
                a_idx=self.greedy_action(s)
                action_vector=np.array(self.action_list[a_idx],dtype=np.int32)
                ns,r,done, _=self.env.step(action_vector)
                assets,shares=s[1:6],s[6:11]
                val=np.dot(assets, shares)+s[0]
                all_rewards[i].append(val)
                ep_reward+=r
                print(f'iteration={i} state = {s}, next state = {ns}, reward={r}, done={done}')
                if done: break
                s=ns
                # print(ns)
            total_rewards.append(ep_reward)
        rewards=np.array(total_rewards)
        return  float(np.mean(rewards)), float(np.std(rewards)), np.max(total_rewards),np.array(all_rewards)

def get_action_list(num_assets,lot_size):
    per_asset=list(range(-lot_size,lot_size+1))
    all_actions=list(itertools.product(per_asset,repeat=num_assets))
    action_list=[np.array(a,dtype=int) for a in all_actions]
    return action_list

def load_model(model_path,network,input_dim,output_dim,device='cpu'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model = network(input_dim, output_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded successfully from {model_path}")
    return model

def part1():
    print_star()
    logging('portfolio')
    start_time=time.time()

    env=DiscretePortfolioOptEnv()
    input_dim=env.obs_length+1
    action_list=get_action_list(env.num_assets,env.lot_size)
    output_dim=len(action_list)
    print(f'Input dimension : {input_dim}, Output Dimension : {output_dim}')
    hp={
        'target_update':10,
        'ep_start':1.0,
        'ep_decay':0.994,
        'ep_end':0.05,
        'buffer_size':30000,
        'batch_size':128,
        'gamma':0.99,
        'lr':1e-3,
        'num_episodes':5000,
        'max_ep_length':env.step_limit,
        'input_dim':input_dim,
        'output_dim':output_dim
    }
    model_path='models/best_portfolio.pth'

    print_star()
    print(f'hyperparameters : {hp}')
    print()
    print(f'network specification...')
    network_details(NonLinearDQN,input_dim,output_dim)
    print_star()

    print('Starting training...')
    a=time.time()
    trainer=DQNTrainer(env,NonLinearDQN,hp,action_list)
    ep_rewards,all_loss=trainer.train()
    print(f'Total training time : {(time.time()-a):.4f} seconds')
    best_model=copy.deepcopy(trainer.Q_network.state_dict())
    print(f'Plotting rewards over all episodes...')
    path='plots/portfolio_rewards.png'
    plot_ep_rewards_vs_iterations(ep_rewards,'rewards_vs_episodes',path)
    print(f'Plotting loss over all optimization steps...')
    path='plots/portfolio_loss.png'
    plot_ep_rewards_vs_iterations(all_loss,'loss_vs_optimize_step',path)
    print_star()

    # print(f'using loaded model')
    # model=load_model(model_path, NonLinearDQN, input_dim, output_dim, device)
    # trainer=DQNTrainer(env,NonLinearDQN,hp,action_list)
    # trainer.Q_network=model
    mean_r,std_r,max_r,all_rewards=trainer.evaluate_100()
    print(f'Evaluation over 100 seeds: mean total wealth at the end of 10 steps : {mean_r:.4f}')
    print(f' std : {std_r:.4f}, max reward : {max_r:.4f}')
    print_star()
    
    path='plots/portfolio_mean_std_10.png'
    mean_arr=[np.mean(all_rewards[i]) for i in range(10)]
    std_arr=[np.std(all_rewards[i]) for i in range(10)]
    plot_ep_rewards_vs_iterations2(mean_arr,std_arr,'mean_reward_vs_steps',path)

    print(f'Ratio of mean and standard-deviation at the final timestep : {(mean_arr[-1]/std_arr[-1]):.4f}')
    print_star()
    json_path='evaluation/portfolio_evaluation_results.json'
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    print(f'Saving best lunar model')
    torch.save(best_model,model_path)   


    if os.path.exists(json_path):
        with open(json_path, "r") as json_file:
            json_data = json.load(json_file)
    else:
        json_data = {
            "mean_reward": 0.0,
            "std_reward": 0.0
        }
    json_data['mean_reward'] = mean_r
    json_data['std_reward'] = std_r
    with open(json_path, "w") as json_file:
        json.dump(json_data, json_file, indent=4)    
    print(f"Saved portfolio evaluation results to {json_path}")


    end_time=time.time()
    print(f"Execution time : {((end_time-start_time)/60.0):.4f} minutes")
    print_star()

def part2():
    print_star()

    start_time = time.time()

    env = RewardShapedDiscretePortfolioOptEnv()
    input_dim = env.obs_length + 1
    action_list = get_action_list(env.num_assets, env.lot_size)
    output_dim = len(action_list)

    print(f'Input dimension: {input_dim}, Output Dimension: {output_dim}')
    print(f'Using Reward-Shaped Environment for maximizing wealth at all time steps')

    hp = {
        'target_update': 10,
        'ep_start': 1.0,
        'ep_decay': 0.994,
        'ep_end': 0.05,
        'buffer_size': 30000,
        'batch_size': 128,
        'gamma': 0.99,
        'lr': 1e-3,
        'num_episodes': 5000,
        'max_ep_length': env.step_limit,
        'input_dim': input_dim,
        'output_dim': output_dim
    }
    model_path = 'models/best_portfolio_reward_shaped.pth'

    print_star()
    print(f'Hyperparameters: {hp}')
    print()
    print(f'Network specification...')
    network_details(NonLinearDQN, input_dim, output_dim)
    print_star()

    a = time.time()
    print('Starting training with reward shaping...')

    trainer = DQNTrainer(env, NonLinearDQN, hp, action_list)
    ep_rewards, all_loss = trainer.train()
    print(f'Total training time: {(time.time() - a):.4f} seconds')

    best_model = copy.deepcopy(trainer.Q_network.state_dict())
    
    print(f'Plotting rewards over all episodes...')
    path = 'plots/portfolio_rewards_shaped.png'
    plot_ep_rewards_vs_iterations(ep_rewards, 'Rewards vs Episodes (Reward Shaped)', path)

    print(f'Plotting loss over all optimization steps...')
    path = 'plots/portfolio_loss_shaped.png'
    plot_ep_rewards_vs_iterations(all_loss, 'Loss vs Optimization Step (Reward Shaped)', path)
    print_star()

    mean_r, std_r, max_r, all_rewards = trainer.evaluate_100()
    print(f'Evaluation over 100 seeds:')
    print(f'  Mean total reward at the end of 10 steps: {mean_r:.4f}')
    print(f'  Standard deviation: {std_r:.4f}')
    print(f'  Max wealth: {max_r:.4f}')
    print_star()

    path = 'plots/portfolio_mean_std_10_shaped.png'
    mean_arr = [np.mean(all_rewards[i]) for i in range(10)]
    std_arr = [np.std(all_rewards[i]) for i in range(10)]
    plot_ep_rewards_vs_iterations2(mean_arr, std_arr, 'Mean Wealth vs Steps (Reward Shaped)', path)

    end_mean = mean_arr[-1]
    end_std = std_arr[-1]
    ratio = end_mean / end_std if end_std > 0 else 0

    print(f'Mean wealth after 10 steps: {end_mean:.4f}')
    print(f'Mean std after 10 steps: {end_std:.4f}')
    print(f'Ratio of mean to standard deviation (averaged): {ratio:.4f}')

    print_star()
    json_path = 'evaluation/portfolio_evaluation_results_shaped.json'
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    if os.path.exists(json_path):
        with open(json_path, "r") as json_file:
            json_data = json.load(json_file)
    else:
        json_data = {
            "mean_reward": 0.0,
            "std_reward": 0.0
        }
    json_data['mean_reward'] = float(end_mean)
    json_data['std_reward'] = float(end_std)
    with open(json_path, "w") as json_file:
        json.dump(json_data, json_file, indent=4)    
    print(f"Saved portfolio evaluation results to {json_path}")

    print(f'Saving best model...')
    torch.save(best_model, model_path)
    print(f"Model saved to {model_path}")
    

    end_time = time.time()
    print(f"Total execution time: {((end_time - start_time) / 60.0):.4f} minutes")
    print_star()

if __name__=="__main__":
    # logging('portfolio')
    # part1()
    # sys.stdout = original_stdout; current_logger.close()
    logging('portfolio_reward_shaped')
    part2()
