import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os
from collections import deque
import json
import copy
from collections import deque
import imageio
import os
import numpy as np
import sys
import time
import matplotlib.pyplot as plt
from helper_fn import Logger,vis_agent,set_device,print_star,plot_ep_rewards_vs_iterations
import gymnasium as gym
from datetime import datetime
import matplotlib.pyplot as plt

os.makedirs("logs",exist_ok=True)
os.makedirs("gifs",exist_ok=True)
os.makedirs("models",exist_ok=True)
os.makedirs("plots",exist_ok=True)

device=set_device()
original_stdout = sys.stdout

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def logging(s='log'):
    global current_logger
    log_path = os.path.join('logs', f'output_{s}.txt')
    current_logger = Logger(log_path)
    sys.stdout = current_logger
    print("Logging started — all terminal output will also go to:", log_path)
    print("Using device:", device)


class NonLinearDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NonLinearDQN, self).__init__()
        # TODO: Implement non-linear network with hidden layers
        self.layers=nn.Sequential(
            nn.Linear(input_dim,32),
            nn.ReLU(),
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64,output_dim)
        )
    def forward(self, x):
        # TODO: Implement forward pass
        return self.layers(x)

class TempReplayBuffer:
    def __init__(self,capacity):
        self.buffer=deque(maxlen=capacity)
    
    def push(self,s,a,r,ns,na,done):
        self.buffer.append((s,a,r,ns,na,done))
    
    def get_sample(self,batch_size,clear=True):
        t=random.sample(self.buffer,batch_size)
        s,a,r,ns,na,d=zip(*t)

        s=torch.tensor(np.stack([i for i in s]),dtype=torch.float32, device=device)
        ns=torch.tensor(np.stack([i for i in ns]),dtype=torch.float32, device=device)
        a=torch.tensor(a,dtype=torch.int64,device=device).unsqueeze(1)
        na=torch.tensor(na,dtype=torch.int64,device=device).unsqueeze(1)
        r=torch.tensor(r,dtype=torch.float32,device=device).unsqueeze(1)
        d=torch.tensor(d,dtype=torch.float32,device=device).unsqueeze(1)

        if clear:self.empty()
        return s,a,r,ns,na,d

    def empty(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)

class DQNTrainer:
    def __init__(self, env, network,hp, optimizer=optim.Adam):
        self.env = env
        self.net = network
        self.optimizer = optimizer
        
        # TODO: Initialize target network, replay buffer, epsilon, etc.
        self.input_dim=8
        self.output_dim=env.action_space.n
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

        self.target_calls=0
    
        self.target_network=self.net(self.input_dim,self.output_dim).to(device)
        self.Q_network=self.net(self.input_dim,self.output_dim).to(device)
        self.target_network.load_state_dict(self.Q_network.state_dict())

        self.buffer=TempReplayBuffer(self.buffer_size)
        self.opt=self.optimizer(self.Q_network.parameters(),lr=self.lr)

    def select_action(self, state):
        s=torch.tensor(state,dtype=torch.float32,device=device).unsqueeze(0)
        if np.random.rand()<self.ep:
            return np.random.randint(self.env.action_space.n)
        else:
            with torch.no_grad():
                q_values=self.Q_network(s).cpu().numpy().squeeze(0)
                return int(np.argmax(q_values))

    def greedy_action(self,state):
        s=torch.tensor(state,dtype=torch.float32,device=device).unsqueeze(0)
        with torch.no_grad():
            q_values=self.Q_network(s).cpu().numpy().squeeze(0)
            return int(np.argmax(q_values))

    def update_epsilon(self):
        self.ep=max(self.ep_end,self.ep*self.ep_decay)
    
    def optimize_model(self):

        s,a,r,ns,na,done=self.buffer.get_sample(self.batch_size,clear=True)

        cur_q_values=self.Q_network(s).gather(1,a)
        with torch.no_grad():
            next_q_values=self.target_network(ns).gather(1,na)

        target_q_values=r + self.gamma*next_q_values*(1-done)
        loss=nn.functional.mse_loss(cur_q_values,target_q_values)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.Q_network.parameters(),10.0)
        self.opt.step()

        self.target_network.load_state_dict(self.Q_network.state_dict())

        return loss.item()

    def train(self):
        # TODO: Implement training loop with experience collection
        # Return average reward
        all_ep_rewards=[]
        all_ep_loss=[]
        max_eval_reward=-np.inf
        best_model=None

        for ep in range(self.num_episodes):
            s,_=self.env.reset()
            ep_reward=0
            done=False

            a=self.select_action(s)
            for i in range(self.max_ep_length):
                ns,r,done,_,_=self.env.step(a)
                na=self.select_action(ns)
                self.buffer.push(s,a,r,ns,na,done)
                s=ns
                a=na
                ep_reward+=r
                if (len(self.buffer))>=self.batch_size:
                    loss=self.optimize_model()
                    all_ep_loss.append(loss)
                if done:break
                
            if (ep+1)%10==0:
                self.Q_network.eval()
                m,s,mm=self.evaluate_10()
                self.Q_network.train()
                # print(f'Episode {ep+1}/{self.num_episodes} | '
                    # f'Eval 10 Results : '
                    # f'Avg Reward: {(m):7.2f} | '
                    # f'Std Reward: {(s):7.2f} | '
                    # f'Max Reward = {mm:4.0f} | '
                    # f'Epsilon: {self.ep:.3f}'
                    # )
                if m>max_eval_reward:
                    max_eval_reward=m
                    best_model=copy.deepcopy(self.Q_network.state_dict())
            k=500
            if (ep+1)%k==0:
                avg_reward=np.mean(all_ep_rewards[-k:])
                avg_loss=np.mean(all_ep_loss[-k:])
                print(f'--> '
                    f'Ep {ep+1}/{self.num_episdodes} | '
                    f'Avg Reward: {avg_reward:7.2f} | '
                    f'Avg Loss: {avg_loss:7.2f} | '
                    f'Max Reward = {np.max(all_ep_rewards[-k:]):4.0f} | '
                    # f'Safe visits ={int(safe_visits):3d} | '
                        # f'Risky visits ={int(risky_visits):3d} | '
                    f'Epsilon: {self.ep:.3f}'
                    )
                
            self.update_epsilon()
            all_ep_rewards.append(ep_reward)

        return all_ep_rewards,best_model
    
    def evaluate_10(self, num_episodes=10):
        # TODO: Implement evaluation without exploration
        # Return mean and std of evaluation rewards
        total_rewards=[]
        for seed in range(num_episodes):
            s,*_=self.env.reset(seed=seed)
            done=False
            ep_reward=0
            for i in range(1000):
                a=self.greedy_action(s)
                ns,r,te,tr, _=self.env.step(a)
                done=te or tr
                ep_reward+=r

                # print(f'iteration={i} state = {s}, next state = {ns}, reward={r}, done={done}')
                if done: break
                s=ns
            total_rewards.append(ep_reward)
        rewards=np.array(total_rewards)
        m,s,mm=np.mean(rewards),np.std(rewards),np.max(total_rewards)
        # print(f'Eval results for 10 episodes : '
        #     f'Avg Reward: {m:7.2f} | '
        #     f'Std Reward: {s:7.2f} | '
        #     f'Max Reward = {mm:4.0f} | '
        #     )
        return m,s,mm
        
    def evaluate_100(self, num_episodes=100):
        # TODO: Implement evaluation without exploration
        # Return mean and std of evaluation rewards
        self.Q_network.eval()
        total_rewards=[]
        for seed in range(num_episodes):
            s,*_=self.env.reset(seed=seed)
            done=False
            ep_reward=0
            for i in range(1000):
                with torch.no_grad():
                    a=self.greedy_action(s)

                ns,r,te,tr, _=self.env.step(a)
                done=te or tr
                ep_reward+=r

                # print(f'iteration={i} state = {s}, next state = {ns}, reward={r}, done={done}')
                if done: break
                s=ns
            total_rewards.append(ep_reward)
        rewards=np.array(total_rewards)
        return  float(np.mean(rewards)), float(np.std(rewards)), np.max(total_rewards)

def DeepSarsa(env,hps):
    a=time.time()

    trainer=DQNTrainer(env,NonLinearDQN,hps)
    all_rewards,best_model=trainer.train()

    trainer.Q_network.load_state_dict(best_model)

    print()
    print(f'generating reward plot and gif...')
    plot_ep_rewards_vs_iterations(all_rewards,'DeepSarsa : LunarLander', 'plots/all_rewards_lunar.png')
    vis_agent(env,trainer.Q_network,path='gifs/lander.gif',device=device)

    print()
    mean_r,std_r,max_r=trainer.evaluate_100()
    print(f'Eval results for 100 episodes : '
        f'Avg Reward: {mean_r:7.2f} | '
        f'Std Reward: {std_r:7.2f} | '
        f'Max Reward = {max_r:4.0f} | '
        )
    json_path='evaluation/lunarlandar_evaluation_results.json'
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    print()
    print(f'Saving best lunar model')
    torch.save(best_model,'models/best_lunar.pth')   


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
    print(f"Saved lunar evaluation results to {json_path}")
    print(f'total execution time = {((time.time()-a)/60.0):.3f} minutes')

def main(ab_path):
    print_star()
    env = gym.make('LunarLander-v3', render_mode='rgb_array')
    env.reset(seed=42)
    env.action_space.seed(42)
    env.observation_space.seed(42)

    hps={
        "target_update":100,
        'ep_start':1.0,
        'ep_decay':0.9995,
        'ep_end':0.1,
        'buffer_size':100000,
        'batch_size':64,
        'gamma':0.99,
        'lr':1e-3,
        'num_episodes':10000,
        'max_ep_length':1000,
    }
    print(f'hyper parameters : {hps}')

    print_star()
    DeepSarsa(env,hps)
    print_star()

if __name__=="__main__":
    print_star()

    set_seed(42)
    logging("lunar")
    absolute_path=os.getcwd()
    main(absolute_path)
    sys.stdout = original_stdout; current_logger.close()
