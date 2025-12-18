import os
import sys
import time
import json
import copy
import torch
import random
import imageio
import numpy as np
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from collections import deque
import matplotlib.pyplot as plt
import torch.nn.functional as F
from cliff import MultiGoalCliffWalkingEnv
from helper_fn import render_gif,Logger,plot_ep_rewards_vs_iterations

os.makedirs("logs",exist_ok=True)
os.makedirs("gifs",exist_ok=True)
os.makedirs("models",exist_ok=True)
os.makedirs("plots",exist_ok=True)

# Hyperparameters
NUM_SEEDS = 1

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available(): 
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


original_stdout = sys.stdout


def logging(s='1'):
    global current_logger
    log_path = os.path.join('logs', f'output_{s}.txt')
    current_logger = Logger(log_path)
    sys.stdout = current_logger
    print("Logging started — all terminal output will also go to:", log_path)
    print("Using device:", device)

def print_star(n=100):
    print("*"*n)

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

def state_to_tensor(state, env):
    """Converts the environment's discrete state into a feature tensor."""
    grid_size = env.height * env.width
    checkpoint_status = state // grid_size
    position_index = state % grid_size
    
    y = position_index // env.width
    x = position_index % env.width

    checkpoints_binary = [(checkpoint_status >> i) & 1 for i in range(2)]
    
    features = [y / env.height, x / env.width] + checkpoints_binary
    return torch.tensor(features, dtype=torch.float32,device=device).unsqueeze(0)

def network_details(net):
    # print(net)
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    # print(f"Trainable parameters: {trainable_params}")

class LinearDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearDQN, self).__init__()
        # TODO: Implement linear network
        self.layers=nn.Sequential(
            nn.Linear(input_dim,output_dim)
        )
    
    def forward(self, x):
        # TODO: Implement forward pass
        return self.layers(x)

class NonLinearDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NonLinearDQN, self).__init__()
        # TODO: Implement non-linear network with hidden layers
        self.layers=nn.Sequential(
            nn.Linear(input_dim,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,output_dim)
        )
    def forward(self, x):
        # TODO: Implement forward pass
        return self.layers(x)

class ReplayBuffer:
    def __init__(self,capacity):
        self.buffer=deque(maxlen=capacity)
    
    def push(self,s,a,r,ns,done):
        self.buffer.append((s,a,r,ns,done))
    
    def random_sample(self,batch_size):
        t=random.sample(self.buffer,batch_size)
        s,a,r,ns,d=zip(*t)
        return (
            torch.cat(s),
            torch.LongTensor(a).to(device),
            torch.FloatTensor(r).to(device),
            torch.cat(ns),
            torch.FloatTensor(d).to(device)
        )
    
    def __len__(self):
        return len(self.buffer)

class DQNTrainer:
    def __init__(self, env, network,hp, optimizer=optim.Adam,input_dim=4,output_dim=4):
        self.env = env
        self.net = network
        self.optimizer = optimizer
        
        # TODO: Initialize target network, replay buffer, epsilon, etc.
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.batch_size=hp['batch_size']
        self.buffer_size=hp['buffer_size']
        self.gamma=hp['gamma']
        self.lr=hp['lr']
        self.ep=hp['ep_start']
        self.ep_end=hp['ep_end']
        self.ep_decay=hp['ep_decay']
        self.target_update=hp['target_update']
        self.num_episodes=hp['num_episodes']
        self.max_ep_length=hp['max_ep_length']
    
        self.target_network=self.net(self.input_dim,self.output_dim).to(device)
        self.Q_network=self.net(self.input_dim,self.output_dim).to(device)
        self.target_network.load_state_dict(self.Q_network.state_dict())

        self.buffer=ReplayBuffer(self.buffer_size)
        self.opt=self.optimizer(self.Q_network.parameters(),lr=self.lr)
        
    def select_action(self, state_tensor):
        # TODO: Implement action selection
        if np.random.rand()<self.ep:
            return np.random.randint(self.env.action_space.n)
        else:
            with torch.no_grad():
                q_values=self.Q_network(state_tensor)
                return q_values.max(1)[1].item()

    def update_epsilon(self):
        self.ep=max(self.ep_end,self.ep*self.ep_decay)
    
    def optimize_model(self):
        # TODO: Implement DQN loss computation and backpropagation

        if len(self.buffer) < self.batch_size*10:return 0.0

        s,a,r,ns,done=self.buffer.random_sample(self.batch_size)

 
        cur_q_values=self.Q_network(s).gather(1,a.unsqueeze(1))
        with torch.no_grad():
            next_actions = self.Q_network(ns).argmax(1, keepdim=True)
            next_q_values = self.target_network(ns).gather(1, next_actions)
            target_q_values = r + self.gamma * next_q_values * (1 - done)

        loss=nn.MSELoss()(cur_q_values.squeeze(),target_q_values)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.item()

    def update_target_network(self):
        self.target_network.load_state_dict(self.Q_network.state_dict())

    def train(self):
        # TODO: Implement training loop with experience collection
        # Return average reward
        all_ep_rewards=[]
        safe_visits, risky_visits=0.0,0.0
        opt_counter=0
        for ep in range(self.num_episodes):
            s,_=self.env.reset()
            st=state_to_tensor(s,self.env)
            ep_reward=0
            done=False
            loss=0.0
            
            for i in range(self.max_ep_length):

                a=self.select_action(st)
                ns,r,done,_,_=self.env.step(a)
                nst=state_to_tensor(ns,self.env)
                self.buffer.push(st,a,r,nst,done)
                st=nst
                ep_reward+=r
                if r==40: safe_visits+=1
                if r==200: risky_visits+=1

                if (len(self.buffer))>=self.batch_size: loss=self.optimize_model()
                if done:break
                opt_counter+=1
                if (opt_counter)% self.target_update==0: 
                    opt_counter=0
                    self.update_target_network()
            self.update_epsilon()

            all_ep_rewards.append(ep_reward)

            if (ep+1)%500==0:
                avg_reward=np.mean(all_ep_rewards[-500:])
                print(f'--> '
                    f'Episode {ep+1}/{self.num_episodes} | '
                    f'Avg Reward: {avg_reward:7.2f} | '
                    f'max reward = {np.max(all_ep_rewards[-500:]):4.0f} | '
                    f'Safe visits ={int(safe_visits):3d} | '
                    f'Risky visits ={int(risky_visits):3d} | '
                    f'Epsilon: {self.ep:.3f}'
                    )

        return all_ep_rewards,safe_visits/self.num_episodes,risky_visits/self.num_episodes
    
    def evaluate(self, num_episodes=100):
        # TODO: Implement evaluation without exploration
        # Return mean and std of evaluation rewards
        total_rewards=[]
        safe_visits, risky_visits = 0, 0
        for seed in range(num_episodes):
            s,*_=self.env.reset(seed=seed)
            s_tensor=state_to_tensor(s,self.env)
            done=False
            ep_reward=0

            for i in range(1000):
                with torch.no_grad():
                    q_values=self.Q_network(s_tensor)
                    a=q_values.max(1)[1].item()

                ns,r,te,tr, _=self.env.step(a)
                done=te or tr
                ep_reward+=r
                s_tensor=state_to_tensor(ns,self.env)   
                if r == 40: safe_visits += 1
                elif r == 200: risky_visits += 1
                # print(f'iteration={i} state = {s}, next state = {ns}, reward={r}, done={done}')
                if done: break
            total_rewards.append(ep_reward)
        rewards=np.array(total_rewards)
        return  float(np.mean(rewards)), float(np.std(rewards)), int(safe_visits), int(risky_visits)

def main_linear(train_env,eval_env):
    input_dim = 4  # State representation dimension
    output_dim = train_env.action_space.n
    hyper_parameters1={
        'target_update':1000,
        'ep_start':1.0,
        'ep_decay':0.9998,
        'ep_end':0.1,
        'buffer_size':100000,
        'batch_size':64,
        'gamma':0.99,
        'lr':2e-3,
        'num_episodes':15000,
        'max_ep_length':1000
    }

    # Train Linear Agent
    print_star()
    print("Training Linear Agent...")
    print(f'Hyper Parameters = {hyper_parameters1}' )

    # TODO: Implement multi-seed training for linear agent
    linear_rewards_all=[]
    best_Q_linear_network_state=None
    max_reward=-float('inf')

    for seed in range(NUM_SEEDS):
        print_star(80)
        s_t=time.time()
        set_seed(seed)
        print(f'seed {seed+1}/{NUM_SEEDS}')

        trainer=DQNTrainer(train_env,LinearDQN,hyper_parameters1)
        seed_rewards,sv,rv=trainer.train()
        linear_rewards_all.append(seed_rewards)

        if np.mean(seed_rewards)>max_reward:
            max_reward=np.mean(seed_rewards)
            best_Q_linear_network_state=trainer.Q_network.state_dict()

        e_t=time.time()
        print(f'Training time for seed {seed+1}: {((e_t - s_t)/60.0):.3f} minutes')
        break
    print_star()
    print()
    print(f'Plotting avg rewards over all seeds...')
    avg_episode_rewards=np.mean(linear_rewards_all,axis=0)
    path='plots/cliff_average_rewards_linear.png'
    plot_ep_rewards_vs_iterations(avg_episode_rewards,'DQN Linear',path)

    print()
    print(f'rendering gif for best model...')
    best_Q_linear=LinearDQN(input_dim,output_dim)
    best_Q_linear.load_state_dict(best_Q_linear_network_state)
    # render_gif(train_env,trainer.Q_network)
    render_gif(train_env,best_Q_linear,filename="gifs/linear_DQN.gif")
    
    # Save best model
    print() 
    print(f'Saving best linear model')
    torch.save(best_Q_linear.state_dict(),'models/best_linear.pth') 

    print()
    print(f'Evaluating best Linear model obtained...')
    model=DQNTrainer(eval_env,LinearDQN,hyper_parameters1)
    model.Q_network.load_state_dict(best_Q_linear_network_state)
    mean_reward_linear,std_reward_linear,safe_visits,risky_visits=model.evaluate(100)
    print(f'LinearDQN Evaluation Results : mean = {mean_reward_linear:.2f}, std = {std_reward_linear:.2f}, safe visits = {safe_visits}, risky visits = {risky_visits}')\

    json_path='evaluation/cliff_evaluation_results.json'
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    if os.path.exists(json_path):
        with open(json_path, "r") as json_file:
            json_data = json.load(json_file)
    else:
        json_data = {
            "linear": {},
            "non_linear": {}
        }
    json_data.setdefault("linear", {})["mean_reward"] = mean_reward_linear
    json_data.setdefault("linear", {})["std_reward"] = std_reward_linear
    with open(json_path, "w") as json_file:
        json.dump(json_data, json_file, indent=4)
    print(f"Saved linear evaluation results to {json_path}")   

    print_star()

def main_non_linear(train_env,eval_env):
    print_star()

    input_dim = 4  # State representation dimension
    output_dim = train_env.action_space.n

    hyper_parameters={
        'target_update':1000,
        'ep_start':1.0,
        'ep_decay':0.9998,
        'ep_end':0.1,
        'buffer_size':100000,
        'batch_size':64,
        'gamma':0.99,
        'lr':1e-3,
        'num_episodes':15000,
        'max_ep_length':1000
    }
    # Train Non-Linear Agent  
    print("Training Non-Linear Agent...")
    print(f'Hyper Parameters = {hyper_parameters}' )
    # TODO: Implement multi-seed training for non-linear agent

    non_linear_rewards_all=[]
    best_Q_non_linear_network_state=None
    max_reward=-float('inf')

    for seed in range(NUM_SEEDS):
        print_star(80)
        s_t=time.time()
        set_seed(seed)
        print(f'seed {seed+1}/{NUM_SEEDS}')

        trainer=DQNTrainer(train_env,NonLinearDQN,hyper_parameters)
        seed_rewards,sv,rv=trainer.train()
        non_linear_rewards_all.append(seed_rewards)

        if np.mean(seed_rewards)>max_reward:
            max_reward=np.mean(seed_rewards)
            best_Q_non_linear_network_state=trainer.Q_network.state_dict()

        e_t=time.time()
        print(f'Training time for seed {seed+1}: {((e_t - s_t)/60.0):.3f} minutes')
        break
    print_star()
    print()
    # TODO: Create training reward plots
    print(f'Plotting avg rewards over all seeds...')
    avg_episode_rewards=np.mean(non_linear_rewards_all,axis=0)
    path='plots/cliff_average_rewards_nonlinear.png'
    plot_ep_rewards_vs_iterations(avg_episode_rewards,'DQN NonLinear',path)

    print()
    print(f'rendering gif for best model...')
    best_Q_nonlinear=NonLinearDQN(input_dim,output_dim)
    best_Q_nonlinear.load_state_dict(best_Q_non_linear_network_state)
    # render_gif(train_env,trainer.Q_network)
    render_gif(train_env,best_Q_nonlinear,filename="gifs/nonlinear_DQN.gif")

    # Save best model
    print_star()
    print(f'Saving best non linear model')
    torch.save(best_Q_nonlinear.state_dict(),'models/best_nonlinear.pth')   

    print()
    print(f'Evaluating best non Linear model obtained...')
    model=DQNTrainer(eval_env,NonLinearDQN,hyper_parameters)
    model.Q_network.load_state_dict(best_Q_non_linear_network_state)
    mean_reward_linear,std_reward_linear,safe_visits,risky_visits=model.evaluate(100)
    print(f'NonLinearDQN Evaluation Results : mean = {mean_reward_linear:.2f}, std = {std_reward_linear:.2f}, safe visits = {safe_visits}, risky visits = {risky_visits}')\

    json_path='evaluation/cliff_evaluation_results.json'
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    if os.path.exists(json_path):
        with open(json_path, "r") as json_file:
            json_data = json.load(json_file)
    else:
        json_data = {
            "linear": {},
            "non_linear": {}
        }
    json_data.setdefault("non_linear", {})["mean_reward"] = mean_reward_linear
    json_data.setdefault("non_linear", {})["std_reward"] = std_reward_linear
    with open(json_path, "w") as json_file:
        json.dump(json_data, json_file, indent=4)    
    print(f"Saved non linear evaluation results to {json_path}")   


    print_star()

def main(dir_path=''):
    big_start_time=time.time()

    print_star()
    os.makedirs(dir_path, exist_ok=True)
    os.chdir(dir_path)
    train_env = MultiGoalCliffWalkingEnv(train=True,render_mode='rgb_array')
    eval_env = MultiGoalCliffWalkingEnv(train=False)
    
    logging('non_linear')
    main_non_linear(train_env,eval_env)
    sys.stdout = original_stdout; current_logger.close()

    logging('linear')
    main_linear(train_env,eval_env)

    big_end_time=time.time()
    print(f'--> Total execution time = {((big_end_time-big_start_time)/60.0):.4f} minutes')
    print_star()

if __name__ == '__main__':
    absolute_path=os.getcwd()
    main(absolute_path)