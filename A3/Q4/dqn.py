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
from env import TreasureHunt_v2
import matplotlib.pyplot as plt

os.makedirs("logs",exist_ok=True)
os.makedirs("gifs",exist_ok=True)
os.makedirs("models",exist_ok=True)
os.makedirs("plots",exist_ok=True)
os.makedirs('evaluation',exist_ok=True)

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


def network_details(net):
    print(net)
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    # print(f"Trainable parameters: {trainable_params}")

class NonLinearDQN(nn.Module):
    def __init__(self, input_dim,height,width, output_dim):
        super(NonLinearDQN, self).__init__()
        self.conv_layers=nn.Sequential(
            nn.Conv2d(input_dim,64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3,stride=2,padding=1),
            nn.ReLU()
        )
        h=height//4+1
        w=width//4+1
        c_op_size=64*h*w
        self.fc_layers=nn.Sequential(
            nn.Linear(c_op_size,64),
            nn.ReLU(),
            nn.Linear(64,output_dim)
        )
    def forward(self, x):
        x=self.conv_layers(x)
        x=x.view(x.size(0),-1)
        x=self.fc_layers(x)
        return x

class PrioritizedReplayBuffer(object):
    def __init__(self,capacity,alpha=0.8):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha
        self.epsilon = 1e-6 

    def push(self, s, a, r, ns, done):
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.buffer.append((s, a, r, ns, done))
        self.priorities.append(max_priority)

    def get_sample(self, batch_size, beta=0.4, clear=False):

        priorities = np.array(self.priorities, dtype=np.float64)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities, replace=False)
        
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()  

        samples = [self.buffer[idx] for idx in indices]
        s, a, r, ns, d = zip(*samples)
        
        s = torch.from_numpy(np.stack(s)).float().to(device)
        ns = torch.from_numpy(np.stack(ns)).float().to(device)
        a = torch.tensor(a, dtype=torch.int64, device=device).unsqueeze(1)
        r = torch.tensor(r, dtype=torch.float32, device=device).unsqueeze(1)
        d = torch.tensor(d, dtype=torch.float32, device=device).unsqueeze(1)
        weights = torch.tensor(weights, dtype=torch.float32, device=device).unsqueeze(1)
        
        if clear:
            self.empty()
        
        return s, a, r, ns, d, weights, indices

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            priority = abs(td_error) + self.epsilon
            self.priorities[idx] = priority
    
    def empty(self):
        self.buffer.clear()
        self.priorities.clear()
    
    def __len__(self):
        return len(self.buffer)

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
    def __init__(self, env, network,hp, optimizer=optim.Adam,input_dim=4,output_dim=4):
        self.env = env
        self.net = network
        self.optimizer = optimizer
        
        # TODO: Initialize target network, replay buffer, epsilon, etc.
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.height=hp['height']
        self.width=hp['width']
        self.batch_size=hp['batch_size']
        self.buffer_size=hp['buffer_size']
        self.gamma=hp['gamma']
        self.lr=hp['lr']
        self.ep_start=hp['ep_start']
        self.ep=hp['ep_start']
        self.ep_end=hp['ep_end']
        self.ep_decay=hp['ep_decay']
        self.target_update=hp['target_update']
        self.num_episodes=hp['num_episodes']
        self.max_ep_length=hp['max_ep_length']
    
        self.target_network=self.net(self.input_dim,self.height,self.width,self.output_dim).to(device)
        self.Q_network=self.net(self.input_dim,self.height,self.width,self.output_dim).to(device)
        self.target_network.load_state_dict(self.Q_network.state_dict())

        # self.buffer=ReplayBuffer(self.buffer_size)
        self.buffer=PrioritizedReplayBuffer(self.buffer_size,alpha=0.8)
        self.beta_start=0.4
        self.beta_frames=hp['num_episodes']
        self.frame=0
        print(f'generating random treasure hunt gif..')
        self.opt=self.optimizer(self.Q_network.parameters(),lr=self.lr)
        self.env.visualize_policy_execution(self.get_policy_array(),'gifs/random_treasure.png')

    def select_action(self, state):
        if np.random.rand()<self.ep:
            return np.random.randint(self.env.env.num_actions)
        else:
            with torch.no_grad():
                s_tensor=torch.tensor(state,dtype=torch.float32,device=device).unsqueeze(0)
                q_values=self.Q_network(s_tensor)
                return q_values.argmax(dim=1).item()

    def get_policy_array(self):
        self.Q_network.eval()
        
        all_states = self.env.get_all_states()
        num_states = all_states.shape[0]

        policy = np.zeros((num_states, self.output_dim))
        batch_size = 128
        
        with torch.no_grad():
            for i in range(0, num_states, batch_size):
                end_idx = min(i + batch_size, num_states)
                states_batch = torch.tensor(all_states[i:end_idx], dtype=torch.float32, device=device)
                q_values = self.Q_network(states_batch).cpu().numpy()
                policy[i:end_idx] = q_values
        
        self.Q_network.train()
        return policy
        
    def update_epsilon(self,episode):
        self.ep=max(self.ep_start*(self.ep_decay**episode),self.ep_end)
    
    def optimize_model(self):
        if (len(self.buffer))<self.batch_size*5:return None

        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)


        s,a,r,ns,done,weights,indices=self.buffer.get_sample(self.batch_size,beta=beta)

        cur_q_values=self.Q_network(s).gather(1,a)
        with torch.no_grad():

            next_actions = self.Q_network(ns).argmax(1, keepdim=True)
            next_q_values = self.target_network(ns).gather(1, next_actions)
            target_q_values = r + self.gamma * next_q_values * (1 - done)

        td_errors = (cur_q_values - target_q_values).detach().cpu().numpy().flatten()

        loss=(weights * F.smooth_l1_loss(cur_q_values, target_q_values, reduction='none')).mean()

        self.buffer.update_priorities(indices,td_errors)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.Q_network.parameters(),10.0)
        self.opt.step()
        return loss.item()
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.Q_network.state_dict())

    def check_done(self,state):
        ship_map=state[3]
        ship_cord=np.unravel_index(np.argmax(ship_map),ship_map.shape)
        treasures_remaining = np.sum(state[2]) > 0
        if(not treasures_remaining) and (ship_cord==(9,9)):return True
        return False

    def train(self):
        all_ep_rewards=[]
        all_ep_loss=[]
        all_end_steps=[]
        opt_counter=0
        best_reward = -float('inf')
        st=time.time()
        for ep in range(self.num_episodes):
            s=self.env.reset()
            ep_reward=0
            done=False
            for i in range(self.max_ep_length):
                a=self.select_action(s)
                ns,r=self.env.step(a)
                done=self.check_done(ns)
                self.buffer.push(s.copy(),a,r,ns.copy(),done)
                s=ns
                ep_reward+=r
                loss=self.optimize_model()
                if loss is not None: all_ep_loss.append(loss)

                opt_counter+=1
                if (opt_counter)% self.target_update==0: 
                    opt_counter=0
                    self.update_target_network()
                if done:break
            self.update_epsilon(ep)
            all_ep_rewards.append(ep_reward)
            all_end_steps.append(i)

            k=50
            if (ep+1)%k==0:
                avg_reward=np.mean(all_ep_rewards[-k:])
                avg_steps=np.mean(all_end_steps[-k:])
                bt=time.time()
                print(f'--> '
                    f'Ep {ep+1}/{self.num_episodes} | '
                    f'Avg Reward : {avg_reward:7.2f} | '
                    f'Avg steps : {avg_steps:7.2f} | '
                    f'Max Reward : {np.max(all_ep_rewards[-k:]):4.0f} | '
                    f'Epsilon : {self.ep:.3f} | '
                    f'Time : {((bt-st)/60.0):.3f} min | '
                    )
                st=time.time()
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    torch.save(self.Q_network.state_dict(), 'models/best_model.pth')

            if (ep+1)%1000==0:
                file=f'train_{ep}.png'
                path=os.path.join('gifs',file)
                env.visualize_policy_execution(self.get_policy_array(),path)

        return all_ep_rewards
        
    def evaluate_100(self, num_episodes=100):
        self.Q_network.eval()
        total_rewards=[]
        for seed in range(num_episodes):
            s=self.env.reset()
            done=False
            ep_reward=0
            for i in range(self.max_ep_length):
                state_tensor = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                a=self.Q_network(state_tensor).argmax(dim=1).item()
                ns,r=self.env.step(a)
                done = self.check_done(ns)
                ep_reward+=r
                # print(f'iteration={i} state = {s}, next state = {ns}, reward={r}, done={done}')
                if done: break
                s=ns
            total_rewards.append(ep_reward)
        rewards=np.array(total_rewards)
        return  float(np.mean(rewards)), float(np.std(rewards)), np.max(total_rewards)


def main():
    print_star()

    hyper_parameters={
        'target_update':1000,
        'ep_start':1.0,
        'ep_decay':0.9999,
        'ep_end':0.05,
        'buffer_size':100000,
        'batch_size':64,
        'gamma':0.99,
        'lr':1e-4,
        'num_episodes':30000,
        'max_ep_length':300,
        'channels':4,
        'height':10,
        'width':10,
        'output_dim':4
    }

    print("Training TreasureHunt... ")
    print(f'Hyper Parameters : {hyper_parameters}')
    print_star()
    print('Starting training...')
    a=time.time()
    env=TreasureHunt_v2()
    trainer=DQNTrainer(env,NonLinearDQN,hyper_parameters)
    ep_rewards=trainer.train()
    print(f'Total training time : {(time.time()-a):.4f} seconds')
    best_model=copy.deepcopy(trainer.Q_network.state_dict())
    print(f'Saving best TreasureHunt model')
    model_path = os.path.join('models', 'final_treasure_hunt_best.pth')
    torch.save(best_model,model_path)   

    print('generating gif...')
    policy=trainer.get_policy_array()
    # env.visualize_policy(policy,'plots/policy_vis.png')
    env.visualize_policy_execution(policy,'gifs/final_treasure.png')
    print_star()
    print(f'Evaluation...')
    mean_r,std_r,max_r=trainer.evaluate_100()
    print(f'over 100 seeds: std : {std_r:.4f}, max reward : {max_r:.4f}')
    print_star()

if __name__=="__main__":
    print_star()
    a=time.time()
    logging('hunt')
    set_seed(42)
    env=TreasureHunt_v2()
    main()

    b=time.time()
    print_star()
    print(f'Total Execution time is {((b-a)/60.0):.4f} minutes')
    print_star()
    sys.stdout = original_stdout; current_logger.close()

