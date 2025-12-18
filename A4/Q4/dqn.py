import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from collections import deque
import json
import copy
import sys
import time
from helper_fn import Logger,vis_agent,set_device,print_star,plot_ep_rewards_vs_iterations,network_details,plot_ep_rewards_vs_iterations2,plot_a2c_curves
import gymnasium as gym

os.makedirs("logs",exist_ok=True)
os.makedirs("gifs",exist_ok=True)
os.makedirs("models",exist_ok=True)
os.makedirs("plots",exist_ok=True)
os.makedirs("rewards",exist_ok=True)

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

def save_json(mean_r, std_r, algo_name="default"):
    json_path = 'evaluation/evaluation_results.json'
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    if os.path.exists(json_path):
        with open(json_path, "r") as json_file:
            json_data = json.load(json_file)
    else:
        json_data = {}

    if algo_name not in json_data:
        json_data[algo_name] = {
            "mean_reward": mean_r,
            "std_reward": std_r
        }
    else:
        json_data[algo_name]["mean_reward"] = mean_r
        json_data[algo_name]["std_reward"] = std_r

    with open(json_path, "w") as json_file:
        json.dump(json_data, json_file, indent=4)

    print(f"Saved {algo_name} evaluation results to {json_path}")

class NonLinearDQN(nn.Module):
    def __init__(self, input_dim, output_dim,hidden_dim):
        super(NonLinearDQN, self).__init__()
        self.layers=nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,output_dim)
        )
    def forward(self, x):
        return self.layers(x)

class ReplayBuffer:
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
    def __init__(self, env, network,hp, optimizer=optim.Adam):
        self.env = env
        self.net = network
        self.optimizer = optimizer
        
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
        self.hidden_dim=hp['hidden_dim']
        self.log_every_episodes=hp['log_every']
        self.dqn_type=hp['dqn_type']
        self.save_path=hp['save_path']
        self.target_running_avg=hp['traget_running_avg']
        self.target_calls=0
    
        self.target_network=self.net(self.input_dim,self.output_dim,self.hidden_dim).to(device)
        self.Q_network=self.net(self.input_dim,self.output_dim,self.hidden_dim).to(device)
        self.target_network.load_state_dict(self.Q_network.state_dict())

        self.buffer=ReplayBuffer(self.buffer_size)
        self.opt = optim.Adam(self.Q_network.parameters(), lr=self.lr, amsgrad=True)

    def load_and_eval_5(self,actor_losses,critic_losses,episode_rewards):
        print_star()
        print(f'Evaluation and plots...')
        print(f'plotting curves...')
        plot_a2c_curves(actor_losses,critic_losses,episode_rewards,algo='dqn')

        env = gym.make('LunarLander-v3', render_mode='rgb_array')
        env.reset(seed=42)
        env.action_space.seed(42)
        env.observation_space.seed(42)

        dqn_path=self.save_path
        checkpint=torch.load(dqn_path,map_location='cpu',weights_only=False)
        save_path='plots/A2C_vs_DQN.png'
        dqn=self.net(self.input_dim,self.output_dim,self.hidden_dim)

        dqn.load_state_dict(checkpint['Q_network'])
        dqn.eval()

        print_star()
        print(f'DQN: Evaluation and plots and gif...')
        vis_agent(env,dqn,path='gifs/lander_dqn.gif',device=device)
        mean_r,std_r=self.eval_5(dqn,env,5)
        save_json(mean_r,std_r,'dqn')
        print_star()

    def eval_5(self,net,env,num_episodes):
        total_rewards=[]
        for seed in range(num_episodes):
            s,_=env.reset(seed=seed)
            done=False
            ep_reward=0
            for i in range(1000):
                with torch.no_grad():
                    s_tensor=torch.tensor(s, dtype=torch.float32).unsqueeze(0)
                    qv_dqn=net(s_tensor).detach().cpu().numpy().squeeze(0)
                a=int(np.argmax(qv_dqn))
                ns,r,term,trunc,_=env.step(a)
                done = term or trunc
                ep_reward+=r
                if done:break
                s=ns
            total_rewards.append(ep_reward)
        rewards=np.array(total_rewards)
        mean_r,std_r,max_r=float(np.mean(rewards)), float(np.std(rewards)), np.max(total_rewards)
        print(f'Eval results for 5 episodes : '
            f'Avg Reward: {mean_r:7.2f} | '
            f'Std Reward: {std_r:7.2f} | '
            f'Max Reward = {max_r:4.0f} | '
            )
        return mean_r,std_r


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

    def update_target_network(self):
        self.target_network.load_state_dict(self.Q_network.state_dict())

    def optimize_model(self):

        s,a,r,ns,done=self.buffer.get_sample(self.batch_size)

        cur_q_values=self.Q_network(s).gather(1,a)
        with torch.no_grad():
            next_q_values=self.target_network(ns).max(1)[0].unsqueeze(1)
            target_q_values=r + self.gamma*next_q_values*(1-done)

        loss=nn.functional.mse_loss(cur_q_values,target_q_values)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.Q_network.parameters(),10.0)
        self.opt.step()
        return loss.item()

    def train(self):
        all_ep_rewards=[]
        all_ep_losses=[]
        best_running_avg=-float('inf')
        best_model=None
        loss=0
        running_avg_window = deque(maxlen=100)
        call_opt=4
        freq_opt=1
        for ep in range(self.num_episodes):
            s,_=self.env.reset()
            ep_reward=0
            done=False
            for i in range(self.max_ep_length):
                a=self.select_action(s)
                ns,r,done,_,_=self.env.step(a)
                self.buffer.push(s,a,r,ns,done)
                s=ns
                ep_reward+=r

                if (len(self.buffer))>=self.batch_size and i%call_opt==0:
                    for _ in range(freq_opt):loss=self.optimize_model()
                if done:break

            if (ep+1)% self.target_update==0: self.update_target_network()
            all_ep_rewards.append(ep_reward)
            all_ep_losses.append(loss)
            self.update_epsilon()
            
            running_avg_window.append(ep_reward)
            running_avg = np.mean(running_avg_window) if len(running_avg_window)>0 else 0.0

            if running_avg>best_running_avg and len(running_avg_window)>10:
                best_running_avg=running_avg
                torch.save({'Q_network':self.Q_network.state_dict(),'episode':ep,'running_avg':running_avg},self.save_path)

            if running_avg >= self.target_running_avg and len(running_avg_window) >= 100:
                print(f"Solved at episode {ep} with avg reward {running_avg:.2f}. Saving model.")
                torch.save({'Q_network':self.Q_network.state_dict(),'episode':ep,'running_avg':running_avg},self.save_path)
                break

            if ep % self.log_every_episodes == 0:
                print(f'--> '
                    f'Ep {ep}/{self.num_episodes} | '
                    f'Avg Reward: {running_avg:7.2f} | '
                    f'Loss: {loss:7.2f} | '
                    f'Max Reward: {np.max(all_ep_rewards[-self.log_every_episodes:]):4.0f}'
                    )
        return all_ep_losses,all_ep_rewards
    
def DQN(env):
    print_star()
    print('Training DQN for lunarlander...')
    print_star()

    print(f'Network Details')
    input_dim=8
    output_dim=env.action_space.n
    network_details(NonLinearDQN,input_dim,output_dim,128)

    print()
    hps={
        'input_dim':input_dim,
        'output_dim':output_dim,
        "target_update":10,
        'hidden_dim':128,
        'ep_start':1.0,
        'ep_decay':0.9998,
        'ep_end':0.05,
        'buffer_size':100000,
        'batch_size':128,
        'gamma':0.99,
        'lr':1e-3,
        'traget_running_avg':250.0,
        'log_every':100,
        'num_episodes':20000,
        'max_ep_length':1000,
        'dqn_type':'dqn',
        "save_path":'models/dqn.pt'
    }
    print(f'hyper parameters : {hps}')
    print_star()

    print(f'Training model...')
    trainer=DQNTrainer(env,NonLinearDQN,hps)
    all_ep_losses,all_ep_rewards=trainer.train()
    np.save("rewards/dqn_rewards.npy", all_ep_rewards)
    trainer.load_and_eval_5(all_ep_losses,all_ep_losses,episode_rewards=all_ep_rewards)
    

def main(ab_path):
    print_star()
    a=time.time()
    env = gym.make('LunarLander-v3')
    env.reset(seed=42)
    env.action_space.seed(42)
    env.observation_space.seed(42)

    logging("lunar_dqn")
    DQN(env)

    print_star()
    b=time.time()
    print(f'Total Execution Time: {((b-a)/60.0):.3f} minutes')
    sys.stdout = original_stdout; current_logger.close()
    print_star()

if __name__=="__main__":
    set_seed(42)
    absolute_path=os.getcwd()
    main(absolute_path)

