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
from helper_fn import Logger,vis_agent,set_device,print_star,plot_ep_rewards_vs_iterations,network_details,plot_ep_rewards_vs_iterations2
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
    def __init__(self, input_dim, output_dim):
        super(NonLinearDQN, self).__init__()
        self.layers=nn.Sequential(
            nn.Linear(input_dim,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,output_dim)
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

        self.dqn_type=hp['dqn_type']

        self.target_calls=0
    
        self.target_network=self.net(self.input_dim,self.output_dim).to(device)
        self.Q_network=self.net(self.input_dim,self.output_dim).to(device)
        self.target_network.load_state_dict(self.Q_network.state_dict())

        self.buffer=ReplayBuffer(self.buffer_size)
        self.opt = optim.Adam(self.Q_network.parameters(), lr=self.lr, amsgrad=True)

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

    def optimize_model_ddqn(self):

        s,a,r,ns,done=self.buffer.get_sample(self.batch_size)

        cur_q_values=self.Q_network(s).gather(1,a)
        with torch.no_grad():
            na=self.Q_network(ns).argmax(1,keepdim=True)
            next_q_values=self.target_network(ns).gather(1,na)
            target_q_values=r + self.gamma*next_q_values*(1-done)
        loss=nn.functional.mse_loss(cur_q_values,target_q_values)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.Q_network.parameters(),10.0)
        self.opt.step()
        return loss.item()
    
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
        best_mean_reward=-np.inf
        best_model=None
        loss=0
        call_opt=4
        freq_opt=1
        k=100
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
                    for _ in range(freq_opt):
                        if self.dqn_type=='dqn':loss=self.optimize_model()
                        else:loss=self.optimize_model_ddqn()
                if done:break
            if (ep+1)% self.target_update==0: self.update_target_network()
            all_ep_rewards.append(ep_reward)
            self.update_epsilon()

            avg_reward=np.mean(all_ep_rewards[-k:])

            if avg_reward>best_mean_reward:
                best_mean_reward=avg_reward
                best_model=copy.deepcopy(self.Q_network.state_dict())

            if (ep+1)%(k*5)==0:
                print(f'--> '
                    f'Ep {ep+1}/{self.num_episodes} | '
                    f'Avg Reward: {avg_reward:7.2f} | '
                    f'Curr Loss: {loss:7.2f} | '
                    f'Max Reward = {np.max(all_ep_rewards[-k:]):4.0f} | '
                    f'Epsilon: {self.ep:.3f}'
                    )
                
        return all_ep_rewards,best_model
        
    def evaluate_100(self, num_episodes=100):
        self.Q_network.eval()
        total_rewards=[]
        for seed in range(num_episodes):
            s,*_=self.env.reset(seed=seed)
            done=False
            ep_reward=0
            for i in range(1000):
                with torch.no_grad():
                    a=self.greedy_action(s)

                ns,r,done,_,_=self.env.step(a)
                ep_reward+=r

                # print(f'iteration={i} state = {s}, next state = {ns}, reward={r}, done={done}')
                if done: break
                s=ns
            total_rewards.append(ep_reward)
        rewards=np.array(total_rewards)
        return  float(np.mean(rewards)), float(np.std(rewards)), np.max(total_rewards)

def DQN(env):
    print_star()
    a=time.time()
    print('Training DQN for lunarlander...')
    print_star()

    print(f'Network Details')
    input_dim=8
    output_dim=env.action_space.n
    network_details(NonLinearDQN,input_dim,output_dim)

    print()
    hps={
        'input_dim':input_dim,
        'output_dim':output_dim,
        "target_update":10,
        'ep_start':1.0,
        'ep_decay':0.9998,
        'ep_end':0.05,
        'buffer_size':100000,
        'batch_size':128,
        'gamma':0.99,
        'lr':1e-3,
        'num_episodes':20000,
        'max_ep_length':1000,
        'dqn_type':'dqn'
    }
    print(f'hyper parameters : {hps}')
    print_star()

    print(f'Training model...')
    trainer=DQNTrainer(env,NonLinearDQN,hps)
    all_rewards,best_model=trainer.train()

    trainer.Q_network.load_state_dict(best_model)

    print()
    print(f'Saving best lunar model')
    path=f"models/{hps['dqn_type']}.pth"
    torch.save(best_model,path)

    print()
    print(f'generating reward plot and gif...')
    plot_ep_rewards_vs_iterations(all_rewards,f"{hps['dqn_type']} : LunarLander", f'plots/all_rewards_lunar_{hps["dqn_type"]}.png')
    # vis_agent(env,trainer.Q_network,path=f'gifs/lander_{hps["dqn_type"]}.gif',device=device)

    print()
    mean_r,std_r,max_r=trainer.evaluate_100()
    print(f'Eval results for 100 episodes : '
        f'Avg Reward: {mean_r:7.2f} | '
        f'Std Reward: {std_r:7.2f} | '
        f'Max Reward = {max_r:4.0f} | '
        )
    save_json(mean_r,std_r,hps['dqn_type'])

    print(f'total execution time = {((time.time()-a)/60.0):.3f} minutes')
    print_star()
    return all_rewards

def DDQN(env):
    print_star()
    a=time.time()
    print('Training DDQN for lunarlander...')
    print_star()

    print(f'Network Details')
    input_dim=8
    output_dim=env.action_space.n
    network_details(NonLinearDQN,input_dim,output_dim)

    print()
    hps={
        'input_dim':input_dim,
        'output_dim':output_dim,
        "target_update":10,
        'ep_start':1.0,
        'ep_decay':0.9998,
        'ep_end':0.05,
        'buffer_size':100000,
        'batch_size':128,
        'gamma':0.99,
        'lr':1e-3,
        'num_episodes':20000,
        'max_ep_length':1000,
        'dqn_type':'ddqn'
    }
    print(f'hyper parameters : {hps}')
    print_star()

    print(f'Training model...')
    trainer=DQNTrainer(env,NonLinearDQN,hps)
    all_rewards,best_model=trainer.train()

    trainer.Q_network.load_state_dict(best_model)

    print()
    print(f'Saving best lunar model')
    path=f"models/{hps['dqn_type']}.pth"
    torch.save(best_model,path)

    print()
    print(f'generating reward plot and gif...')
    plot_ep_rewards_vs_iterations(all_rewards,f"{hps['dqn_type']} : LunarLander", f"plots/all_rewards_lunar_{hps['dqn_type']}.png")
    # vis_agent(env,trainer.Q_network,path=f"gifs/lander_{hps['dqn_type']}.gif",device=device)

    print()
    mean_r,std_r,max_r=trainer.evaluate_100()
    print(f'Eval results for 100 episodes : '
        f'Avg Reward: {mean_r:7.2f} | '
        f'Std Reward: {std_r:7.2f} | '
        f'Max Reward = {max_r:4.0f} | '
        )
    save_json(mean_r,std_r,'double_dqn')

    print(f'total execution time = {((time.time()-a)/60.0):.3f} minutes')
    print_star()
    return all_rewards

def main(ab_path):
    print_star()
    env = gym.make('LunarLander-v3')
    env.reset(seed=42)
    env.action_space.seed(42)
    env.observation_space.seed(42)

    logging("lunar_dqn")
    r_dqn = DQN(env)
    np.save("rewards/dqn_rewards.npy", r_dqn)
    print_star()
    sys.stdout = original_stdout; current_logger.close()

    logging("lunar_ddqn")
    r_ddqn=DDQN(env)
    np.save("rewards/ddqn_rewards.npy", r_ddqn)
    sys.stdout = original_stdout; current_logger.close()
    print_star()

    plot_ep_rewards_vs_iterations2(r_dqn,r_ddqn,'plots/reward_curves.png')

def load_and_eval():
    print_star()
    start_time=time.time()
    print(f'Evaluation and plot of qvalues per action...')
    dqn_path='models/dqn.pth'
    ddqn_path='models/ddqn.pth'
    save_path='plots/q_values_per_action.png'

    env = gym.make('LunarLander-v3', render_mode='rgb_array')
    env.reset(seed=42)
    env.action_space.seed(42)
    env.observation_space.seed(42)

    dqn=NonLinearDQN(8,4)
    ddqn=NonLinearDQN(8,4)

    dqn.load_state_dict(torch.load(dqn_path, map_location='cpu'))
    ddqn.load_state_dict(torch.load(ddqn_path, map_location='cpu'))

    dqn.eval()
    ddqn.eval()
    
    q_dqn=[[] for _ in range(4)]
    q_ddqn=[[] for _ in range(4)]
    timesteps=[]
    t=0
    for seed in range(100):
        s,_=env.reset(seed=seed)
        s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
        done=False
        steps=0

        while not done and steps<1000:
            with torch.no_grad():
                s_tensor=torch.tensor(s, dtype=torch.float32).unsqueeze(0)
                qv_dqn=dqn(s_tensor).detach().cpu().numpy().squeeze(0)
                qv_ddqn=ddqn(s_tensor).detach().cpu().numpy().squeeze(0)
            for a in range(4):
                q_dqn[a].append(qv_dqn[a])
                q_ddqn[a].append(qv_ddqn[a])
            timesteps.append(t)

            a=int(np.argmax(qv_dqn))
            s,_,term,trunc,_=env.step(a)
            done = term or trunc
            t+=1
            steps+=1

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()
    for a in range(4):
        ax = axes[a]
        ax.plot(timesteps, q_dqn[a], label="DQN", linewidth=1)
        ax.plot(timesteps, q_ddqn[a], label="Double DQN", linewidth=1)
        ax.set_title(f"Action {a}")
        ax.set_ylabel("Q-value")
        ax.set_ylim(-200, 200)
        ax.grid(True, linestyle="--", alpha=0.4)
        if a >= 2:ax.set_xlabel("Timestep")
        if a == 0:ax.legend()

    fig.suptitle("Per-Action Q-Value Comparison (DQN vs Double DQN)", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f'Generating Gifs...')
    vis_agent(env,dqn,path="gifs/lander_dqn.gif",device=device)
    vis_agent(env,ddqn,path="gifs/lander_ddqn.gif",device=device)

    print(f'total execution time = {((time.time()-start_time)/60.0):.3f} minutes')
    print(f"Saved Q-value comparison plot at: {save_path}")
    print_star()

if __name__=="__main__":
    set_seed(42)
    absolute_path=os.getcwd()
    main(absolute_path)
    load_and_eval()


