import math
import time
from collections import deque, namedtuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import json
import copy
import sys
import os 
import random
import time
from helper_fn import Logger,vis_agent,set_device,print_star,plot_ep_rewards_vs_iterations,network_details,plot_ep_rewards_vs_iterations2,plot_a2c_curves

os.makedirs("logs",exist_ok=True)
os.makedirs("gifs",exist_ok=True)
os.makedirs("models",exist_ok=True)
os.makedirs("plots",exist_ok=True)
os.makedirs("rewards",exist_ok=True)

# device=set_device()
device=torch.device("cpu")
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
    
class ActorNet(nn.Module):
    def __init__(self, input_dim, output_dim,hidden_dim):
        super(ActorNet, self).__init__()
        self.layers=nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,output_dim)
        )
    def forward(self, x):
        return self.layers(x)

class CriticNet(nn.Module):
    def __init__(self, input_dim, output_dim,hidden_dim):
        super(CriticNet, self).__init__()
        self.layers=nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,output_dim)
        )
    def forward(self, x):
        return self.layers(x).squeeze(-1)

class A2CTrainer:
    def __init__(self,env,actor,critic,hps,optimizer=optim.Adam):

        self.env=env

        self.input_dim= env.observation_space.shape[0]
        self.output_dim =env.action_space.n
        self.gamma=hps['gamma']
        self.max_episodes=hps['num_episodes']
        self.max_ep_length=hps['max_ep_length']
        self.lr_actor=hps['lr_actor']
        self.lr_critic=hps['lr_critic']
        self.save_path=hps['save_path']
        self.log_every_episodes=hps['log_every']
        self.target_running_avg=hps['traget_running_avg']
        self.entropy_beta=hps['entropy_beta']
        self.grad_clip=hps['grad_clip']
        self.hidden_dim=hps['hidden_dim']

        self.actor=actor(self.input_dim,self.output_dim,self.hidden_dim).to(device)
        self.critic=critic(self.input_dim,1,self.hidden_dim).to(device)

        print_star()

        print(f'Network Details')
        network_details(ActorNet,self.input_dim,self.output_dim,self.hidden_dim)
        network_details(CriticNet,self.input_dim,1,self.hidden_dim)
        print_star()

        self.actor_opt=optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_opt=optim.Adam(self.critic.parameters(), lr=self.lr_critic)

    def load_and_eval_5(self,actor_losses,critic_losses,episode_rewards):
        print_star()
        print(f'Evaluation and plots...')
        print(f'plotting curves...')
        plot_a2c_curves(actor_losses,critic_losses,episode_rewards)

        env = gym.make('LunarLander-v3', render_mode='rgb_array')
        env.reset(seed=42)
        env.action_space.seed(42)
        env.observation_space.seed(42)

        a2c_path='models/a2c_lunar_lander_separate.pt'
        checkpint=torch.load(a2c_path,map_location='cpu',weights_only=False)
        save_path='plots/A2C_vs_DQN.png'
        actor=ActorNet(self.input_dim,self.output_dim,self.hidden_dim)

        actor.load_state_dict(checkpint['actor'])
        actor.eval()

        print_star()
        print(f'A2C: Evaluation and plots and gif...')
        vis_agent(env,actor,path='gifs/lander_a2c.gif',device=device)
        mean_r,std_r=self.eval_5(actor,env,5)
        save_json(mean_r,std_r,'A2C')
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

    def compute_returns_and_advantages(self,transitions, last_value, gamma=0.99):

        rewards=[t.reward for t in transitions]
        dones=[t.done for t in transitions]
        values=[t.value for t in transitions]

        returns=[]
        G=last_value

        for r,d in zip(reversed(rewards),reversed(dones)):
            if d:G=r
            else:G=r+gamma*G
            returns.insert(0,G)
        returns=torch.tensor(returns,dtype=torch.float32,device=device)

        advantages=[]
        for i in range(len(transitions)):
            v=values[i]
            if transitions[i].done:
                td_target=rewards[i]
            else:
                if i<len(transitions)-1:
                    td_target=rewards[i]+gamma*values[i+1]
                else:
                    td_target=rewards[i]+gamma*last_value
            advantage=td_target-v
            advantages.append(advantage)
        advantages=torch.tensor(advantages,dtype=torch.float32,device=device)
        return returns, advantages

    def optimize(self,states,actions,values,returns,advantages):
        logits=self.actor(states)
        probs=F.softmax(logits, dim=-1)
        dist=Categorical(probs)
        log_probs=dist.log_prob(actions)
        entropy=dist.entropy().mean()

        advantages=(advantages-advantages.mean())/(advantages.std()+1e-8)
        actor_loss=-(log_probs*advantages.detach()).mean()-self.entropy_beta*entropy

        critic_loss=F.mse_loss(values,returns.detach())

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(),self.grad_clip)
        self.actor_opt.step()

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(),self.grad_clip)
        self.critic_opt.step()

        return actor_loss.item(),critic_loss.item()

    def train(self):
        episode_rewards = []
        running_avg_window = deque(maxlen=100)
        total_steps = 0

        Transition = namedtuple("Transition", ("state", "action", "log_prob", "reward", "done", "value"))
        actor_losses, critic_losses, total_losses = [], [], []

        best_running_avg=-float('inf')
        start_time = time.time()
        print(f'Starting training...')
        for ep in range(1, self.max_episodes + 1):
            state, _ = self.env.reset()
            ep_reward = 0.0
            done = False
            transitions = []

            for t in range(self.max_ep_length):
                total_steps+=1
                s_tensor=torch.tensor(state,dtype=torch.float32,device=device).unsqueeze(0)
                logits=self.actor(s_tensor)
                probs=F.softmax(logits,dim=-1)
                dist=Categorical(probs)
                action=dist.sample()
                log_prob=dist.log_prob(action).squeeze(0)
                value=self.critic(s_tensor)

                ns,r,te,tr,_=self.env.step(action.item())
                done=te or tr
                transitions.append(Transition(state=state,action=action.item(),log_prob=log_prob,
                                            reward=r,done=done,value=value))
                ep_reward+=r
                state=ns
                if done:break
            if done:last_value=0.0
            else:
                next_state_tensor=torch.tensor(state,dtype=torch.float32,device=device).unsqueeze(0)
                last_value=self.critic(next_state_tensor).item()
            
            returns,advantages=self.compute_returns_and_advantages(transitions,last_value,gamma=self.gamma)

            states=torch.tensor([t.state for t in transitions],dtype=torch.float32,device=device)
            actions=torch.tensor([t.action for t in transitions],dtype=torch.long,device=device)
            values = self.critic(states)

            actor_loss,critic_loss=self.optimize(states,actions,values,returns,advantages)
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            # total_losses.append(total_loss.item())

            episode_rewards.append(ep_reward)
            running_avg_window.append(ep_reward)
            running_avg = np.mean(running_avg_window) if len(running_avg_window)>0 else 0.0

            if running_avg>best_running_avg and len(running_avg_window)>10:
                best_running_avg=running_avg
                torch.save({'actor':self.actor.state_dict(),'critic':self.critic.state_dict(),
                            'episode':ep,'running_avg':running_avg},self.save_path)

            if running_avg >= self.target_running_avg and len(running_avg_window) >= 100:
                print(f"Solved at episode {ep} with avg reward {running_avg:.2f}. Saving model.")
                torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, self.save_path)
                break

            if ep % self.log_every_episodes == 0:
                print(f'--> '
                    f'Ep {ep}/{self.max_episodes} | '
                    f'Avg Reward: {running_avg:7.2f} | '
                    f'Actor Loss: {actor_loss:7.2f} | '
                    f'Critic Loss: {critic_loss:7.2f} | '
                    f'Max Reward: {np.max(episode_rewards[-self.log_every_episodes:]):4.0f}'
                    )
        return actor_losses,critic_losses,episode_rewards
    
def A2C(env):
    print_star()
    print('Training A2C for lunarlander...')
    print_star()
    hps={
        'hidden_dim':128,
        'gamma':0.99,
        'lr':1e-3,
        'lr_actor':1e-3,
        'lr_critic':1e-3,
        'num_episodes':20000,
        'max_ep_length':1000,
        'log_every':100,
        'traget_running_avg':250.0,
        'entropy_beta':0.001,
        'grad_clip':0.5,
        'save_path':"models/a2c_lunar_lander_separate.pt",
        'eval_episodes':5
    }

    print(f'hyper parameters : {hps}')
    trainer=A2CTrainer(env,ActorNet,CriticNet,hps)
    actor_losses,critic_losses,episode_rewards=trainer.train()
    np.save("rewards/a2c_rewards.npy", episode_rewards)

    trainer.load_and_eval_5(actor_losses,critic_losses,episode_rewards)

def main(ab_path):
    print_star()
    a=time.time()
    env = gym.make('LunarLander-v3')
    env.reset(seed=42)
    env.action_space.seed(42)
    env.observation_space.seed(42)

    logging("lunar_A2C")
    A2C(env)

    print_star()
    b=time.time()
    print(f'Total Execution Time: {((b-a)/60.0):.3f} minutes')
    sys.stdout = original_stdout; current_logger.close()
    print_star()

if __name__=="__main__":
    set_seed(42)
    absolute_path=os.getcwd()
    main(absolute_path)