import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os
from torch.distributions import Normal
import matplotlib.pyplot as plt
from collections import deque
import json
import copy
import sys
import time
from helper_fn import Logger,vis_agent,set_device,print_star,plot_ep_rewards_vs_iterations,network_details,plot_ep_rewards_vs_iterations2,make_json_serializable
import gymnasium as gym

os.makedirs("logs",exist_ok=True)
os.makedirs("models",exist_ok=True)
os.makedirs("plots",exist_ok=True)

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

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim,hidden_dim):
        super(PolicyNetwork, self).__init__()
        self.layers=nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU()
        )
        self.mean=nn.Linear(hidden_dim,output_dim)
        self.log_std=nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        x=self.layers(x)
        mean=self.mean(x)
        std=torch.exp(self.log_std).unsqueeze(0).expand_as(mean)
        std=torch.clamp(std,min=1e-6)
        return mean, std

class ValueNetwork(nn.Module):
    def __init__(self,input_dim,hidden_dim):
        super(ValueNetwork,self).__init__()
        self.layers=nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1)
        )

    def forward(self,x):
        return self.layers(x).squeeze(-1)

class REINFORCETrainer:
    def __init__(self,env,network,hp,optimizer=optim.Adam):
        self.env = env
        self.net = network
        self.optimizer = optimizer

        self.input_dim=hp['input_dim']
        self.output_dim=hp['output_dim']
        self.action_low=hp['action_low']
        self.action_high=hp['action_high']
        self.hidden_dim=hp['hidden_dim']
        self.batch_size=hp['batch_size']
        self.gamma=hp['gamma']
        self.lr=hp['lr']
        self.num_episodes=hp['num_episodes']
        self.max_ep_length=hp['max_ep_length']
        self.algo=hp['algo']

        self.policy=PolicyNetwork(self.input_dim,self.output_dim,self.hidden_dim).to(device)
        self.opt=optim.Adam(self.policy.parameters(),lr=self.lr,amsgrad=True)

        if self.algo=='value_fn_baseline':
            self.value_net=ValueNetwork(self.input_dim,self.hidden_dim).to(device)
            self.value_opt=optim.Adam(self.value_net.parameters(),lr=self.lr,amsgrad=True)

        self.ep_rewards_deque=deque(maxlen=100)

    def compute_returns(self,rewards):
        all_returns=[]
        for ep_r in rewards:
            l=len(ep_r)
            G=sum((self.gamma**t)*r for t,r in enumerate(ep_r))
            all_returns.extend([G]*l)
        return np.array(all_returns,dtype=np.float32)
    
    def compute_returns_rewards_to_go(self,rewards):
        all_returns=[]
        for ep_r in rewards:
            returns_ep=[]
            G=0.0
            for r in reversed(ep_r):
                G=r+self.gamma*G
                returns_ep.insert(0,G)
            all_returns.extend(returns_ep)
        return np.array(all_returns,dtype=np.float32)
    
    def compute_returns_with_avg_baseline(self,rewards):
        all_returns=[]
        for ep_r in rewards:
            l=len(ep_r)
            G=sum((self.gamma**t)*r for t,r in enumerate(ep_r))
            all_returns.extend([G]*l)
        all_returns=np.array(all_returns,dtype=np.float32)
        baseline=np.mean(all_returns)
        return all_returns - baseline

    def optimize_model(self,all_rewards,all_logs):
        if self.algo=='no_baseline':
            returns=self.compute_returns(all_rewards)
            returns=(returns-returns.mean())/(returns.std()+1e-8)
        elif self.algo=='reward_to_go':
            returns=self.compute_returns_rewards_to_go(all_rewards)
            returns=(returns-returns.mean())/(returns.std()+1e-8)
        elif self.algo=='avg_reward_baseline':
            returns=self.compute_returns_with_avg_baseline(all_rewards)

        returns_t=torch.tensor(returns,dtype=torch.float32,device=device)
        log_prob_t = torch.cat(all_logs).to(device)

        assert returns_t.shape == log_prob_t.shape, f"shapes mismatch: returns {returns_t.shape} vs logs {log_prob_t.shape}"

        loss=-(log_prob_t*returns_t).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()
    
    def optimize_value_model(self,all_rewards,all_logs,all_states):

        returns=self.compute_returns_rewards_to_go(all_rewards)
        returns_t=torch.tensor(returns,dtype=torch.float32,device=device)

        states_t=torch.tensor(np.array(all_states,dtype=np.float32),dtype=torch.float32,device=device)
        values=self.value_net(states_t)
        advantages=returns_t-values.detach()

        adv_mean=advantages.mean()
        adv_std=advantages.std()+1e-8
        norm_adv=(advantages-adv_mean)/adv_std

        log_prob_t=torch.cat(all_logs).to(device)
        assert returns_t.shape == log_prob_t.shape, f"shapes mismatch: returns {returns_t.shape} vs logs {log_prob_t.shape}"

        policy_loss=-(log_prob_t*norm_adv).mean()

        value_loss=F.mse_loss(values,returns_t)

        self.opt.zero_grad()
        policy_loss.backward()
        self.opt.step()

        self.value_opt.zero_grad()
        value_loss.backward()
        self.value_opt.step()

        return float(policy_loss.item())
        
    def get_episode(self):
        states,actions,rewards,logs=[],[],[],[]
        s,_=self.env.reset()
        done=False
        for i in range(self.max_ep_length):
            s_tensor=torch.tensor(s,dtype=torch.float32,device=device).unsqueeze(0)
            mean,std=self.policy(s_tensor)
            dist=Normal(mean,std)
            action=dist.sample()
            log_prob=dist.log_prob(action).sum(dim=-1)
            action_np=action.detach().cpu().numpy()[0]

            a=np.clip(action_np,self.action_low,self.action_high)

            ns,r,te,tr,_=self.env.step(a)
            done=te or tr

            states.append(s)
            actions.append(action_np)
            rewards.append(float(r))
            logs.append(log_prob)
            s=ns
            if done:break
        return states,actions,rewards,logs

    def train(self):
        all_ep_rewards=[]
        best_mean_reward=-np.inf
        best_model=None
        episode=0
        while episode<self.num_episodes:
            # all_states,all_actions,all_rewards,all_logs=[],[],[],[]
            batch_states,batch_actions,batch_rewards,batch_logs=[],[],[],[]

            for batch in range(self.batch_size):
                ep_s,ep_a,ep_r,ep_logs=self.get_episode()

                batch_states.extend(ep_s)
                batch_actions.extend(ep_a)
                batch_rewards.append(ep_r)
                batch_logs.extend(ep_logs)

                ep_total=sum(ep_r)
                self.ep_rewards_deque.append(ep_total)
                all_ep_rewards.append(ep_total)
                episode+=1

            if self.algo=='value_fn_baseline':loss_val=self.optimize_value_model(batch_rewards,batch_logs,batch_states)
            else: loss_val=self.optimize_model(batch_rewards,batch_logs)
            
            avg_reward=np.mean(list(self.ep_rewards_deque)) if len(self.ep_rewards_deque)>0 else 0.0

            if avg_reward>best_mean_reward:
                best_mean_reward=avg_reward
                best_model=copy.deepcopy(self.policy.state_dict())
            if episode%100==0:
                print(f'--> '
                    f'Ep {episode}/{self.num_episodes} | '
                    f'Avg Reward: {avg_reward:7.2f} | '
                    f'Curr Loss: {loss_val:7.2f} | '
                    f'Max Reward = {np.max(all_ep_rewards[-self.batch_size:]):4.0f} | '
                    )
            if avg_reward>500.0:
                print(f'Avg reward reach over 500 i.e., {avg_reward}')
                break
        return all_ep_rewards,best_model

def reinforce_no_baseline(env,hps):
    print_star()
    start_time=time.time()
    print('Training REINFORCE FOR Inverted Pendulum with NO BASELINE...')
    print_star()

    print(f'hyper parameters : {hps}')
    print_star()

    print(f'Training model...')
    trainer=REINFORCETrainer(env,PolicyNetwork,hps)
    all_rewards,best_model=trainer.train()

    print()
    print(f'Saving REINFORCE NO BASELINE model...')
    path=f"models/reinforce_{hps['algo']}.pth"
    torch.save(best_model,path)

    print()
    print(f'generating reward plots...')
    plot_ep_rewards_vs_iterations(all_rewards,f"REINFORCE NO BASELINE : InvertedPendulum", f"plots/all_rewards_{hps['algo']}.png")

    print(f'total execution time = {((time.time()-start_time)/60.0):.3f} minutes')
    print_star()

def reinforce_reward_to_go(env,hps):
    print_star()
    start_time=time.time()
    print('Training REINFORCE FOR Inverted Pendulum with Reward-to-go baseline...')
    print_star()

    print(f'hyper parameters : {hps}')
    print_star()

    print(f'Training model...')
    trainer=REINFORCETrainer(env,PolicyNetwork,hps)
    all_rewards,best_model=trainer.train()

    print()
    print(f'Saving REINFORCE Reward-to-go baseline model...')
    path=f"models/reinforce_{hps['algo']}.pth"
    torch.save(best_model,path)

    print()
    print(f'generating reward plots...')
    plot_ep_rewards_vs_iterations(all_rewards,f"REINFORCE Reward-to-go baseline : InvertedPendulum", f"plots/all_rewards_{hps['algo']}.png")

    print(f'total execution time = {((time.time()-start_time)/60.0):.3f} minutes')
    print_star()

def reinforce_average_reward(env,hps):
    print_star()
    start_time=time.time()
    print('Training REINFORCE FOR Inverted Pendulum with Average reward baseline...')
    print_star()

    print(f'hyper parameters : {hps}')
    print_star()

    print(f'Training model...')
    trainer=REINFORCETrainer(env,PolicyNetwork,hps)
    all_rewards,best_model=trainer.train()

    print()
    print(f'Saving REINFORCE Average reward baseline model...')
    path=f"models/reinforce_{hps['algo']}.pth"
    torch.save(best_model,path)

    print()
    print(f'generating reward plots...')
    plot_ep_rewards_vs_iterations(all_rewards,f"REINFORCE Average reward baseline : InvertedPendulum", f"plots/all_rewards_{hps['algo']}.png")

    print(f'total execution time = {((time.time()-start_time)/60.0):.3f} minutes')
    print_star()

def reinforce_value_baseline(env,hps):
    print_star()
    start_time=time.time()
    print('Training REINFORCE FOR Inverted Pendulum with Value-function baseline...')
    print_star()

    print(f'hyper parameters : {hps}')
    print_star()

    print(f'Training model...')
    trainer=REINFORCETrainer(env,PolicyNetwork,hps)
    all_rewards,best_model=trainer.train()

    print()
    print(f'Saving REINFORCE Value-baseline model (policy weights saved)...')
    path=f"models/reinforce_{hps['algo']}.pth"
    torch.save(best_model,path)

    print()
    print(f'generating reward plots...')
    plot_ep_rewards_vs_iterations(all_rewards,f"REINFORCE Value baseline : InvertedPendulum", f"plots/all_rewards_{hps['algo']}.png")

    print(f'total execution time = {((time.time()-start_time)/60.0):.3f} minutes')
    print_star()

def compute_gradient_estimte(policy,trajectories,gamma=0.99):
    policy.zero_grad()
    grads=[]
    batch_loss=0.0
    for traj in trajectories:
        states=torch.tensor(np.array(traj['states']),dtype=torch.float32)
        actions=torch.tensor(np.array(traj['actions']),dtype=torch.float32)
        rewards=traj['rewards']

        returns=[]
        G=0.0

        for r in reversed(rewards):
            G=r+gamma*G
            returns.insert(0,G)

        returns=torch.tensor(returns,dtype=torch.float32)
        # returns=(returns-returns.mean())/(returns.std()+1e-8)

        mean,std=policy(states)
        dist=Normal(mean,std)
        log_probs=dist.log_prob(actions).sum(dim=-1)

        batch_loss+=-(log_probs*returns).mean()

    batch_loss/=len(trajectories)
    batch_loss.backward()

    grad_vec=torch.cat([p.grad.view(-1) for p in policy.parameters()])
    return grad_vec.detach().cpu().numpy()

def get_500_traj(path,algo,sample_sizes,repetitions,hps):
    print()
    env=get_env()
    input_dim=hps['input_dim']
    output_dim=hps['output_dim']
    act_low=hps['action_low']
    act_high=hps['action_high']
    hidden_dim=hps['hidden_dim']

    policy=PolicyNetwork(input_dim,output_dim,hidden_dim).to(device)
    policy.load_state_dict(torch.load(path,map_location=device))
    policy.eval()

    trajectories_500=[]
    print(f'Collecting trajectories for {algo} model...')
    for ep in range(500):
        s,_=env.reset()
        done=False
        traj={'states':[],'actions':[],'rewards':[]}

        for t in range(1000):
            s_tensor=torch.tensor(s,dtype=torch.float32,device=device).unsqueeze(0)
            with torch.no_grad():
                mean,std=policy(s_tensor)
                dist=Normal(mean,std)
                action=dist.sample()
            action_np=action.detach().cpu().numpy()[0]
            a=np.clip(action_np,act_low,act_high)
            ns,r,te,tr,_=env.step(a)
            done=te or tr

            traj['states'].append(s)
            traj['actions'].append(a)
            traj['rewards'].append(float(r))
            s=ns
            if done: break
            
        trajectories_500.append(traj)
        if (ep+1)%100==0:
            print(f'-->Collected {ep+1}/{500} ')

    print()
    print("Running Gradient Estimation Experiments...")
    print_star()

    results={}

    mean_magnitudes=[]
    std_magnitudes=[]

    for n in sample_sizes:
        grad_estimates=[]
        for rep in range(repetitions):
            batch_traj=random.sample(trajectories_500,n)
            grad_est=compute_gradient_estimte(policy,batch_traj)
            grad_estimates.append(grad_est)
        grads = np.array(grad_estimates)
        grad_norms = np.linalg.norm(grads, axis=1)
        mean_magnitudes.append(np.mean(grad_norms))
        std_magnitudes.append(np.std(grad_norms))

    results['mean']=mean_magnitudes
    results['std']=std_magnitudes

    return results

def gradient_estimation_500(hps):
    print_star()
    st=time.time()
    print(f'Loading models and storing 500 trajectories of each model...')
    path_no_baseline="models/reinforce_no_baseline.pth"
    path_avg_baseline='models/reinforce_avg_reward_baseline.pth'
    path_reward_to_go='models/reinforce_reward_to_go.pth'
    path_value_baseline='models/reinforce_value_fn_baseline.pth'

    sample_sizes=[20,30,40,50,60,70,80,90,100]
    repetitions=10

    no_baseline_json=get_500_traj(path_no_baseline,'NO Baseline',sample_sizes,repetitions,hps)
    avg_baseline_json=get_500_traj(path_avg_baseline,"Average Baseline",sample_sizes,repetitions,hps)
    reward_to_go_json=get_500_traj(path_reward_to_go,'Reward to go',sample_sizes,repetitions,hps)
    value_baseline_json=get_500_traj(path_value_baseline,'Value Function',sample_sizes,repetitions,hps)

    os.makedirs("gradients", exist_ok=True)

    all_gradients = {
        "No Baseline": no_baseline_json,
        "Average Baseline": avg_baseline_json,
        "Reward to Go": reward_to_go_json,
        "Value Function": value_baseline_json
    }

    all_grad_data=make_json_serializable(all_gradients)

    with open("gradients/gradient_estimates.json", "w") as f:
        json.dump(all_grad_data, f, indent=4)

    print(f'Gradient estimate saved in json file...')
    print()
    print(f'total execution time = {((time.time()-st)/60.0):.3f} minutes')
    print_star()

def get_env():
    seed=42
    env = gym.make("InvertedPendulum-v4")
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

def main(ab_path):
    print_star()
    start_time=time.time()
    print('Training REINFORCE FOR Inverted Pendulum...')
    print_star()

    env=get_env()
    print(f'Network Details')
    input_dim=env.observation_space.shape[0]
    output_dim=env.action_space.shape[0]
    act_low=env.action_space.low
    act_high=env.action_space.high

    hps={
        'input_dim':input_dim,
        'output_dim':output_dim,
        'action_low':act_low,
        'action_high':act_high,
        'hidden_dim':64,
        'batch_size':5,
        'gamma':0.99,
        'lr':2e-3,
        'num_episodes':2000,
        'max_ep_length':1000,
        'algo':'no_baseline'
    }
    network_details(PolicyNetwork,input_dim,output_dim,hps['hidden_dim'])


    reinforce_no_baseline(env,hps) 

    hps['algo']='reward_to_go'
    reinforce_reward_to_go(env,hps)

    hps['algo']='avg_reward_baseline'
    reinforce_average_reward(env,hps)

    hps['algo']='value_fn_baseline'
    reinforce_value_baseline(env,hps)

    print(f'total execution time = {((time.time()-start_time)/60.0):.3f} minutes')
    print_star()
    return hps

def plot_gradient_estimate():
    with open("gradients/gradient_estimates.json", "r") as f:
        data = json.load(f)

    fig,axes=plt.subplots(2,2,figsize=(12,10))
    fig.suptitle("Gradient Estimate Magnitude vs Sample Size",fontsize=16,fontweight='bold')

    baselines=list(data.keys())
    ax_list=axes.flatten()
    sample_sizes=[20,30,40,50,60,70,80,90,100]

    for i,baseline in enumerate(baselines):
        ax=ax_list[i]
        mean_magnitudes=np.array(data[baseline]['mean'])
        std_magnitudes=np.array(data[baseline]['std'])

        ax.plot(sample_sizes,mean_magnitudes,label=f"{baseline}",linewidth=2)
        ax.fill_between(sample_sizes,mean_magnitudes - std_magnitudes,mean_magnitudes + std_magnitudes,alpha=0.3)

        ax.set_title(baseline,fontsize=13,fontweight='bold')
        ax.set_xlabel("Sample Size (No. of trajectories)")
        ax.set_ylabel("Gradient Magnitude")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
    plt.tight_layout(rect=[0,0.03,1,0.95])
    os.makedirs("plots",exist_ok=True)
    plt.savefig("plots/gradient_estimate_variance.png",dpi=300)
    # plt.show()
    print("Gradient variance plot saved at: plots/gradient_estimate_variance.png")

if __name__=="__main__":
    logging("REINFORCE")
    set_seed(42)
    absolute_path=os.getcwd()
    hps=main(absolute_path)
    sys.stdout = original_stdout; current_logger.close()
    logging("eval")
    gradient_estimation_500(hps)
    plot_gradient_estimate()
    sys.stdout = original_stdout; current_logger.close()


