import imageio
import os
import numpy as np
import sys
import time
import matplotlib.pyplot as plt

def initialize_Q(env,num_actions):

    # zero initialization
    # value_function=np.zeros(num_states)
    # policy = np.zeros(num_states, dtype=int)
    # random initialization

    Q=np.zeros((env.height*env.width*2*2,num_actions))
    #terminal state
    # print(env.safe_goal,env.risky_goal)
    goal1=env._state_to_index(env.safe_goal[0],env.safe_goal[1],True,True)
    goal2=env._state_to_index(env.safe_goal[0],env.safe_goal[1],True,False)
    goal3=env._state_to_index(env.risky_goal[0],env.risky_goal[1],True,True)
    Q[goal1]=0
    Q[goal2]=0
    Q[goal3]=0
    return Q

def plot_ep_rewards_vs_iterations(episode_rewards,algo_name,path):
    plt.figure(figsize=(10,5))
    plt.plot(episode_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title(f'{algo_name} Episode Rewards ')
    plt.savefig(f'{path}')
    plt.show()


# get softmax action
def get_action_boltzman(s,Q, tau=1.0):
    action_values = Q[s]
    # For numerical stability subtract max
    preferences = action_values - np.max(action_values)  
    exp_preferences = np.exp(preferences / tau)
    probs = exp_preferences / np.sum(exp_preferences)
    return np.random.choice(np.arange(4), p=probs)

# get epsilon greedy action 
def get_action_epsilon(s,Q,epsilon):
    if np.random.rand()<epsilon:
        return np.random.randint(4)
    else:
        action_values=Q[s]
        max_q=np.max(action_values)
        max_action_values=np.flatnonzero(action_values==max_q)
        return np.random.choice(max_action_values)
    
def get_action_epsilon_frozen(s,Q,epsilon):
    if np.random.rand()<epsilon:
        return np.random.randint(3)
    else:
        action_values=Q[s]
        max_q=np.max(action_values)
        max_action_values=np.flatnonzero(action_values==max_q)
        return np.random.choice(max_action_values)

def eval_Q(env,bQ):
    policy=np.argmax(bQ,axis=1)
    s,*_=env.reset()
    done=False
    total_reward=0
    for i in range(1000):
        ns, r, done, _, _=env.step(policy[s])
        time.sleep(0.2)
        k=env.render()
        s=ns    
        total_reward+=r
        # print(f'iteration={i} next state = {ns}, reward={r}, action={policy[s]}, terminated={done}')
        if done: break
    env.close()
    print(f'Total eval reward={total_reward}')
    # print(type(k),k.shape)
    return total_reward

def render_gif(env,Q,filename="cliffwalk.gif"):
    policy=np.argmax(Q,axis=1)
    s,*_=env.reset()
    done=False
    total_reward=0
    frames = []
    for i in range(1000):
        ns, r, done, _, _=env.step(policy[s])
        total_reward+=r
        frame = env.render()
        frames.append(frame)
        s=ns    
        if done: break
    env.close()
    print(f'Total eval reward={total_reward}')
    imageio.mimsave(filename, frames, fps=5)
    print(f"Saved gif to {filename}")

def eval_Q_100(env,bQ):
    total_rewards=[]
    safe_visits, risky_visits = 0, 0
    for seed in range(100):
        s,*_=env.reset(seed=seed)
        done=False
        ep_reward=0
        for i in range(1000):
            a=np.argmax(bQ[s])
            ns, r, done, _, _=env.step(a)
            # time.sleep(0.2)
            # env.render()
            ep_reward+=r
            s=ns    
            if r == 40: safe_visits += 1
            elif r == 200: risky_visits += 1
            # print(f'iteration={i} next state = {ns}, reward={r}, action={policy[s]}, terminated={done}')
            if done: break
        total_rewards.append(ep_reward)
        env.close()
    rewards=np.array(total_rewards)
    print(f'Total eval reward={ep_reward}')
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "safe_visits": int(safe_visits),
        "risky_visits": int(risky_visits)
    }
def softmax(q_vals,tau=1.0):
    z=q_vals-np.max(q_vals)
    exp=np.exp(z/tau)
    return exp/np.sum(exp)

def plot_avg_goal_visits(results, save_dir="plots", filename="avg_goal_visits.png"):
    algos = list(results.keys())
    safe_counts = [results[algo]["safe"] for algo in algos]
    risky_counts = [results[algo]["risky"] for algo in algos]

    x = np.arange(len(algos))  # algo positions
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, safe_counts, width, label="Safe Goal")
    rects2 = ax.bar(x + width/2, risky_counts, width, label="Risky Goal")

    ax.set_ylabel("Average Visits (across 10 seeds)")
    ax.set_title("Average Safe vs Risky Goal Visits")
    ax.set_xticks(x)
    ax.set_xticklabels(algos)
    ax.legend()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points above
                        textcoords="offset points",
                        ha="center", va="bottom")
    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Saved plot: {filepath}")

def print_star(k=50):
    print('*'*k)