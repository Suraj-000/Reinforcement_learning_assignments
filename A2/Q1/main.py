from helper_fns import initialize_Q, plot_ep_rewards_vs_iterations,get_action_boltzman,get_action_epsilon,eval_Q,softmax,plot_avg_goal_visits,eval_Q_100,print_star,render_gif,get_action_epsilon_frozen
from cliff import MultiGoalCliffWalkingEnv
from frozenlake import DiagonalFrozenLake
import numpy as np
import random
import time
import json
from agent import SARSA
from agent import expected_SARSA
from agent import q_learning_for_cliff
from agent import monte_carlo
from agent import q_learning_for_frozenlake
import matplotlib.pyplot as plt


print(f'Implenenting Cliff Walking environment ...')

env=MultiGoalCliffWalkingEnv(render_mode='rgb_array')
print_star()

Q_sarsa, episode_rewards_sarsa, avg_safe_sarsa, avg_risky_sarsa=SARSA(env)
print(f'SARSA: avg reward={np.mean(episode_rewards_sarsa)} safe visits={avg_safe_sarsa}, risky visits={avg_risky_sarsa}')
path='plots/sarsa_plot.png'
plot_ep_rewards_vs_iterations(episode_rewards_sarsa,'SARSA',path)
eval_Q(env,Q_sarsa)
print_star()

Q_ql, episode_rewards_ql, avg_safe_q, avg_risky_q=q_learning_for_cliff(env)
print(f'Q-learning: avg reward={np.mean(episode_rewards_ql)} safe visits={avg_safe_q}, risky visits={avg_risky_q}')
path='plots/qlearning_plot.png'
plot_ep_rewards_vs_iterations(episode_rewards_ql,'Q-learning',path)
eval_Q(env,Q_ql)
print_star()

Q_exp, episode_rewards_exp, avg_safe_exp,  avg_risky_exp=expected_SARSA(env)
print(f'Expected SARSA: avg reward={np.mean(episode_rewards_exp)} safe visits={avg_safe_exp}, risky visits={avg_risky_exp}')
path='plots/expected_sarsa_plot.png'
plot_ep_rewards_vs_iterations(episode_rewards_exp,'exp-SARSA',path)
eval_Q(env,Q_exp)
print_star()

results={
    "SARSA": {"safe": avg_safe_sarsa, "risky": avg_risky_sarsa},
    "Q-learning": {"safe": avg_safe_q, "risky": avg_risky_q},
    "Expected SARSA": {"safe": avg_safe_exp, "risky": avg_risky_exp}
}
print("Plotting average goal visits ...")
plot_avg_goal_visits(results,filename='average_goal_visits.png')
print_star()

print("Evaluating learned policies over 100 episodes ...")

sarsa=eval_Q_100(env,Q_sarsa)
print(f'SARSA:{sarsa}')
ql=eval_Q_100(env,Q_ql)
print(f'Q-learning: {ql}')
exp_sarsa=eval_Q_100(env,Q_exp)
print(f'Expected sarsa: {exp_sarsa}')

json_data={
    "Sarsa": {
        "mean": sarsa["mean_reward"], 
        "std": sarsa["std_reward"]
    },
    "QLearning": {
        "mean": ql["mean_reward"], 
        "std": ql["std_reward"]
    },
    "ExpectedSarsa": {
        "mean": exp_sarsa["mean_reward"], 
        "std": exp_sarsa["std_reward"]
    }
}
json_path='evaluation/cliff_evaluation_results.json'
with open(json_path, "w") as json_file:
    json.dump(json_data, json_file)

print(f"Saved evaluation results to {json_path}")   
print_star()

print("generating gifs ...  ")
render_gif(env,Q_sarsa,filename="gifs/sarsa.gif")
render_gif(env,Q_ql,filename="gifs/qlearning.gif")
render_gif(env,Q_exp,filename="gifs/expected_sarsa.gif")
print_star()


print_star(100)
print("Implementing Frozen Lake environment ...")
print('training for start state (0,5) ...')
env=DiagonalFrozenLake(render_mode='rgb_array')
env.reset(seed=42)
Q_mc_5, episode_rewards_mc_5,*_=monte_carlo(env)
print(f'Monte Carlo: avg reward={np.mean(episode_rewards_mc_5)}')
path='plots/frozenlake_mc_(0,5).png'
plot_ep_rewards_vs_iterations(episode_rewards_mc_5,'Monte Carlo',path)
eval_Q(env,Q_mc_5)
print_star()

Q_ql_5, episode_rewards_ql_5,*_=q_learning_for_frozenlake(env)
print(f'Q-learning: avg reward={np.mean(episode_rewards_ql_5)}')
path='plots/frozenlake_qlearning_(0,5).png'
plot_ep_rewards_vs_iterations(episode_rewards_ql_5,'Q-learning',path)
eval_Q(env,Q_ql_5)
print_star()


env=DiagonalFrozenLake(render_mode='rgb_array',start_state=(0,3))
print('training for start state (0,3) ...')
Q_mc_3, episode_rewards_mc_3,*_=monte_carlo(env)
print(f'Monte Carlo: avg reward={np.mean(episode_rewards_mc_3)}')
path='plots/frozenlake_mc_(0,3).png'
plot_ep_rewards_vs_iterations(episode_rewards_mc_3,'Monte Carlo',path)
eval_Q(env,Q_mc_3)
print_star()

Q_ql_3, episode_rewards_ql_3,*_=q_learning_for_frozenlake(env)
print(f'Q-learning: avg reward={np.mean(episode_rewards_ql_3)}')
path='plots/frozenlake_qlearning_(0,3).png'
plot_ep_rewards_vs_iterations(episode_rewards_ql_3,'Q-learning',path)
eval_Q(env,Q_ql_3)
print_star()



print("Evaluating learned policies over 100 episodes ...")

Qmc5=eval_Q_100(env,Q_mc_5)
print(f'Monte carlo(0,5):{Qmc5}')
ql5=eval_Q_100(env,Q_ql_5)
print(f'Q-learning(0,5): {ql5}')
Qmc3=eval_Q_100(env,Q_mc_3)
print(f'Monte carlo(0,3):{Qmc3}')
ql3=eval_Q_100(env,Q_ql_3)
print(f'Q-learning(0,3): {ql3}')

json_data={
    "QLearning(0, 3)": {
        "mean": ql3["mean_reward"], 
        "std": ql3["std_reward"]
    },
    "MonteCarloOnPolicy(0, 3)": {
        "mean": Qmc3["mean_reward"], 
        "std": Qmc3["std_reward"]
    },
    "QLearning(0, 5)": {
        "mean": ql5["mean_reward"], 
        "std": ql5["std_reward"]
    },
    "MonteCarloOnPolicy(0, 5)": {
        "mean": Qmc5["mean_reward"], 
        "std":  Qmc5["std_reward"]
    }
}
json_path='evaluation/frozenlake_variant_evaluation_results.json'
with open(json_path, "w") as json_file:
    json.dump(json_data, json_file)

print(f"Saved evaluation results to {json_path}")   
print_star()




env=DiagonalFrozenLake(render_mode='rgb_array')
print("generating gifs ...  ")
render_gif(env,Q_mc_5,filename="gifs/frozenlake_mc_(0,5).gif")
render_gif(env,Q_ql_5,filename="gifs/frozenlake_qlearning_(0,5).gif")

env=DiagonalFrozenLake(render_mode='rgb_array',start_state=(0,3))
render_gif(env,Q_mc_3,filename="gifs/frozenlake_mc_(0,3).gif")
render_gif(env,Q_ql_3,filename="gifs/frozenlake_qlearning_(0,3).gif")