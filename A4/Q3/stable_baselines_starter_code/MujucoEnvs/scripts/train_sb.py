import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import numpy as np
from tqdm import tqdm
import cv2
import json
import imageio
import random
import gymnasium as gym
import config as exp_config
import utils.utils as utils
import utils.pytorch_util as ptu
from utils.logger import Logger

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

os.makedirs("logs",exist_ok=True)
os.makedirs("gifs",exist_ok=True)
os.makedirs("models",exist_ok=True)
os.makedirs("plots",exist_ok=True)
os.makedirs("evaluation",exist_ok=True)

def save_result_json(env_name, algo, reward, path="evaluation/results.json"):
    if not os.path.exists(path):
        data = {}
    else:
        with open(path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}

    if env_name not in data:
        data[env_name] = {}
    data[env_name][algo] = float(reward)

    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Saved result: {env_name} / {algo} = {reward}")


def make_env(env_name, seed):
    env = gym.make(env_name,render_mode="rgb_array")
    env = Monitor(env)
    env.reset(seed=seed)
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    return env

def setup_agent(args, config,seed):
    global agent, env,agent_algo,env_name
    env_name=args.env_name
    agent_algo = args.algo
    env = make_env(env_name,seed)

    if agent_algo=="A2C":
        model = A2C(
            "MlpPolicy",
            env,
            learning_rate=config["learning_rate"],
            n_steps=config["n_steps"],
            gamma=config["gamma"],
            gae_lambda=config["gae_lambda"],
            ent_coef=config["ent_coef"],
            vf_coef=config["vf_coef"],
            max_grad_norm=config["max_grad_norm"],
            verbose=1,
            tensorboard_log=f"logs/{env_name}/{agent_algo}/tb_seed{seed}"
        )

    elif agent_algo == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=config["learning_rate"],
            n_steps=config["n_steps"],
            batch_size=config["batch_size"],
            n_epochs=config["n_epochs"],
            gamma=config["gamma"],
            gae_lambda=config["gae_lambda"],
            clip_range=config["clip_range"],
            ent_coef=config["ent_coef"],
            vf_coef=config["vf_coef"],
            max_grad_norm=config["max_grad_norm"],
            verbose=1,
            tensorboard_log=f"logs/{env_name}/{agent_algo}/tb_seed{seed}"
        )
    if args.load_checkpoint is not None:
        model = type(model).load(args.load_checkpoint, env=env)

    return model, env

def train_agent(model, config, total_timesteps, seed=0):
    log_dir = f"logs/{env_name}/{agent_algo}/seed{seed}"
    os.makedirs(log_dir, exist_ok=True)

    eval_env = make_env(env_name, seed)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=5000,
        deterministic=True
    )

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save(f"{log_dir}/final_model")

    eval_env.close()
    env.close()
    
def test_agent(args, model, env):
    obs, _ = env.reset()
    rewards = []
    frames = []

    for k in tqdm(range(1000)):
        action, _ = model.predict(obs, deterministic=True)
        obs, rew, terminated, truncated, info = env.step(action)
        frame = env.render()
        rewards.append(rew)
        frames.append(frame)

        if terminated or truncated:
            break
    # save gif results in json
    gif_reward = sum(rewards)
    print(f"Test Reward: {gif_reward}")
    
    gif_path = f"gifs/{args.algo}_{args.env_name}.gif"
    imageio.mimsave(gif_path, frames, fps=30)
    print(f"Saved GIF: {gif_path}")
    save_result_json(args.env_name, args.algo, gif_reward)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--load_checkpoint', type=str,default=None)
    parser.add_argument('--timesteps',type=int,default=500000)
    parser.add_argument('--algo', type=str, default="A2C", choices=["A2C", "PPO"])
    parser.add_argument('--seed', type=int, default=3)

    args = parser.parse_args()
    
    configs = exp_config.configs[args.env_name][args.algo]["hyperparameters"]

    if args.test:
        best_reward=-np.inf
        best_seed=None
        best_path=None
        
        for seed in range(args.seed):
            seed_dir = f"logs/{args.env_name}/{args.algo}/seed{seed}"
            model_path = f"{seed_dir}/best_model.zip"
            eval_file = f"{seed_dir}/evaluations.npz"

            if os.path.exists(model_path) and os.path.exists(eval_file):
                data = np.load(eval_file)
                mean_reward = data["results"].mean()

                if mean_reward > best_reward:
                    best_reward = mean_reward
                    best_seed = seed
                    best_path = model_path

        if best_path is None:
            raise FileNotFoundError("No best_model.zip found in any seed folder.")

        print(f"Loading best model from seed {best_seed} (Reward: {best_reward:.2f})")

        args.load_checkpoint = best_path
        model, env = setup_agent(args, configs, seed=best_seed)
        test_agent(args, model, env)
    else:
        for seed in range(args.seed):
            model, env = setup_agent(args, configs,seed)
            train_agent(model, configs, args.timesteps,seed)