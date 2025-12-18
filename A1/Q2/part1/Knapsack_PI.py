import sys
import numpy as np
from or_gym.envs.classic_or.knapsack import OnlineKnapsackEnv
import matplotlib.pyplot as plt
from scipy.stats import mode
import time
from tqdm import tqdm
from helper_fns import policy_evaluation_diff_seeds,plot_heatmap
np.random.seed(42)


class PolicyIterationOnlineKnapsack:
    def __init__(self, env, time_steps,gamma=0.95, epsilon=1e-4, eval_iterations=100):
        self.env=env
        self.gamma=gamma
        self.epsilon=epsilon
        self.eval_iterations=eval_iterations

        self.num_items=self.env.N
        self.time_steps=time_steps
        self.max_weight=self.env.max_weight

        self.num_states=(self.num_items)*(self.time_steps)*(self.max_weight+1)
        self.num_actions=2
        
        # value function (current_weight, item_idx, time_step)
        self.value_function=np.zeros((self.max_weight+1,self.num_items,self.time_steps+1))
        self.policy=np.zeros((self.max_weight+1,self.num_items,self.time_steps+1),dtype=int)

    def get_reward_and_done(self, current_weight, item_idx, action):
        item_weight,item_value=self.env.item_weights[item_idx],self.env.item_values[item_idx]
        if action==0: return 0,False,current_weight
        if action==1:
            if current_weight+item_weight<=self.max_weight:
                return item_value, False, current_weight+item_weight
            else: return float('-inf'), True, current_weight
        
    def policy_evaluation(self):
        for itr in range(self.eval_iterations):
            delta=0
            new_value_function=self.value_function.copy()
            for time in reversed(range(self.time_steps)):
                next_time_step=self.value_function[:,:,time+1]

                for weight in range(self.max_weight+1):
                    values=next_time_step[weight,:]
                    actions=self.policy[weight,:,time]

                    weighted_value=self.gamma*np.dot(self.env.item_probs,values)
                    value_array=np.full(self.num_items,weighted_value)

                    next_weights=weight+self.env.item_weights
                    usefull_mask=next_weights<=self.max_weight
                    
                    usefull_next=next_time_step[next_weights[usefull_mask],:]
                    usefull_values=(self.env.item_values[usefull_mask]+self.gamma*np.sum(usefull_next*self.env.item_probs,axis=1))

                    next_values=np.full(self.num_items,-np.inf)
                    next_values[usefull_mask]=usefull_values

                    value_array[actions==1]=next_values[actions==1]

                    error=np.abs(self.value_function[weight,:,time]-value_array)
                    delta=max(delta,np.max(error))
                    new_value_function[weight,:,time]=value_array
            self.value_function=new_value_function
            if delta<self.epsilon:
                # print(f'policy evaluation converged in {itr+1} iterations ')
                break

    def policy_improvement(self):
        policy_stable=True
        for time in reversed(range(self.time_steps)):
            next_time_step=self.value_function[:,:,time+1]
            for weight in range(self.max_weight+1):
                values=next_time_step[weight,:]
                weighted_value=self.gamma*(np.dot(self.env.item_probs,values))

                next_weights=weight+self.env.item_weights
                usefull_mask=next_weights<=self.max_weight

                next_values=np.full(self.num_items,-np.inf)
                if np.any(usefull_mask):
                    usefull_next=next_time_step[next_weights[usefull_mask],:]
                    usefull_values=(self.env.item_values[usefull_mask]+self.gamma*np.sum(usefull_next*self.env.item_probs,axis=1))
                    next_values[usefull_mask]=usefull_values
                
                best_actions=(next_values>weighted_value).astype(int)
                if not np.array_equal(best_actions,self.policy[weight,:,time]):
                    policy_stable=False
                self.policy[weight,:,time]=best_actions
        # print('Updated policy')
        return policy_stable

    def run_policy_iteration(self, max_iterations=50):
        for i in tqdm(range(max_iterations),desc='Policy Iteration',unit="iter"):
            self.policy_evaluation()
            flag=self.policy_improvement()
            if flag:
                print(f'policy converged after {i+1} iterations')
                break
        # print(f'iteration {i+1} done')
        return self.policy,self.value_function

    def get_action(self, state,time):
        weight, item_index,_,_=state
        return self.policy[weight,item_index,time]

if __name__=="__main__":
    env=OnlineKnapsackEnv()
    state=env._RESET()

    '''
    Implements the online knapsack problem using policy iteration
    '''
    print('Online Knapsack Problem')
    print('Starting policy iteration training')
    pi=PolicyIterationOnlineKnapsack(env,time_steps=50)
    pi_policy,pi_value_function=pi.run_policy_iteration()
    print()
    '''
    Reports the value of the knapsack after training for five different seeds. 
    Plot the value of the knapsack as the items
    are presented during evaluation for the five seeds
    '''
    print(f'Policy evaluation for different seed values')
    plt.figure(figsize=(10,8))
    for i in range(5):
        arr_plot=policy_evaluation_diff_seeds(pi,seed=i)
        plt.plot(arr_plot,label=f'Seed = {i} Final Value = {arr_plot[-1]}')
    plt.xlabel('Time Step')
    plt.ylabel("Knapsack Value")
    plt.title('Online Knapsack Policy Iteration for different Seeds')
    plt.legend()
    plt.savefig(f"plots/knapsack_pi_seed_{i}.png")
    plt.show()

    '''
    plotting heatmaps.
    For a seed plot the heatmap of the final value function
    with value of each state as the intensity such that Current weight of knapsack is on the y-axis. On the
    x-axis plot weight, value and ratio of weight to value. Sort the values on the x-axis in increasing order
    before plotting.
    '''
    plot_heatmap(pi,'pi')
