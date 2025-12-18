import numpy as np
import itertools
from collections import deque
import matplotlib.pyplot as plt
from scipy.stats import mode
import time
import sys
import random
from helper_fns import extract_from_trajectory,policy_simulate,print_star,plot,plot_episode

random.seed(42)       
np.random.seed(42) 
from or_gym.envs.finance.discrete_portfolio_opt import DiscretePortfolioOptEnv

class PolicyIterationPortfolio:
    def __init__(self, env, gamma=1.0,transaction_cost=1,cash_limit=100, epsilon=1e-6, eval_iterations=500):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.eval_iterations = eval_iterations
        
        self.initial_cash = self.env.initial_cash
        self.step_limit = self.env.step_limit
        self.holding_limit = self.env.holding_limit[0]

        self.cash_limit = cash_limit
        self.asset_prices = self.env.asset_prices[0]

        self.transaction_cost=transaction_cost

        self.value_function = np.zeros((self.cash_limit+1, self.holding_limit+1, self.step_limit+1))
        self.policy = np.zeros((self.cash_limit+1, self.holding_limit+1, self.step_limit), dtype=int)

    def get_asset_price(self,time):
        if time>=self.step_limit:return self.asset_prices[-1]
        return self.asset_prices[time]
    
    def get_qvalue(self,state,action):
        cash,holdings,time=state
        curr_price=self.get_asset_price(time)
        if time>=self.step_limit:return cash + holdings*curr_price

        next_cash=cash
        next_holdings=holdings
        reward=0

        if action != 0:
            if action>0:
                cost=action*(curr_price + self.transaction_cost)

                if cash>=cost:
                    next_cash=cash-cost
                    next_holdings=holdings+action
                else: return -np.inf
            else:
                asset_sell=abs(action)
                if holdings>=asset_sell:
                    cost=asset_sell*(curr_price - self.transaction_cost)
                    next_cash=cash+cost
                    next_holdings=holdings-asset_sell
                else: return -np.inf
        if next_cash<0 or next_cash>self.cash_limit or next_holdings<0 or next_holdings>self.holding_limit: return -np.inf
        next_t=time+1

        if next_t>=self.step_limit:
            next_price=self.get_asset_price(next_t)
            final_wealth=next_cash+next_holdings*next_price
            return reward+self.gamma*final_wealth
        else:
            next_state_value=self.value_function[next_cash,next_holdings,next_t]
            return reward+self.gamma*next_state_value

    def get_best_action(self,state):
        c,h,t=state
        all_actions = [-2, -1, 0, 1, 2] 
        best_action=0
        best_q_value=-np.inf

        for action in all_actions:
            q_val=self.get_qvalue(state,action)
            if q_val>best_q_value:
                best_q_value=q_val
                best_action=action
        return best_action,best_q_value

    def policy_evaluation(self):
        for itr in range(self.eval_iterations):
            delta = 0
            for time in reversed(range(self.step_limit+1)):
                for cash in range(self.cash_limit+1):
                    for holding in range(self.holding_limit+1):

                        if time>=self.step_limit: continue

                        state = (cash, holding, time)
                        action = self.policy[cash,holding,time]
                        cur_value=self.value_function[cash,holding,time]

                        self.value_function[cash,holding,time]=self.get_qvalue(state,action)
                        delta = max(delta, abs(cur_value - self.value_function[cash,holding,time]))
            if delta < self.epsilon:
                # print(f'policy eval converged after {itr+1} iterations')
                break
    
    def policy_improvement(self):
        policy_stable = True
        for time in reversed(range(self.step_limit)):
            for cash in range(self.cash_limit+1):
                for holding in range(self.holding_limit+1):
                    state = (cash, holding, time)
                    old_action = self.policy[cash, holding, time]

                    new_action,_=self.get_best_action(state)
                    self.policy[cash,holding,time]=new_action

                    if new_action!=old_action:policy_stable=False
        # print('policy updated')
        return policy_stable

    def run_policy_iteration(self, max_iterations=50):
        wealth_history = []
        for i in range(max_iterations):
            self.policy_evaluation()
            stable = self.policy_improvement()
            tra, final_wealth = policy_simulate(self,i)
            wealth_history.append(final_wealth)
            if stable:
                print(f"Policy converged after {i+1} iterations")
                break
        else:
            print(f"Reached max iterations ({max_iterations})")
        return self.policy, self.value_function,wealth_history,tra


if __name__=="__main__":

    prices=[
        [1, 3, 5, 5 , 4, 3, 2, 3, 5, 8],
        [4, 1, 4, 1 ,4, 4, 4, 1, 1, 4],
        [2, 2, 2, 4 ,2, 2, 4, 2, 2, 2]
    ]
    discount_factor=[0.999,1.0]
    transaction_cost=1
    print_star(100)
    print('Portfolio Optimization')
    '''
    QUESTION 1 
    part 1 of question: Train and Plot the total portfolio wealth at the end of each 
    episode as the training proceedsfor each discount factor.
    part 2 After training, plot the evolution of total wealth, 
    cash and the number of units of asset held across time steps for an episode.
    '''
    for i,price in enumerate(prices):
        for gamma in discount_factor:
            env = DiscretePortfolioOptEnv(prices=price)
            print(f' Asset prices={price} discount factor={gamma}')
            print('Starting Policy iteration training')
            P=PolicyIterationPortfolio(env,gamma,transaction_cost)
            policy,value_function,wealth_history,tra=P.run_policy_iteration()
            # plot(wealth_history,price,gamma,'PI',i)
            print(f'Wealth history {wealth_history}')
            # plot_episode(P,tra,price,gamma,'PI',i)
            print_star()
    
    '''
    part 3 report the execution time of the code for sequence prices[0] 
    and discount factor 0.999
    '''
    start_time=time.time()  
    env =DiscretePortfolioOptEnv(prices=prices[0])
    print(f' Asset prices={prices[0]} discount factor={discount_factor[0]}')
    print('Starting Policy iteration training')
    P=PolicyIterationPortfolio(env,discount_factor[0],transaction_cost)
    policy,value_function,wealth_history,tra=P.run_policy_iteration()
    print(f'Wealth history {wealth_history}')
    end_time=time.time()
    print(f"Execution time:{end_time-start_time:.4f} seconds")
