import matplotlib.pyplot as plt
import numpy as np

def extract_from_trajectory(P, trajectory):
    cash_hist = []
    holdings_hist = []
    wealth_hist = []

    for (cash, holdings, t, action) in trajectory:
        price = P.get_asset_price(t)
        wealth = cash + holdings * price

        cash_hist.append(cash)
        holdings_hist.append(holdings)
        wealth_hist.append(wealth)

    return wealth_hist, cash_hist, holdings_hist


def policy_simulate(P,i=0):
    cash=P.initial_cash
    holdings=0
    time_step=0
    trajectory=[]
    while time_step<P.step_limit:
        if cash>P.cash_limit:    cash=P.cash_limit
        if holdings>P.holding_limit: holdings=P.holding_limit

        price=P.get_asset_price(time_step)
        curr_wealth=cash+holdings*price
        action=P.policy[cash,holdings,time_step]

        if action != 0:
            if action > 0:  # Buy
                cost = action *( price + P.transaction_cost)
                if cash >= cost:
                    cash -= cost
                    holdings += action
            else:  # Sell
                assets_to_sell = abs(action)
                if holdings >= assets_to_sell:
                    revenue = assets_to_sell * (price - P.transaction_cost)
                    cash += revenue
                    holdings -= assets_to_sell
        # print(f"itr {time_step} action={action}  reward={curr_wealth}")
        trajectory.append((int(cash),int(holdings),int(time_step),int(action)))
        time_step += 1

    final_price = P.get_asset_price(time_step)
    final_wealth = cash + holdings * final_price
    # print(f'itr {i+1} Total Wealth = {final_wealth}')
    return trajectory, int(final_wealth)
    
def print_star(n=60):
        print("="*n)

def plot(wealth_history,prices,gamma,str,i):
    plt.plot(range(1, len(wealth_history)+1), wealth_history, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Final Wealth")
    plt.title(f"{str}:Portfolio Wealth vs Value Itrs.\n Assets={prices} gamma={gamma}")
    plt.grid(True)
    plt.savefig(f'plots/{str}_{gamma}_{i}.png')
    plt.show()

def plot_episode(P,trajectory, price, gamma,str,i):
    wealth,cash,holding=extract_from_trajectory(P,trajectory)
    plt.figure(figsize=(12, 6))

    plt.plot(wealth, label="Total Wealth", marker='o')
    plt.plot(cash, label="Cash", marker='s')
    plt.plot(holding, label="Units Held", marker='^')

    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.title(f"{str} : Evolution of Wealth, Cash, and Holdings\nAsset={price}, γ={gamma}")
    plt.grid(True)
    plt.legend()
    plt.savefig(f'plots/Evolution_{str}_{gamma}_{i}.png')
    plt.show()
