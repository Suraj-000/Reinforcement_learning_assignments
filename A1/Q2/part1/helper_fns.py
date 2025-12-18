import numpy as np
import matplotlib.pyplot as plt


def policy_evaluation_diff_seeds(P,seed):
    np.random.seed(seed)
    state=P.env._RESET()
    total_reward=0
    done=False
    current_state=state
    i=0
    arr_plot=[total_reward]
    while not done:
        i+=1
        action = P.get_action(current_state['state'],P.env.step_counter)
        next_state, reward, done, info = P.env._STEP(action)
        total_reward += reward
        arr_plot.append(total_reward)
        # print(f"itr {i} action={action} next state={next_state['state']} reward={reward}")
        current_state=next_state
    print(f"Episode finished. Total reward: {total_reward} for SEED={seed}")
    return arr_plot

def plot_heatmap(P,str):
    print('plotting heatmap')
    state_values=P.value_function[:,:,0]
    item_weights=P.env.item_weights
    item_values=P.env.item_values
    item_ratios=item_values/item_weights

    def helper(sort,xlabel):
        sorted_idx=np.argsort(sort)
        ordered_values=state_values[:,sorted_idx]

        plt.figure(figsize=(10,8))
        plt.imshow(ordered_values,aspect='auto',origin='lower',cmap='viridis')
        plt.colorbar(label='State Values')
        plt.xlabel(xlabel)
        plt.ylabel("Current weight (W)")
        plt.title('Value function heatmap')
        plt.savefig(f"plots/kpk_heatmp_{str}_{xlabel}.png")
        plt.show()
    
    helper(item_weights,'Weights')
    helper(item_values,'Values')
    helper(item_ratios,'Ratios')
