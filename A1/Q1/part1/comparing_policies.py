from policy_iteration import policy_iteration
from value_iteration import value_iteration

discount_factor = 0.95
threshold = 1e-6
num_iterations = 1000
seed=42


print(f'Starting Policy iteration Gamma={discount_factor}')
pi_policy,pi_value_function,nof_iterations=policy_iteration(get_gif=False)

print(f'Starting Value iteration Gamma={discount_factor}')
vi_policy,vi_value_function,nof_iterations2=value_iteration(get_gif=False)


print(f'Comparing policies')
flag=1
for i in range(len(vi_policy)): 
    if vi_policy[i]!=pi_policy[i] and vi_value_function[i]==pi_value_function[i]:
        flag=0

if flag: print('Both learned policies are identicial')