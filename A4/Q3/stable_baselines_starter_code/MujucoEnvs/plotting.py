import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import numpy as np

env="hc"
algo="ppo"

data_path = f"plot_data/{env}_{algo}"

csv_files = sorted(glob.glob(os.path.join(data_path, "*.csv")))
print("Found files:", csv_files)

seed_values = []
steps = None

plt.figure(figsize=(14,8))

for file in csv_files:
    df = pd.read_csv(file)

    if steps is None:
        steps = df["Step"]

    reward = df["Value"]
    seed_values.append(reward)

    label = os.path.basename(file).replace(".csv","")
    plt.plot(steps, reward, alpha=0.7, label=f"{label}")

mean_values = np.mean(np.vstack(seed_values), axis=0)

plt.plot(steps, mean_values, linewidth=2, color="black", label="Mean over seeds")

plt.xlabel("Step")
plt.ylabel("Reward")
plt.title(f"Reward vs Step ({algo}, 3 Seeds)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"plots/{env}_{algo}.png")
plt.show()
