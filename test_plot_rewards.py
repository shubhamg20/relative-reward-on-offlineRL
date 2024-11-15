import numpy as np
from d3rlpy.datasets import get_d4rl

import matplotlib.pyplot as plt

dataset, env = get_d4rl('maze2d-open-v0')

threshold = 0.85
count = 0

for episode in dataset.episodes:
    if np.max(episode.rewards) > threshold:
        count += 1

print(f"Number of episodes with maximum reward > {threshold}: {count}/{len(dataset.episodes)}")


plt.figure(figsize=(10, 6))
for episode in dataset.episodes[:100]:
    plt.plot(episode.rewards)

plt.xlabel('Timestep')
plt.ylabel('Reward')
plt.title('Rewards per Episode')
plt.legend()
plt.show()