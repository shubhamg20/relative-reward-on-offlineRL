import numpy as np
from d3rlpy.dataset import MDPDataset
from d3rlpy.datasets import get_d4rl

# Load the dataset and environment
dataset, env = get_d4rl('halfcheetah-expert-v2')

# Specify the number of episodes you want to keep
max_episodes = 10

# Initialize lists to hold sliced data
observations, actions, rewards, terminals = [], [], [], []

# Iterate over the first `max_episodes` episodes
for ep in dataset.episodes[:max_episodes]:
    observations.append(ep.observations)
    actions.append(ep.actions)
    rewards.append(ep.rewards)
    # Set the last step in each episode to `1` (True) as the terminal flag
    terminals.append(np.zeros(len(ep.rewards), dtype=bool))
    terminals[-1][-1] = 1  # Mark the last step of each episode as terminal

# Concatenate the lists to create arrays
observations = np.concatenate(observations, axis=0)
actions = np.concatenate(actions, axis=0)
rewards = np.concatenate(rewards, axis=0)
terminals = np.concatenate(terminals, axis=0)

# Create a new MDPDataset with these concatenated arrays
dataset = MDPDataset(observations, actions, rewards, terminals)

# Check the dataset properties
print(len(dataset.episodes))          # Should print `max_episodes`
print(dataset.observations.shape)     # Should show the shape with the concatenated data