import numpy as np
import matplotlib.pyplot as plt
from d3rlpy.datasets import get_d4rl

# Load the dataset and environment
dataset, env = get_d4rl('maze2d-open-v0')

# Print dataset information
print(f"Number of episodes: {len(dataset.episodes)}")
print(f"Observations shape: {dataset.observations.shape}")

# Extract initial points and goal points
initial_points = []
goal_points = []

for episode in dataset.episodes:
    observations = episode.observations
    initial_points.append(observations[0])
    goal_points.append(observations[-1])

initial_points = np.array(initial_points)
goal_points = np.array(goal_points)
print(initial_points.shape)
print(goal_points.shape)
# Plot initial points and goal points
plt.figure(figsize=(10, 6))
plt.scatter(initial_points[:, 0], initial_points[:, 1], color='red', marker='x', s=100, label='Start')
plt.scatter(goal_points[:, 0], goal_points[:, 1], color='blue', marker='*', s=100, label='Goal')
print(initial_points[:10, 0])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Initial Points and Goal Points')
plt.legend()
plt.show()