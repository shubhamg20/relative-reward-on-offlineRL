import sys, tqdm
sys.path.append('/home/shubham/diffusion-relative-rewards/locomotion')
sys.path.append('/home/shubham/diffusion-relative-rewards/d3rlpy/')
import numpy as np
import matplotlib.pyplot as plt
from d3rlpy.algos import IQL
from d3rlpy.datasets import get_d4rl

def evaluate(env, model, goal=None, n_episodes=10):
    eval_total_rewards = []
    trajectories = []
    reward_arr_list = []
    for episode in range(n_episodes):
        obs = env.reset()
        if goal is not None:
            env.set_target(goal)  # Set the goal in the environment
        done = False
        episode_reward = 0
        episode_trajectory = []
        reward_arr = []
        while not done:
            action = model.predict([obs])[0]
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            reward_arr.append(reward)
            episode_trajectory.append(obs)
            # Check if the goal is reached
            if goal is not None and np.linalg.norm(obs[:2] - goal) < 1e-2:  # Example threshold
                done = True
        reward_arr_list.append(reward_arr)
        eval_total_rewards.append(episode_reward)
        trajectories.append(np.array(episode_trajectory))
    print(f"Evaluated on {n_episodes} episodes: avg_rewd: {np.mean(eval_total_rewards)}")
    return np.mean(eval_total_rewards), trajectories, reward_arr_list

iql = IQL(
    actor_learning_rate=3e-4,
    critic_learning_rate=3e-4,
    value_learning_rate=3e-4,
    batch_size=256,
    actor_update_frequency=1,
    critic_update_frequency=1, 
    value_update_frequency=1,
    target_update_interval=2000,
    n_critics=2,
    discount=0.99,
)

dir = 'd3rlpy_logs/IQL_20241115025509/model_'   #diff
dataset, env = get_d4rl('maze2d-open-v0')
iql.build_with_env(env)

goal = np.array([1, 1])
n_episodes = 10

path = dir + str(50000) + '.pt'
iql.load_model(path)
rewards, trajectories, rewards_arr = evaluate(env, iql, goal, n_episodes)
print("A Iteration: ", 50000, "Avg_rewards: ", rewards)

# Extract initial and goal points
initial_points = [trajectory[0] for trajectory in trajectories]
goal_points = [trajectory[-1] for trajectory in trajectories]

initial_points = np.array(initial_points)
goal_points = np.array(goal_points)

# Create a single figure with multiple subplots
plt.figure(figsize=(18, 12))

# Subplot 1: Trajectories of the first 5 episodes
plt.subplot(3, 1, 1)
for i, trajectory in enumerate(trajectories[:5]):  # Plot first 5 trajectories
    plt.plot(trajectory[:, 0], trajectory[:, 1], label=f"Episode {i+1}")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Trajectories of First 5 Episodes")
plt.legend()

# Subplot 2: Reward function for each trajectory
plt.subplot(3, 1, 2)
for i, rewards in enumerate(rewards_arr[:n_episodes]):
    plt.plot(rewards, label=f"Episode {i+1}")
plt.xlabel('Step')
plt.ylabel('Reward')
plt.title('Reward Function for Each Trajectory')
plt.legend()

# Subplot 3: Initial and goal points
plt.subplot(3, 1, 3)
plt.scatter(initial_points[:, 0], initial_points[:, 1], color='red', marker='x', s=100, label='Start')
plt.scatter(goal_points[:, 0], goal_points[:, 1], color='blue', marker='*', s=100, label='Goal')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Initial and Goal Points')
plt.legend()

plt.tight_layout()
plt.show()