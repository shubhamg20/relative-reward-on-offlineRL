import sys, tqdm
import torch, pickle
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from d3rlpy.datasets import get_d4rl, MDPDataset
from d3rlpy.algos import IQL

# Adjust paths if necessary
sys.path.append('/home/shubham/diffusion-relative-rewards/locomotion')
sys.path.append('/home/shubham/diffusion-relative-rewards/d3rlpy/')

class TrainingPipeline:
    def __init__(self, dataset, env):
        self.dataset = dataset
        self.env = env
        self.iql = IQL(
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
        self.actor_losses = []
        self.critic_losses = []
        self.q_values_list = []
        self.cosine_similarities = []
        self.eval_total_rewards = []
        self.epoch = 1
        
    def evaluate(self, env, model, n_episodes=10):
        eval_total_rewards = []
        trajectories = []
        reward_arr_list = []
        for episode in range(n_episodes):
            obs = env.reset()
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

            reward_arr_list.append(reward_arr)
            eval_total_rewards.append(episode_reward)
            trajectories.append(np.array(episode_trajectory))
        print(f"Evaluated on {n_episodes} episodes: avg_rewd: {np.mean(eval_total_rewards)}")
        return np.mean(eval_total_rewards), trajectories, reward_arr_list
    
    def callback(self, iql, epoch, total_steps):
        self.iql = iql
        if self.epoch != epoch:
            self.epoch = epoch
            n_episodes=10
            avg_rewd, trajectories, rewards_arr= self.evaluate(self.env, iql, n_episodes)
            self.eval_total_rewards.append(avg_rewd)
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
            plt.savefig(f"org_general_IQ/eval_{epoch}.png")
            

    def train(self, n_steps=500000, n_steps_per_epoch=5000):
        sample_indices = np.random.choice(self.dataset.observations.shape[0], size=100, replace=False)
        states = np.array([self.dataset.observations[i] for i in sample_indices])
        actions = np.array([self.dataset.actions[i] for i in sample_indices])
        
        for epoch, metrics in self.iql.fit(self.dataset, n_steps=n_steps, n_steps_per_epoch=n_steps_per_epoch, callback=self.callback):
            self.actor_losses.append(metrics['actor_loss'])
            self.critic_losses.append(metrics['critic_loss'])

            q_values = self.iql.predict_value(states, actions)
            self.q_values_list.append(q_values)
            if epoch > 1:
                previous_q_values = self.q_values_list[-2]
                cosine_similarity = 1 - cosine(q_values.flatten(), previous_q_values.flatten())
                self.cosine_similarities.append(cosine_similarity)

    def plot_metrics(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.actor_losses, label="Actor Loss")
        plt.plot(self.critic_losses, label="Critic Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Actor and Critic Losses Over Time")
        plt.legend()
        plt.savefig("org_general_IQ/openmaze_actor_critic_losses.png")
        
        plt.figure(figsize=(10, 6))
        plt.hist(self.eval_total_rewards, bins=20, alpha=0.7, label="Reward Distribution")
        plt.xlabel("Reward")
        plt.ylabel("Frequency")
        plt.title("Reward Distribution Across Evaluation Episodes")
        plt.legend()
        plt.savefig("org_general_IQ/openmaze__reward_distribution.png")


        # Plot Average Reward over Epochs
        plt.figure(figsize=(10, 6))
        plt.plot(self.eval_total_rewards, label="Average Reward")
        plt.xlabel("Epoch")
        plt.ylabel("Average Reward")
        plt.title("Training Performance (Average Reward Over Epochs)")
        plt.legend()
        plt.savefig("org_general_IQ/openmaze_average_reward_over_epochs.png")


dataset, env = get_d4rl('maze2d-open-v0')
path = "/home/shubham/diffusion-relative-rewards/maze2d/required_datasets/general/maze2d-open-v0_general_pt_1.hdf5"
with h5py.File(path, 'r') as f:
    observations = f['observations'][:]
    actions = f['actions'][:]
    rewards = f['rewards'][:]
    terminals = f['terminals'][:]

dataset = MDPDataset(observations=observations, actions=actions, rewards=rewards, terminals=terminals)

print(len(dataset.episodes))
env.set_target(np.array([2, 4]))
print(f"Goal: {env.get_target()}")
# Training
pipeline = TrainingPipeline(dataset, env)
pipeline.train(500000, 10000)
pipeline.plot_metrics()
