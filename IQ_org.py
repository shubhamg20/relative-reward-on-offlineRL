import sys, tqdm
import torch, pickle
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
        for episode in range(n_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            while not done:
                action = model.predict([obs])[0]
                obs, reward, done, _ = env.step(action)
                episode_reward += reward
            self.eval_total_rewards.append(episode_reward)
        print(f"Evaluated on {n_episodes} episodes: avg_rewd: {np.mean(self.eval_total_rewards)}")
        return np.mean(self.eval_total_rewards)
        
    def callback(self, iql, epoch, total_steps):
        self.iql = iql
        if self.epoch != epoch:
            self.epoch = epoch
            avg_rewd = self.evaluate(self.env, iql, n_episodes=10)
            self.eval_total_rewards.append(avg_rewd)

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
        plt.savefig("actor_critic_losses.png")

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.cosine_similarities) + 1), self.cosine_similarities, marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Cosine Similarity")
        plt.title("Cosine Similarity Between Q-Values Across Epochs")
        plt.savefig("cosine_similarity_q_values.png")
        
        plt.figure(figsize=(10, 6))
        plt.hist(self.eval_total_rewards, bins=20, alpha=0.7, label="Reward Distribution")
        plt.xlabel("Reward")
        plt.ylabel("Frequency")
        plt.title("Reward Distribution Across Evaluation Episodes")
        plt.legend()
        plt.savefig("reward_distribution.png")


        # Plot Average Reward over Epochs
        plt.figure(figsize=(10, 6))
        plt.plot(self.eval_total_rewards, label="Average Reward")
        plt.xlabel("Epoch")
        plt.ylabel("Average Reward")
        plt.title("Training Performance (Average Reward Over Epochs)")
        plt.legend()
        plt.savefig("average_reward_over_epochs.png")


dataset, env = get_d4rl('halfcheetah-expert-v2')
# Training
pipeline = TrainingPipeline(dataset, env)
pipeline.train(500000, 10000)
pipeline.plot_metrics()
