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

# Define classes

class ModelLoader:
    @staticmethod
    def load_checkpoint(chk_path):
        with open(chk_path, 'rb') as f:
            chk = pickle.load(f)
        return chk

    @staticmethod
    def load_config(config_path):
        with open(config_path, 'rb') as f:
            cfg = pickle.load(f)
        return cfg


class DatasetProcessor:
    def __init__(self, env_name, max_episodes=1000):
        self.dataset, self.env = get_d4rl(env_name)
        self.max_episodes = max_episodes
        self.observations = []
        self.actions = []
        self.rewards = []
        self.terminals = []
        
    def process_episodes(self):
        print(len(self.dataset.episodes))
        for ep in self.dataset.episodes[:self.max_episodes]:
            self.observations.append(ep.observations)
            self.actions.append(ep.actions)
            self.rewards.append(ep.rewards)
            terminals = np.zeros(len(ep.rewards), dtype=bool)
            terminals[-1] = 1
            self.terminals.append(terminals)
        
        self.observations = np.concatenate(self.observations, axis=0)
        self.actions = np.concatenate(self.actions, axis=0)
        self.rewards = np.concatenate(self.rewards, axis=0)
        self.terminals = np.concatenate(self.terminals, axis=0)
        return MDPDataset(self.observations, self.actions, self.rewards, self.terminals)


class RewardEvaluator:
    def __init__(self, model, batch_size=128):
        self.model = model
        self.batch_size = batch_size

    def compute_rewards(self, dataset, cond_dim=23):
        state_actions = np.concatenate([dataset.observations, dataset.actions], axis=-1)
        state_actions = np.expand_dims(state_actions, axis=(1))
        state_actions_tensor = torch.tensor(state_actions, dtype=torch.float32)
        print(state_actions_tensor.shape)

        cond = torch.zeros((1, cond_dim), dtype=torch.float32)
        t = torch.zeros((1), dtype=torch.float32)
        
        rewards = []
        num_batches = (len(state_actions_tensor) + self.batch_size - 1) // self.batch_size
        for i in tqdm.tqdm(range(num_batches)):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, len(state_actions_tensor))
            state_action_batch = state_actions_tensor[start_idx:end_idx]
            cond_batch = cond.expand(len(state_action_batch), -1)
            t_batch = t.expand(len(state_action_batch))

            with torch.no_grad():
                reward_batch = self.model(state_action_batch, cond_batch, t_batch)
            rewards.extend(reward_batch.cpu().numpy())
        return np.array(rewards)

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


# Main Execution
model_config_path = '/home/shubham/diffusion-relative-rewards/locomotion/locomotion_diffusion_logs/logs/halfcheetah-expert-v2/gradient_matching/H4_Dmixed_DIMS16,16,32,32_ARCHmodels.ValueFunction_from_replay/model_config.pkl'
checkpoint_path = '/home/shubham/diffusion-relative-rewards/locomotion/locomotion_diffusion_logs/logs/halfcheetah-expert-v2/gradient_matching/H4_Dmixed_DIMS16,16,32,32_ARCHmodels.ValueFunction_from_replay/model_config.pkl'

# Load model and configuration
model_config = ModelLoader.load_config(model_config_path)
reward_model = model_config(23)
chk = ModelLoader.load_checkpoint(checkpoint_path)
reward_model.load_state_dict(chk, strict=False)
reward_model.eval()

# Process dataset
processor = DatasetProcessor('halfcheetah-medium-expert-v2')
dataset = processor.process_episodes()

# Compute rewards
reward_evaluator = RewardEvaluator(reward_model)
rewards = reward_evaluator.compute_rewards(dataset)
dataset = MDPDataset(dataset.observations, dataset.actions, rewards, dataset.terminals)

# Training
pipeline = TrainingPipeline(dataset, processor.env)
pipeline.train(500000, 10000)
pipeline.plot_metrics()