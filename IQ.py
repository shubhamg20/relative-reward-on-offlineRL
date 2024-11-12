
import sys, tqdm
sys.path.append('/home/shubham/diffusion-relative-rewards/locomotion')
import diffuser
import torch, pickle
import gym
import numpy as np
from typing import List
import diffuser.utils as utils
import d3rlpy, gym
from d3rlpy.datasets import get_d4rl
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from gym import RewardWrapper
from d3rlpy.dataset import MDPDataset

def load_checkpoint(chk_path):
    with open(chk_path, 'rb') as f:
        chk = pickle.load(f)
    return chk

def load_config(config_path):
    with open(config_path, 'rb') as f:
        cfg = pickle.load(f)
    return cfg

def evaluate(env, iql_model, n_episodes=10):
    total_rewards = []
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = iql_model.predict([obs])[0]  # Predict action for the current state
            obs, reward, done, _ = env.step(action)  # Step the environment
            episode_reward += reward  # Accumulate reward
        total_rewards.append(episode_reward)
    print(f"Evaluated on {n_episodes}: avg_rewd: {np.mean(total_rewards)}")
    return np.mean(total_rewards)  # Return average reward over episodes

def epoch_callback(algo, epoch, total_step):
    # epoch_rewards.append(metrics['average_value'])
    # print(f"Epoch {epoch}: {metrics}")
    pass

model_config_path = '/home/shubham/diffusion-relative-rewards/locomotion/locomotion_diffusion_logs/logs/halfcheetah-expert-v2/gradient_matching/H4_Dmixed_DIMS16,16,32,32_ARCHmodels.ValueFunction_from_replay/model_config.pkl'
checkpoint_path = '/home/shubham/diffusion-relative-rewards/locomotion/locomotion_diffusion_logs/logs/halfcheetah-expert-v2/gradient_matching/H4_Dmixed_DIMS16,16,32,32_ARCHmodels.ValueFunction_from_replay/model_config.pkl'
model_config = load_config(model_config_path)
reward_model = model_config(23)

chk = load_checkpoint(checkpoint_path)
reward_model.load_state_dict(chk, strict=False)
reward_model.eval()  
print(reward_model)
    
# Get dataset and environment
dataset, env = get_d4rl('halfcheetah-expert-v2')
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

print(len(dataset.episodes))          
print(dataset.observations.shape)
    
# Initialize cond and t tensors
cond = torch.zeros((1, 23), dtype=torch.float32)
t = torch.zeros((1), dtype=torch.float32)

# Concatenate observations and actions
state_actions = np.concatenate([dataset.observations, dataset.actions], axis=-1)
# Convert state_actions to a tensor
state_actions = np.expand_dims(state_actions, axis=(1))
state_actions_tensor = torch.tensor(state_actions, dtype=torch.float32)


# Batch size
batch_size = 128
num_batches = (len(state_actions_tensor) + batch_size - 1) // batch_size

# Process in batches
rewards = []
print(num_batches)
for i in tqdm.tqdm(range(num_batches)):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, len(state_actions_tensor))
    
    state_action_batch = state_actions_tensor[start_idx:end_idx]
    cond_batch = cond.expand(len(state_action_batch), -1)
    t_batch = t.expand(len(state_action_batch))
    
    with torch.no_grad():
        reward_batch = reward_model(state_action_batch, cond_batch, t_batch)
    rewards.extend(reward_batch.cpu().numpy())
rewards = np.array(rewards)
dataset = MDPDataset(dataset.observations, dataset.actions, rewards, dataset.terminals)
print(len(dataset.episodes))          
print(dataset.observations.shape)

# Initialize the Implicit Q-Learning (IQL) algorithm
iql = d3rlpy.algos.IQL(
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

# # Prepare the dataset and environment
# iql.build_with_dataset(dataset)

# Metrics storage
epoch_rewards = []
actor_losses = []
critic_losses = []
q_values_list = []
cosine_similarities = []

# Select a fixed set of random state-action pairs for stability testing
sample_indices = np.random.choice(dataset.observations.shape[0], size=100, replace=False)
states = np.array([dataset.observations[i] for i in sample_indices])
actions = np.array([dataset.actions[i] for i in sample_indices])

# Training and logging loop
for epoch, metrics in iql.fit(dataset, n_steps=20, n_steps_per_epoch=10, epoch_callback=epoch_callback): 
    actor_losses.append(metrics['actor_loss'])
    critic_losses.append(metrics['critic_loss'])
    
    q_values = iql.predict_value(states, actions)
    q_values_list.append(q_values)
    if epoch-1> 0:
        previous_q_values = q_values_list[-2]
        cosine_similarity = 1 - cosine(q_values.flatten(), previous_q_values.flatten())
        cosine_similarities.append(cosine_similarity)

# Plot Actor and Critic Losses
plt.figure(figsize=(10, 6))
plt.plot(actor_losses, label="Actor Loss")
plt.plot(critic_losses, label="Critic Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Actor and Critic Losses Over Time")
plt.legend()
plt.savefig("actor_critic_losses.png")

# Plot Cosine Similarity of Q-Values Across Epochs
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cosine_similarities) + 1), cosine_similarities, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Cosine Similarity")
plt.title("Cosine Similarity Between Q-Values Across Epochs")
plt.savefig("cosine_similarity_q_values.png")


# Reward Distribution Plot
plt.figure(figsize=(10, 6))
plt.hist(epoch_rewards, bins=20, alpha=0.7, label="Reward Distribution")
plt.xlabel("Reward")
plt.ylabel("Frequency")
plt.title("Reward Distribution Across Evaluation Episodes")
plt.legend()
plt.savefig("reward_distribution.png")


# Plot Average Reward over Epochs
plt.figure(figsize=(10, 6))
plt.plot(epoch_rewards, label="Average Reward")
plt.xlabel("Epoch")
plt.ylabel("Average Reward")
plt.title("Training Performance (Average Reward Over Epochs)")
plt.legend()
plt.savefig("average_reward_over_epochs.png")



