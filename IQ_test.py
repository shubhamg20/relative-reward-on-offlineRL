import sys, tqdm
sys.path.append('/home/shubham/diffusion-relative-rewards/locomotion')
sys.path.append('/home/shubham/diffusion-relative-rewards/d3rlpy/')
import numpy as np
import matplotlib.pyplot as plt
from d3rlpy.algos import IQL
from d3rlpy.datasets import get_d4rl

def evaluate(env, model, n_episodes=10):
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
        # print(reward_arr)
        # print("\n")
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
# dir = '/home/shubham/diffusion-relative-rewards/d3rlpy_logs/IQL_20241113090959/model_' #org
dataset, env = get_d4rl('maze2d-open-v0')
iql.build_with_env(env)

eval_total_rewards = []
all_trajectories = []
rewards_arr = []
for num in [50000]:
    path = dir + str(num) + '.pt'
    iql.load_model(path)
    rewards, trajectories, rewards_arr = evaluate(env, iql, n_episodes=20)
    rewards_arr.append(rewards)
    print("A Iteration: ", num, "Avg_rewards: ", rewards)
    eval_total_rewards.append(rewards)
    all_trajectories.extend(trajectories)

# plt.figure(figsize=(10, 6))
# plt.plot(eval_total_rewards, label="Average Reward")
# plt.xlabel("Epoch")
# plt.ylabel("Average Reward")
# plt.title("Training Performance (Average Reward Over Epochs)")
# plt.legend()
# plt.savefig("average_reward_over_epochs.png")

# Plot trajectories
plt.figure(figsize=(10, 6))
for i, trajectory in enumerate(all_trajectories[:5]):  # Plot first 5 trajectories
    plt.plot(trajectory[:, 0], trajectory[:, 1], label=f"Episode {i+1}")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Trajectories of First 5 Episodes")
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
for i, rewards in enumerate(rewards_arr[:10]):
    plt.plot(rewards)

plt.xlabel('Step')
plt.ylabel('Reward')
plt.title('Reward Function for Each Trajectory')
plt.legend()
plt.show()