import h5py
import numpy as np
import matplotlib.pyplot as plt

def plot_episode(observations, ax):
    x = observations[:, 0]
    y = observations[:, 1]
    ax.plot(x, y, marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

def main():
    dataset_path = '/home/shubham/diffusion-relative-rewards/maze2d/d4rl_datasets/maze2d-open-v0.hdf5'  #goal1
    with h5py.File(dataset_path, 'r') as f:
        observations = f['observations'][:]
        terminals = f['terminals'][:]
    
    episodes = []
    episode = []
    for i in range(len(observations)):
        episode.append(observations[i])
        if terminals[i]:
            episodes.append(np.array(episode))
            episode = []

    print(f"Number of episodes: {len(episodes)}")
    print(f"Observations shape: {len(observations)}")
    num_episodes_to_plot = 100
    num_cols = 10
    num_rows = (num_episodes_to_plot + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 2))
    axes = axes.flatten()

    for i, episode in enumerate(episodes[500:500+num_episodes_to_plot]):
        plot_episode(episode, axes[i])
        axes[i].set_title(f"Episode {i+1}")

    for i in range(num_episodes_to_plot, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()