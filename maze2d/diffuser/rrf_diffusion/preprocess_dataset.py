import sys
sys.path.append("/home/shubham/diffusion-relative-rewards/maze2d/")
import diffuser.utils as utils
import argparse
import pathlib
import os 

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type = str, default = 'maze2d-open-v0', help = 'Dataset path, e.g. maze2d-umaze-v1')
parser.add_argument('--dataset_size', type = str, default = '10k', help = 'Dataset size, e.g. 10m, 10k, 1m')
parser.add_argument('--goal', type = str, default = 'topleft', help = 'Name of goal, e.g. topleft, general')
parser.add_argument('--version', type = str, default = 'v1', help = 'Version being used, e.g. v1, v2')
parser.add_argument('--horizon', type = int, default = 256, help = 'Horizon for the diffusion model. Needs to be a power of 2')
parser.add_argument('--max_path_length', type = int, default = 1000, help = 'Maximum path size. Preprocessed buffer will be n_paths x max_path_length x channels.')
parser.add_argument('--datadir', type = str, default = '/home/shubham/.d4rl/datasets', help = "Directory containing folder with data for experiment.")
parser.add_argument('--savedir', type = str, default = '/home/shubham/diffusion-relative-rewards/maze2d', help = "Directory in which to save the buffer.")
args = parser.parse_args()

pathlib.Path(os.path.abspath(args.savedir)).mkdir(exist_ok = True, parents = True)
save_path = f"{args.savedir}/{args.dataset}_{args.dataset_size}_{args.version}_{args.goal}.pkl"
h5path = f"{args.datadir}/{args.dataset}.hdf5"
new_dataset_config = utils.Config(
    'datasets.GoalDataset',
    savepath=('/tmp', 'dataset_config.pkl'),
    env=args.dataset,
    horizon=args.horizon,
    normalizer='LimitsNormalizer',
    preprocess_fns=['maze2d_set_terminals'],
    use_padding=False,
    max_path_length=args.max_path_length,
    max_n_episodes=300000,
    h5path=h5path,
)
print(h5path)
new_dataset = new_dataset_config()
new_dataset.clean_fields(for_saving = True)
new_dataset.fields.save(save_path)
