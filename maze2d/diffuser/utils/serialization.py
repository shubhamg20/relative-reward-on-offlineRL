import os
import pickle
import glob
import torch
import pdb
from .config import Config

from collections import namedtuple

DiffusionExperiment = namedtuple('Diffusion', 'dataset renderer model diffusion ema trainer epoch')

def mkdir(savepath):
    """
        returns `True` iff `savepath` is created
    """
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        return True
    else:
        return False

def get_latest_epoch(loadpath):
    states = glob.glob1(os.path.join(*loadpath), 'state_*')
    latest_epoch = -1
    for state in states:
        epoch = int(state.replace('state_', '').replace('.pt', ''))
        latest_epoch = max(epoch, latest_epoch)
    return latest_epoch

def load_config(*loadpath):
    loadpath = os.path.join(*loadpath)
    config = pickle.load(open(loadpath, 'rb'))
    print(f'[ utils/serialization ] Loaded config from {loadpath}')
    print(config)
    return config

def load_diffusion(*loadpath, fields_load_path = None, epoch='latest', device='cuda:0'):
    dataset_config = load_config(*loadpath, 'dataset_config.pkl')
    render_config = load_config(*loadpath, 'render_config.pkl')
    model_config = load_config(*loadpath, 'model_config.pkl')
    diffusion_config = load_config(*loadpath, 'diffusion_config.pkl')
    trainer_config = load_config(*loadpath, 'trainer_config.pkl')

    ## remove absolute path for results loaded from azure
    ## @TODO : remove results folder from within trainer class
    trainer_config._dict['results_folder'] = os.path.join(*loadpath)
    dataset_config._dict['fields_load_path'] = fields_load_path

    dataset = dataset_config()
    renderer = render_config()
    model = model_config()
    diffusion = diffusion_config(model)
    trainer = trainer_config(diffusion, dataset, renderer)

    if epoch == 'latest':
        epoch = get_latest_epoch(loadpath)

    print(f'\n[ utils/serialization ] Loading model epoch: {epoch}\n')

    trainer.load(epoch)

    return DiffusionExperiment(dataset, renderer, model, diffusion, trainer.ema_model, trainer, epoch)

def load_gradient_matching(
        *loadpath, 
        dataset = None,
        diffusion = None,
        epoch='latest', 
        device='cuda:0', 
        seed=None
):
    render_config = load_config(*loadpath, 'render_config.pkl')
    model_config = load_config(*loadpath, 'model_config.pkl')
    trainer_config = load_config(*loadpath, 'trainer_config.pkl')
    # gradient_matching_config = load_config(*loadpath, 'gradient_matching_config.pkl')

    ## remove absolute path for results loaded from azure
    ## @TODO : remove results folder from within trainer class
    trainer_config._dict['results_folder'] = os.path.join(*loadpath)

    renderer = render_config()
    model = model_config(diffusion.transition_dim)

    model = model.to(device = device)
    
    gradient_matching_config = Config(
        "rrf_diffusion.GradientRewardSingleStep",
        diffusion_predicts_mean=True,
        eps_loss = 0.0,
        scale_scores=False,
        alpha = 0.0
    )
    gradient_matching = gradient_matching_config(
        model = model,
        s_expert = diffusion,
        s_general = diffusion
    )
    trainer = trainer_config(
        gradient_matching = gradient_matching,
        s_expert = diffusion,
        s_general = diffusion, 
        expert_dataset = dataset, 
        general_dataset = dataset,
        renderer = renderer,
        heatmap_dataset = dataset
    )

    if epoch == 'latest':
        epoch = get_latest_epoch(loadpath)

    print(f'\n[ utils/serialization ] Loading model epoch: {epoch}\n')

    trainer.load(epoch)

    return DiffusionExperiment(dataset, renderer, trainer.model, diffusion, trainer.ema_model, trainer, epoch)