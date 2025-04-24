
import yaml
import numpy as np


def read_configs(cfg_path:str):
    with open(cfg_path, "r") as file:
        cfgs = yaml.load(file, Loader=yaml.FullLoader)
    return cfgs


def get_cardinals(actions_space, obs_space):
    if hasattr(actions_space, 'n'):
        actions_cardinal = int(actions_space.n)
    else:
        actions_cardinal = max(actions_space.shape)
    if isinstance(actions_space.sample(), np.ndarray):
        action_cardinal = max(actions_space.sample().shape)
    else:
        action_cardinal = 1
    obs_size = obs_space.shape
    return actions_cardinal, action_cardinal, obs_size