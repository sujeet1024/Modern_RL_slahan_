
import yaml
import numpy as np


def read_configs(cfg_path:str):
    with open(cfg_path, "r") as file:
        cfgs = yaml.load(file, Loader=yaml.FullLoader)
        # breakpoint()
    return cfgs


def get_cardinals(actions_space, obs_space, cfgs):
    if hasattr(actions_space, 'n'):
        actions_cardinal = int(actions_space.n)
    else:
        actions_cardinal = max(actions_space.shape)
    if isinstance(actions_space.sample(), np.ndarray):
        action_cardinal = max(actions_space.sample().shape)
    else:
        action_cardinal = 1
    # breakpoint()
    if hasattr(obs_space, 'n'):
        obs_size = np.array([1])
    elif hasattr(obs_space, '__len__'):
        obs_size = np.array([len(obs_space)])
    elif hasattr(obs_space, 'shape'):
        obs_size = obs_space.shape
    cfgs['env']['actions_shape'] = actions_cardinal
    cfgs['env']['action_shape'] = action_cardinal
    cfgs['env']['obs_size'] = obs_size
    return cfgs #actions_cardinal, action_cardinal, obs_size