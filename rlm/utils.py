
import yaml
import torch
import numpy as np
import gymnasium as gym

import logging
from collections import deque


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
    if hasattr(obs_space, 'n'):
        obs_size = np.array([1])
    elif hasattr(obs_space, '__len__'):
        obs_size = np.array([len(obs_space)])
    elif hasattr(obs_space, 'shape'):
        obs_size = np.array(obs_space.shape)
    else:
        obs_size = None
        logging.warning("Observation size could not be extracted, please try a different environment and inform the maintainer")
    if obs_size is not None:
        obs_size[-1]*=cfgs['agent']['history']
        obs_size = tuple(obs_size)
    if hasattr(obs_space, 'high') and hasattr(obs_space, 'low'):
        obs_min = tuple(obs_space.low.tolist())
        obs_max = tuple(obs_space.high.tolist())
    else:
        obs_max, obs_min = None, None
        logging.warning("Observation space could not be extracted, please try a different environment and inform the maintainer")
    if hasattr(actions_space, 'high') and hasattr(actions_space, 'low'):
        act_min = tuple(actions_space.low.tolist())
        act_max = tuple(actions_space.high.tolist())
    elif isinstance(actions_space, gym.spaces.Discrete):
        act_min = [int(actions_space.start)]
        act_max = [int(actions_space.n - act_min[0] - 1)]
    else:
        act_max, act_min = None, None
        logging.warning("Action space could not be extracted, please try a different environment and inform the maintainer")
    cfgs['env']['obs_space'] = {'low': obs_min, 'high':obs_max}
    cfgs['env']['actions_shape'] = actions_cardinal
    cfgs['env']['action_shape'] = action_cardinal
    cfgs['env']['action_space'] = {'low': act_min, 'high': act_max}
    cfgs['env']['obs_size'] = obs_size
    return cfgs


class CustomLogger:
    def __init__(self, txt_logfile):
        self.logfile = txt_logfile
    def write(self, txt:str):
        with open(self.logfile, 'w') as txt_log:
            txt_log.write(txt)
    def append(self, txt:str):
        with open(self.logfile, 'a') as txt_log:
            txt_log.write(txt)


class ObsHist:
    def __init__(self, cfgs):
        self.maxlen = int((cfgs['agent']['history']))
        self.queue = None
    def intialize(self, o):
        self.queue = deque([o] * self.maxlen, maxlen=self.maxlen)
    def add(self, o):
        if self.queue is None:
            self.intialize(o)
        else:
            self.queue.append(o)
    def get(self):
        if self.queue is None:
            return None
        return np.concatenate(self.queue, -1)
    def __len__(self):
        return 0 if self.queue is None else len(self.queue)


def crop_obs(obs):
    j = 1
    while obs.min()/j>=84: j+=1
    scale_d = j-1

def to_buffer_type(obs:np.ndarray=None, action=None, rew=None, obs_=None, done=None, action_=None):
    return_tuple = []
    if obs is not None:
        if obs.size < 1000:
            return_tuple.append(np.array(obs, np.float32))
        else:
            return_tuple.append(np.array(obs, np.uint8))
    if action is not None:
        if isinstance(action, int):
            return_tuple.append(np.array([action], np.int16))
        else:
            return_tuple.append(np.array(action, np.float32))
    if rew is not None:
        return_tuple.append(np.array([rew], np.float32))
    if obs_ is not None:
        if obs_.size < 1000:
            return_tuple.append(np.array(obs_, np.float32))
        else:
            return_tuple.append(np.array(obs_, np.uint8))
    if done is not None:
        return_tuple.append(np.array([done], np.bool))
    if action_ is not None:
        if isinstance(action, int):
            return_tuple.append(np.array([action], np.int16))
        else:
            return_tuple.append(np.array(action, np.float32))
    if len(return_tuple)==1:
        return return_tuple[0]
    return return_tuple


def to_torch(inp, device='cpu'):
    return torch.tensor(inp, dtype=torch.float32).to(device)


@torch.no_grad()
def polyak_update(target_net, online_net, tau):
    for target_param, online_param in zip(target_net.parameters(), online_net.parameters()):
        target_param.lerp_(online_param, tau)