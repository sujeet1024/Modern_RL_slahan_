import os, sys
import logging
import yaml
from rlm.utils import read_configs

import numpy as np
from rlm.Agent import Agent, ActionPolicy
from .action_policies.base import SPolicy



class ESarsa(Agent):
    ''' The Expected-SARSA Agent'''
    def __init__(self, cfgs, device, **kwargs):
        super().__init__()
        if cfgs is not None:
            self.cfgs = cfgs
            self.actions_cardinal = self.cfgs['env']['actions_shape']
            self.action_cardinal = self.cfgs['env']['action_shape']
            self.obs_size = self.cfgs['env']['obs_size']
            self.obs_type = self.cfgs['env']['obs_type']
            self.obs_space = self.cfgs['env']['obs_space']
            self.gamma = self.cfgs['agent']['discount']
            self.exploration = self.cfgs['exploration']['name']
            self.eps = min(abs(self.cfgs['exploration']['param']['eps']), 1)
            
        else:
            self.actions_cardinal = kwargs['actions_shape']
            self.action_cardinal = kwargs['action_shape']
            self.obs_size = kwargs['obs_size']
            self.obs_type = kwargs['obs_type']
            self.gamma = kwargs['discount']
            self.eps = min(kwargs['epsilon'],1)
            self.exploration = kwargs['exploration']
            self.cfgs = None

        obs_indices = tuple((np.array(self.obs_space['high']) - np.array(self.obs_space['low']) + 1).astype(np.int32).tolist())
        self.avl_actions = list(range(self.actions_cardinal))
        self.q_f = np.zeros((*obs_indices, self.actions_cardinal))
        self.last_obs, self.last_action = None, None
        self.iteration = 0

    def step(self, obs, iterations:int, bpolicy:ActionPolicy=None, **kwargs):
        # can choose between on policy, off-policy
        self.last_obs = tuple(obs.astype(np.int32).tolist())
        action_potentials = self.q_f[*self.last_obs, :]
        self.last_action = bpolicy.step(obs, self.avl_actions, action_potentials)
        return int(self.last_action)

    def update(self, reward, next_obs, terminal:bool, tpolicy:SPolicy=None, **kwargs):
        nxt_obs = tuple(next_obs.astype(np.int32).tolist())
        action_potentials = self.q_f[*nxt_obs, :]
        v_ = (1-terminal) * (tpolicy.propose(nxt_obs, self.avl_actions, action_potentials)*self.q_f[*nxt_obs, :]).sum()
        delta = reward + v_ - self.q_f[*self.last_obs, self.last_action]
        self.q_f[*self.last_obs, self.last_action] = self.q_f[*self.last_obs, self.last_action] + delta

    def save(self, path:str=None, **kwargs):
        if not os.path.exists(path):
            logging.warning(f'path: {path} not found, creating one')
            exp_dir = os.path.join(path, f'exp_{1:06d}')
            os.makedirs(exp_dir)
        else:
            exp_dir = os.path.join(path, f'exp_{len(os.listdir(path))+1:06d}')
            os.makedirs(exp_dir)
        logging.info(f'Agent details for the experiment will be saved at {exp_dir}')
        with open(exp_dir+'/configs.yml', 'w') as configs:
            yaml.dump(self.cfgs, configs, default_flow_style=False)
        np.save(self.q_f, f"{exp_dir}/q_f.npy")

    def load(self, path:str=None, **kwargs):
        if not os.path.exists(path+'/configs.yml'):
            logging.error(path+'/configs.yml not found, please make sure you have configs of the model to load')
            sys.exit(1)
        if not os.path.exists(path+'/q_f.npy'):
            logging.error(path+'/q_f.npy not found, please make sure you have the parameters of the learned agent')
            sys.exit(1)
        cfgs = read_configs(path+'/configs.yml')
        agent = self.__class__(cfgs, self.device)
        agent.q_f = np.load(path+'/q_f.npy')
        return agent