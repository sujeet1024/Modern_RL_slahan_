

from rlm.Agent import Agent
from rlm.utils import read_configs
from .reinforce import DiscrPi, VecEnc, ImgEnc, ActHead

import os
import sys
import yaml
import torch
import logging
import numpy as np
import torch.nn as nn



class AcCritic(Agent):
    def __init__(self, cfgs=None, device=None, **kwargs):
        super().__init__()
        ''' 
        TD(1) actor critic
        actions: action space of the environment
        observation_size: what arr size would observations be
        obs_type: ('vec', 'image') are observations normal arrays (in case of joint values etc) or and image
        exploration: ('eps_gr', 'softmax', 'max_ent', 'counts', 'thompson') choose an exploration strategy or provide one'''
        super().__init__()
        if cfgs is not None:
            self.cfgs = cfgs
            self.actions_cardinal = self.cfgs['env']['actions_shape']
            self.action_cardinal = self.cfgs['env']['action_shape']
            self.obs_size = self.cfgs['env']['obs_size']
            self.obs_type = self.cfgs['env']['obs_type']
            self.gamma = self.cfgs['agent']['discount']
            self.exploration = self.cfgs['exploration']['name']
            self.eps = min(abs(self.cfgs['exploration']['param']['eps']), 1)
            # self.buffer = getattr(rbuffs, self.cfgs['ReplayBuffer']['name'])(capacity=self.cfgs['ReplayBuffer']['capacity'], removals=self.cfgs['ReplayBuffer']['remove'])

        else:
            self.actions_cardinal = kwargs['actions_shape']
            self.action_cardinal = kwargs['action_shape']
            self.obs_size = kwargs['obs_size']
            self.obs_type = kwargs['obs_type']
            self.gamma = kwargs['discount']
            self.eps = min(kwargs['epsilon'],1)
            self.exploration = kwargs['exploration']
            self.cfgs = None
            # from rlm.Agent.ReplayBuffers import RBuff
            # self.buffer = RBuff()
        self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.last_obs, self.last_action = None, None
        self.iteration = 0

        self.pi = DiscrPi(self.obs_type, self.obs_size, self.action_cardinal, self.actions_cardinal)
        self.vf = VQN(self.obs_type, self.obs_size)
        self.pi.to(self.device)
        self.vf.to(self.device)
        
        self.pi_optimizer = torch.optim.Adam(self.pi.parameters(), 0.0001)
        self.vf_optimizer = torch.optim.Adam(self.vf.parameters(), 0.0001)

    def step(self, obs, iteration, avl_actions=None, **kwargs):
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        probs = self.pi(obs)
        action = torch.multinomial(probs, 1)
        if self.action_cardinal==1:
            self.last_action = int(action.detach().squeeze())
        else:
            self.last_action = action.detach().squeeze().numpy()
        self.last_obs = obs
        return self.last_action
        # val = self.vf(obs)
        # return val.detach()

    def update(self, rew, next_obs, terminal=False, **kwargs):
        next_obs = torch.tensor(next_obs, dtype=torch.float32).to(self.device)
        delta = rew + self.gamma * (1-terminal) * self.vf(next_obs) - self.vf(self.last_obs)
        loss_vf = - delta**2    # implementation detail: avoid going to -inf
        self.vf_optimizer.zero_grad()
        loss_vf.backward()
        self.vf_optimizer.step()
        loss_pi = - delta.detach() * torch.log(self.pi(self.last_obs)[:,self.last_action])
        self.pi_optimizer.zero_grad()
        loss_pi.backward()
        self.pi_optimizer.step()

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
        torch.save(self.pi, f"{exp_dir}/pi.pth")
        torch.save(self.vf, f"{exp_dir}/vf.pth")

    def load(self, path:str=None, **kwargs):
        if not os.path.exists(path+'/configs.yml'):
            logging.error(path+'/configs.yml not found, please make sure you have configs of the model to load')
            sys.exit(1)
        if not os.path.exists(path+'/pi.pth'):
            logging.error(path+'/pi.pth not found, please make sure you have the parameters of the learned agent')
            sys.exit(1)
        if not os.path.exists(path+'/vf.pth'):
            logging.error(path+'/vf.pth not found, please make sure you have the parameters of the learned agent')
            sys.exit(1)
        
        cfgs = read_configs(path+'/configs.yml')
        agent = self.__class__(cfgs, self.device)
        agent.pi = torch.load(path+'/pi.pth', weights_only=False)
        agent.vf = torch.load(path+'/vf.pth', weights_only=False)
        return agent




class VQN(nn.Module):
    def __init__(self, obs_type, obs_size, **kwargs):
        super().__init__()
        if obs_type=='vec':
            enc_size = 32
            self.enc = VecEnc(max(obs_size), enc_size)
        elif obs_type=='image':
            enc_size = 128
            self.enc = ImgEnc(obs_size, enc_size)
        self.dec = ActHead(enc_size, 1, 1)
    def forward(self, obs):
        enc = self.enc(obs)
        value = self.dec(enc)
        return value
