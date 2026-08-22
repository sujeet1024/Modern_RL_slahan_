

from rlm.Agent import Agent
from rlm.utils import read_configs

import os
import sys
import yaml
import torch
import logging
import numpy as np
import torch.nn as nn


class REINFORCE(Agent):
    ''' REINFORCE with non linear FA '''
    def __init__(self, cfgs=None, device=None, **kwargs):
        ''' actions: action space of the environment
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
            
        else:
            self.actions_cardinal = kwargs['actions_shape']
            self.action_cardinal = kwargs['action_shape']
            self.obs_size = kwargs['obs_size']
            self.obs_type = kwargs['obs_type']
            self.gamma = kwargs['discount']
            self.eps = min(kwargs['epsilon'],1)
            self.exploration = kwargs['exploration']
            self.cfgs = None
        self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.last_obs, self.last_action = None, None
        self.iteration = 0

        self.pi = DiscrPi(self.obs_type, self.obs_size, self.action_cardinal, self.actions_cardinal)
        # print(sum(p.numel() for p in self.pi.parameters()))
        self.pi.to(self.device)
        
        self.optimizer = torch.optim.Adam(self.pi.parameters(), 0.0001)

    def step(self, obs, iteration, avl_actions=None, **kwargs):
        # policy step, multiple such to generate an episode -> try in cartpole
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        probs = self.pi(obs)
        action = torch.multinomial(probs, 1)
        if self.action_cardinal==1:
            return int(action.detach().squeeze())
        else:
            return action.detach().squeeze().numpy()

    def update(self, traj, avl_actions=None, **kwargs):
        # update from trajectory -> list[tuple(s,a,r,t)]
        G = 0
        for (obs, act, rew, t) in reversed(traj):
            G = rew + self.gamma * G
            # params += self.step_size * (self.gamma)**t * G * grad(ln(pi))
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
            loss = - G * torch.log(self.pi(obs)[:,act])
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

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

    def load(self, path:str=None, **kwargs):
        if not os.path.exists(path+'/configs.yml'):
            logging.error(path+'/configs.yml not found, please make sure you have configs of the model to load')
            sys.exit(1)
        if not os.path.exists(path+'/pi.pth'):
            logging.error(path+'/pi.pth not found, please make sure you have the parameters of the learned agent')
            sys.exit(1)
        
        cfgs = read_configs(path+'/configs.yml')
        agent = self.__class__(cfgs, self.device)
        agent.pi = torch.load(path+'/pi.pth', weights_only=False)
        return agent

    



class DiscrPi(nn.Module):
    def __init__(self, obs_type, obs_size, action_cardinal, actions_cardinal):
        super().__init__()
        if obs_type=='vec':
            enc_size = 32
            self.enc = VecEnc(max(obs_size), enc_size)
        elif obs_type=='image':
            enc_size = 128
            self.enc = ImgEnc(obs_size, enc_size)
        self.dec = ActHead(enc_size, action_cardinal, actions_cardinal)
    def forward(self, obs):
        enc = self.enc(obs)
        out = self.dec(enc)
        return out

class VecEnc(nn.Module):
    def __init__(self, obs_size, enc_size):
        super().__init__()
        self.flat1 = nn.Linear(obs_size, 32)
        self.relu1 = nn.ReLU()
        # self.flat2 = nn.Linear(32, 64)
        # self.relu2 = nn.ReLU()
        self.out1 = nn.Linear(32, enc_size)
    def forward(self, obs):
        obs = obs.unsqueeze(0)
        x = self.flat1(obs)
        x = self.relu1(x)
        # x = self.flat2(x)
        # x = self.relu2(x)
        out = self.out1(x)
        return out

class ImgEnc(nn.Module):
    def __init__(self, obs_size, enc_size):
        super().__init__()
        self.c1 = nn.Conv2d(obs_size[-1], 32, 7, 3) # or min(obs_size)
        self.relu1 = nn.ReLU()
        self.c2 = nn.Conv2d(32, 64, 3, 2)
        self.relu2 = nn.ReLU()
        self.c3 = nn.Conv2d(64, 16, 3)
        self.relu3 = nn.ReLU()
        self.f = nn.Flatten(start_dim=1)
        flat_size = self.relu3(self.c3(self.relu2(self.c2(self.relu1(self.c1(torch.ones(obs_size).permute(2,0,1))))))).numel()
        # print('flatten size', flat_size)
        self.f5 = nn.Linear(flat_size, 1024)
        self.relu5 = nn.ReLU()
        self.f6 = nn.Linear(1024, enc_size)
    def forward(self, obs):
        obs = obs.permute(2,0,1).unsqueeze(0)
        x = self.c1(obs)
        x = self.relu1(x)
        x = self.c2(x)
        x = self.relu2(x)
        x = self.c3(x)
        x = self.relu3(x)
        x = self.f(x)
        x = self.f5(x)
        x = self.relu5(x)
        out = self.f6(x)
        return out

class ActHead(nn.Module):
    def __init__(self, enc_size, action_cardinal, actions_cardinal):
        super().__init__()
        self.f1 = nn.Linear(enc_size, 32)
        self.relu1 = nn.ReLU()
        self.f2 = nn.Linear(32, action_cardinal*actions_cardinal)
        self.probs = nn.Softmax()
    def forward(self, enc):
        x = self.f1(enc)
        x = self.relu1(x)
        x = self.f2(x)
        out = self.probs(x)
        return out