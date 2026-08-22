

from rlm.Agent import Agent
from rlm.utils import read_configs

import os
import sys
import yaml
import torch
import logging
import numpy as np
import torch.nn as nn




class OffPAC(Agent):
    '''
    Implementation of linear FA OffPAC, non-linear to be followed
    '''
    def __init__(self, cfgs=None, device=None, **kwargs):
        if cfgs is not None:
            self.cfgs = cfgs
            self.actions_cardinal = self.cfgs['env']['actions_shape']
            self.action_cardinal = self.cfgs['env']['action_shape']
            self.obs_size = self.cfgs['env']['obs_size']
            self.obs_type = self.cfgs['env']['obs_type']
            self.gamma = self.cfgs['agent']['discount']
            self.lamda = self.cfgs['agent']['trace']
            self.exploration = self.cfgs['exploration']['name']
            self.eps = min(abs(self.cfgs['exploration']['param']['eps']), 1)
            # self.buffer = getattr(rbuffs, self.cfgs['ReplayBuffer']['name'])(capacity=self.cfgs['ReplayBuffer']['capacity'], removals=self.cfgs['ReplayBuffer']['remove'])

        else:
            self.actions_cardinal = kwargs['actions_shape']
            self.action_cardinal = kwargs['action_shape']
            self.obs_size = kwargs['obs_size']
            self.obs_type = kwargs['obs_type']
            self.gamma = kwargs['discount']
            self.lamda = kwargs['trace']   # lower lambda close to td, larger lambda close to mc
            self.eps = min(kwargs['epsilon'],1)
            self.exploration = kwargs['exploration']
            self.cfgs = None
            # from rlm.Agent.ReplayBuffers import RBuff
            # self.buffer = RBuff()
        self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.last_obs, self.last_action = None, None
        self.iteration = 0
        if self.obs_type!='vec' or len(self.obs_size)>1:
            logging.error('OffPAC can only work with vector observations, please reconsider the environment')
            sys.exit(1)

        self.offpCorr = np.zeros(self.obs_size)     # off policy correction in GTD(lambda)
        self.ec_trace = np.zeros(self.obs_size)     # trace for critic
        self.ea_trace = np.zeros(self.obs_size[0]+self.action_cardinal)     # trace for actor, same shape as pi params
        
        self.vf = np.random.random(self.obs_size)
        self.pi = LinP(self.obs_size, self.action_cardinal)
        self.pi_params = np.array([p.detach().cpu().numpy() for p in self.pi.parameters()]).squeeze()

        self.step_c = 1e-4
        self.step_opc = 1e-4
        self.step_a = 1e-4

    def step(self, obs, iteration, avl_actions=None, **kwargs):
        # obs = np.array(obs)
        self.last_obs = obs
        avl_actions = list(range(self.actions_cardinal))
        self.last_action = np.random.choice(avl_actions)
        if iteration>4000:
            self.last_action, _ = self.pi.step(obs, avl_actions)
        self.iteration+=1
        return int(self.last_action)

    def update(self, rew, next_obs, terminal=False, **kwargs):
        td = rew + self.gamma * np.dot(self.vf.T, next_obs) - np.dot(self.vf.T, self.last_obs)
        rho = self.pi(self.last_obs, self.last_action).detach().cpu().numpy() / (1/self.actions_cardinal)
        if self.iteration>4000:
            rho = 1.0
        # update critic
        self.ec_trace = rho * (self.last_obs + self.gamma * self.lamda * self.ec_trace)
        self.vf = self.vf + self.step_c * (td * self.ec_trace - self.gamma * (1-self.lamda) * (self.offpCorr.T * self.ec_trace)* self.last_obs)
        self.offpCorr = self.offpCorr + self.step_opc * (td * self.ec_trace - (self.offpCorr.T * self.last_obs)* self.last_obs)
        # update actor
        first_grad = torch.autograd.grad(torch.log(self.pi(self.last_obs, self.last_action)), self.pi.parameters(), create_graph=False)[0].squeeze()
        self.ea_trace = rho * (first_grad.detach().detach().cpu().numpy() + self.gamma * self.lamda * self.ea_trace)
        self.pi_params = self.pi_params + self.step_a * td * self.ea_trace

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
        np.save(self.vf, f"{exp_dir}/vf.npy")
        np.save(self.offpCorr, f"{exp_dir}/offpCorr.npy")
        np.save(self.ec_trace, f"{exp_dir}/ec_trace.npy")
        np.save(self.ea_trace, f"{exp_dir}/ea_trace.npy")

    def load(self, path:str=None, **kwargs):
        if not os.path.exists(path+'/configs.yml'):
            logging.error(path+'/configs.yml not found, please make sure you have configs of the model to load')
            sys.exit(1)
        if not os.path.exists(path+'/pi.pth'):
            logging.error(path+'/pi.pth not found, please make sure you have the parameters of the learned agent')
            sys.exit(1)
        if not os.path.exists(path+'/vf.npy'):
            logging.error(path+'/vf.pth not found, please make sure you have the parameters of the learned agent')
            sys.exit(1)
        
        cfgs = read_configs(path+'/configs.yml')
        agent = self.__class__(cfgs, self.device)
        agent.pi = torch.load(path+'/pi.pth', weights_only=False)
        agent.vf = np.load(path+'/vf.npy')
        agent.offpCorr = np.load(path+'/offpCorr.npy')
        agent.ec_trace = np.load(path+'/ec_trace.npy')
        agent.ea_trace = np.load(path+'/ea_trace.npy')
        return agent


class LinP(torch.nn.Module):
    ''' Linear FA policy '''
    def __init__(self, obs_size, action_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.l = nn.Linear(int(obs_size[0])+int(action_size), 1, bias=False)
        self.softmax = nn.Softmax(0)

    def forward(self, obs, action):
        action = np.array([action], dtype=np.float32)
        out = self.l(torch.tensor(np.concatenate([obs, action])))
        return out

    def step(self, obs, avl_actions):
        pis = []
        for action in avl_actions:
            # action = np.array([action], dtype=np.float32)
            pis.append(self.forward(obs, action).detach().cpu().numpy().item())
        probs = self.softmax(torch.tensor(pis)),1
        out_action = avl_actions[int(torch.multinomial(probs).item())]
        return out_action, probs.detach().cpu().numpy()