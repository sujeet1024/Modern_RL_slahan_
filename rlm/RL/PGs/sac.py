
import importlib
from rlm.Agent.agent import Agent
from rlm.utils import read_configs, polyak_update, to_buffer_type, to_torch
rbuffs = importlib.import_module("rlm.Agent.ReplayBuffers")


import os
import sys
import yaml
import copy
import torch
import logging
import numpy as np
import torch.nn as nn



class SAC(Agent):
    ''' Implementation of sac and applications, 2019 '''
    def __init__(self, cfgs=None, device=None, path=None, **kwargs):
        super().__init__()
        ''' 
        TD(1) ddpg
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
            self.buffer = getattr(rbuffs, self.cfgs['ReplayBuffer']['name'])(capacity=self.cfgs['ReplayBuffer']['capacity'], removals=self.cfgs['ReplayBuffer']['remove'])
            self.normalize_action = self.cfgs['agent']['normalize']

        else:
            self.actions_cardinal = kwargs['actions_shape']
            self.action_cardinal = kwargs['action_shape']
            self.obs_size = kwargs['obs_size']
            self.obs_type = kwargs['obs_type']
            self.gamma = kwargs['discount']
            self.eps = min(kwargs['epsilon'],1)
            self.exploration = kwargs['exploration']
            self.normalize_action = kwargs['normalize_action']
            self.cfgs = None
            from rlm.Agent.ReplayBuffers import RBuff
            self.buffer = RBuff(kwargs['RBcapacity'], kwargs['RBremovals'])
        self.sample_action = False
        self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.last_obs, self.last_action = None, None
        self.iteration = 0
        self.timestep = 0
        self.sav_path = self.get_logs_dir(path) if path is not None else None

        if self.obs_type=='vec':
            self.pi = VecPolicy(self.obs_size[0], self.action_cardinal*2)
            self.qf1 = VecPolicy(self.obs_size[0], self.action_cardinal, action_inp=1)
            self.qf2 = VecPolicy(self.obs_size[0], self.action_cardinal, action_inp=1)
        elif self.obs_type=='image':
            self.pi = PixPolicy(self.obs_size, self.action_cardinal*2)
            self.qf1 = PixPolicy(self.obs_size, self.action_cardinal, action_inp=1)
            self.qf2 = PixPolicy(self.obs_size, self.action_cardinal, action_inp=1)
        # self.entr_reg = nn.Parameter(torch.tensor([min(1/self.action_cardinal, 0.2)], dtype=torch.float32))
        self.entr_reg_log = nn.Parameter(torch.log(torch.tensor([0.3], dtype=torch.float32, device=self.device))) # so alpha>0
        self.pi.to(self.device)
        self.qf1.to(self.device)
        self.qf2.to(self.device)
        self.qf1_ = copy.deepcopy(self.qf1)
        self.qf2_ = copy.deepcopy(self.qf2)
        self.tau = 5e-3
        self.entr_target = -self.action_cardinal
        self.action_sampler = ActionSampler(self.normalize_action)
        
        self.pi_optimizer = torch.optim.Adam(self.pi.parameters(), 3e-4)
        self.qf1_optimizer = torch.optim.Adam(self.qf1.parameters(), 3e-4)
        self.qf2_optimizer = torch.optim.Adam(self.qf2.parameters(), 3e-4)
        self.entr_reg_optimizer = torch.optim.Adam([self.entr_reg_log], 3e-4)

    def step(self, obs, iteratin, avl_action=None, **kwargs):
        obs_ = torch.tensor(obs, dtype=torch.float32).to(self.device)
        self.timestep += 1
        action_mu_logsigma = self.pi(obs_).detach().cpu()
        action_ = action_mu_logsigma[:self.action_cardinal]
        if self.sample_action:
            action_, logpi = self.action_sampler.step(action_mu_logsigma[:self.action_cardinal], action_mu_logsigma[self.action_cardinal:])
        elif self.normalize_action:
            action_ = torch.tanh(action_)
        if self.last_action is not None:
            buff_tuple = to_buffer_type(self.last_obs, self.last_action, self.rew, obs)
            self.buffer.add(buff_tuple)
        self.last_obs = obs
        self.last_action = action_.numpy()
        return np.atleast_1d(self.last_action.squeeze())

    def update(self, rew, next_obs, terminal=False, **kwargs):
        self.rew = rew
        if self.timestep>50:
            # update from buffer batchwise
            obss, actions, rews, obss_ = self.buffer.sample()
            obss = to_torch(obss, self.device)
            actions = to_torch(actions, self.device)
            rews = to_torch(rews, self.device)
            obss_ = to_torch(obss_, self.device)
            # update critic : entirely from dataset (s,a)
            action_mu_logsigma_ = self.pi(obss_).detach()
            actions_, logpi_actions_ = self.action_sampler.step(action_mu_logsigma_[:, :self.action_cardinal], action_mu_logsigma_[:, self.action_cardinal:])
            L_qf1 = (self.qf1(obss, actions) - (rews + self.gamma * (self.qf1_(obss_, actions_) - self.entr_reg_log.exp() * logpi_actions_))).pow(2).mean()
            self.qf1_optimizer.zero_grad()
            L_qf1.backward()
            self.qf1_optimizer.step()
            L_qf2 = (self.qf2(obss, actions) - (rews + self.gamma * (self.qf2_(obss_, actions_) - self.entr_reg_log.exp() * logpi_actions_))).pow(2).mean()
            self.qf2_optimizer.zero_grad()
            L_qf2.backward()
            self.qf2_optimizer.step()
            if self.timestep%1==0:
                # update actor : s from dataset, a from gaussian policy
                action_mu_logsigma = self.pi(obss)
                actions, logpi_actions = self.action_sampler.step(action_mu_logsigma[:, :self.action_cardinal], action_mu_logsigma[:, self.action_cardinal:])
                q_min = torch.min(self.qf1_(obss, actions), self.qf2_(obss, actions))
                J = (self.entr_reg_log.exp() * logpi_actions - q_min).mean()
                self.pi_optimizer.zero_grad()
                J.backward()    # qf grads also calculated, but not backproped, only pi params updated in next line
                self.pi_optimizer.step()
                # update entropy regularization parameter alpha
                J_alpha = - (self.entr_reg_log * (logpi_actions + self.entr_target).detach()).mean() # makes faster grad updates against alpha
                self.entr_reg_optimizer.zero_grad()
                J_alpha.backward()
                self.entr_reg_optimizer.step()
                # update target nets
                polyak_update(self.qf1_, self.qf1, self.tau)
                polyak_update(self.qf2_, self.qf2, self.tau)
                # if terminal: self.timestep = 0

    def save(self, path:str=None, **kwargs):
        if self.sav_path is None:
            if not os.path.exists(path):
                logging.warning(f'path: {path} not found, creating one')
                exp_dir = os.path.join(path, f'exp_{1:06d}')
                os.makedirs(exp_dir)
            else:
                exp_dir = os.path.join(path, f'exp_{len(os.listdir(path))+1:06d}')
                os.makedirs(exp_dir)
            self.sav_path = exp_dir
            logging.info(f'Agent details for the experiment will be saved at {exp_dir}')
        else: exp_dir = self.sav_path
        with open(exp_dir+'/configs.yml', 'w') as configs:
            yaml.dump(self.cfgs, configs, default_flow_style=False)
        torch.save(self.pi, f"{exp_dir}/pi.pth")
        torch.save(self.qf1, f"{exp_dir}/qf1.pth")
        torch.save(self.qf2, f"{exp_dir}/qf2.pth")

    def get_logs_dir(self, path):
        if not os.path.exists(path):
            logging.warning(f'path: {path} not found, creating one')
            exp_dir = os.path.join(path, f'exp_{1:06d}')
            os.makedirs(exp_dir)
        else:
            exp_dir = os.path.join(path, f'exp_{len(os.listdir(path))+1:06d}')
            os.makedirs(exp_dir)
        logging.info(f'Agent details for the experiment will be saved at {exp_dir}')
        return exp_dir

    def load(self, path:str=None, **kwargs):
        if not os.path.exists(path+'/configs.yml'):
            logging.error(path+'/configs.yml not found, please make sure you have configs of the model to load')
            sys.exit(1)
        if not os.path.exists(path+'/pi.pth'):
            logging.error(path+'/pi.pth not found, please make sure you have the parameters of the learned agent')
            sys.exit(1)
        if not os.path.exists(path+'/qf1.pth'):
            logging.error(path+'/qf1.pth not found, please make sure you have the parameters of the learned agent')
            sys.exit(1)
        if not os.path.exists(path+'/qf2.pth'):
            logging.error(path+'/qf2.pth not found, please make sure you have the parameters of the learned agent')
            sys.exit(1)
        
        cfgs = read_configs(path+'/configs.yml')
        agent = self.__class__(cfgs, self.device)
        agent.pi = torch.load(path+'/pi.pth', weights_only=False).to(self.device)
        agent.qf1 = torch.load(path+'/qf1.pth', weights_only=False).to(self.device)
        agent.qf2 = torch.load(path+'/qf2.pth', weights_only=False).to(self.device)
        agent.qf1_ = copy.deepcopy(agent.qf1)
        agent.qf2_ = copy.deepcopy(agent.qf2)
        return agent


class VecPolicy(nn.Module):
    def __init__(self, obs_size, action_size, action_inp=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flat1 = nn.Linear(obs_size, 400-action_inp*action_size)
        self.relu1 = nn.ReLU()
        self.bnorm2 = nn.LayerNorm(400)
        self.flat2 = nn.Linear(400, 300)
        self.relu2 = nn.ReLU()
        self.bnorm3 = nn.LayerNorm(300)
        self.flat3 = nn.Linear(300, action_size)
    def forward(self, obs, action=None):
        x = self.relu1(self.flat1(obs))#.squeeze().unsqueeze(0)))
        if action is not None:
            x = torch.concatenate((x, action), -1)#.unsqueeze(0)
        x = self.relu2(self.flat2(self.bnorm2(x)))
        out = self.flat3(self.bnorm3(x))
        return out

class PixPolicy(nn.Module):
    def __init__(self, obs_size, action_size, action_inp=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(obs_size[-1], 32, 4, 2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(3, 2)
        self.bnorm2 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.relu2 = nn.ReLU()
        self.bnorm3 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.relu3 = nn.ReLU()
        self.f = nn.Flatten(start_dim=1)
        flat_size = self.relu3(self.conv3(self.bnorm3(self.relu2(self.conv2(self.bnorm2(self.pool1(self.relu1(self.conv1(torch.zeros(obs_size).permute(2,0,1)))))))))).numel()
        print('flat_size:', flat_size)
        self.bnorm4 = nn.BatchNorm1d(flat_size)
        flatten_size = 512 if action_size<65 else 1024
        self.flat4 = nn.Linear(flat_size, flatten_size-action_size*action_inp)
        self.relu4 = nn.ReLU()
        self.dec = VecPolicy(flatten_size-action_size*action_inp, action_size, action_inp)
    def forward(self, obs, action=None):
        x = self.pool1(self.relu1(self.conv1(obs)))#.unsqueeze(0))))
        x = self.relu2(self.conv2(self.bnorm2(x)))
        x = self.f(self.relu3(self.conv3(self.bnorm3(x))))
        x = self.relu4(self.flat4(self.bnorm4(x)))
        out = self.dec(x, action)
        return out


class ActionSampler:
    def __init__(self, normalize:bool):
        self.normalize = normalize

    def step(self, mu:torch.Tensor, logsigma:torch.Tensor):
        # dist = torch.distributions.MultivariateNormal(mu, torch.diag(logsigma.exp().pow(2)))
        logsigma = torch.clamp(logsigma, min=-20, max=2)
        dist = torch.distributions.Normal(mu, logsigma.exp())  # creates normal per dimension
        sample_action = dist.rsample() # keeps x differentiable wrt mu, sigma by reparameterizing as x = μ + σ ε, o/w not
        log_prob = dist.log_prob(sample_action)
        if self.normalize:
            sample_action = torch.tanh(sample_action)
            log_prob -= torch.log(1 - torch.tanh(sample_action).pow(2))
        return sample_action, log_prob.sum(dim=-1, keepdim=True)