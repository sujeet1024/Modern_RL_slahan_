from rlm.Agent import Agent
from rlm.utils import read_configs
from rlm.Agent.ReplayBuffers import RBuff

import os
import sys
import yaml
import torch
import logging
import warnings
import numpy as np
import torch.nn as nn


class DQN(Agent):
    def __init__(self, actions_shape, action_shape, obs_size, obs_type:str, exploration='eps_gr', discount=0.99, epsilon=0.1, cfgs=None, device=None):
        ''' actions: action space of the environment
        observation_size: what arr size would observations be
        obs_type: ('vec', 'image') are observations normal arrays (in case of joint values etc) or and image
        exploration: ('eps_gr', 'softmax', 'max_ent', 'counts', 'thompson') choose an exploration strategy or provide one'''
        super().__init__()
        # self.actions = actions
        if cfgs is not None:
            self.cfgs = cfgs
            self.actions_cardinal = self.cfgs['actions_shape']
            self.action_cardinal = self.cfgs['action_shape']
            self.obs_size = self.cfgs['obs_size']
            self.obs_type = self.cfgs['obs_type']
            self.gamma = self.cfgs['discount']
            self.eps = min(abs(self.cfgs['epsilon']), 1)
            self.exploration = self.cfgs['exploration']
            self.buffer = self.cfgs['ReplayBuffer']

        self.actions_cardinal = actions_shape
        self.action_cardinal = action_shape
        self.obs_size = obs_size
        self.gamma = discount
        self.eps = min(epsilon,1)
        self.last_obs, self.last_action = None, None
        self.exploration = exploration
        self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
        # breakpoint()
        self.cfgs = None
        self.buffer = RBuff()
        self.iteration = 0

        if obs_type=='vec':
            self.q_f = QN(max(obs_size), self.action_cardinal)
        elif obs_type=='image':
            self.q_f = CQN(obs_size, self.action_cardinal)
        self.q_f.to(self.device)
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.q_f.parameters(), 0.0001)
    
    def step(self, obs, step, avl_actions=None):
        self.iteration = step
        self.last_obs = torch.tensor(obs, dtype=torch.float32)
        last_obs = torch.tile(self.last_obs, (self.actions_cardinal, 1)).to(self.device)
        if avl_actions==None: #avl_actions = self.actions
            avl_actions = torch.tensor(list(range(self.actions_cardinal)), dtype=torch.float32).unsqueeze(dim=-1)
        avl_actions = avl_actions.to(self.device)
        # breakpoint()  

        # epsilon-greedy strategy
        if self.exploration == 'eps_gr':
            prob = torch.rand(1,).item()
            eps = max(0.05, self.eps*(1-step/1e6))
            if prob<=eps:
                # breakpoint()
                self.last_action = torch.tensor(np.random.choice(avl_actions.cpu().numpy().squeeze()), dtype=torch.float32)
            else:
                # breakpoint()
                self.last_action = torch.argmax(self.q_f(last_obs, avl_actions))
        elif self.exploration == 'softmax':
            self.last_action = torch.multinomial(nn.Softmax(self.q_f(last_obs, avl_actions)),num_samples=1)
        # breakpoint()
        return int(self.last_action.item())
        
    def update(self, reward, next_obs, terminal:bool, avl_actions=None):
        nxt_obs = torch.tile(torch.tensor(next_obs, dtype=torch.float32), (self.actions_cardinal, 1)).to(self.device)
        if avl_actions==None: # avl_actions = self.actions
            avl_actions = torch.tensor(list(range(self.actions_cardinal)), dtype=torch.float32).unsqueeze(dim=-1)
        avl_actions = avl_actions.to(self.device)
        # if terminal:
        #     v_ = 0
        # else:
        #     v_ = torch.max(self.q_f(nxt_obs, avl_actions))
        v_ = (1-terminal) * (torch.max(self.q_f(nxt_obs, avl_actions).detach()))
        target = torch.tensor([reward], dtype=torch.float32).to(self.device) + self.gamma * v_
        # TD-error
        last_obs = torch.tile(self.last_obs, (self.action_cardinal,1)).to(self.device)
        last_action = self.last_action.unsqueeze(dim=-1).unsqueeze(-1) if self.last_action.shape==torch.Size([]) else self.last_action.unsqueeze(-1)
        last_action = last_action.to(self.device)
        # breakpoint()
        loss = self.loss(self.q_f(last_obs, last_action), target)
        loss.backward()
        self.optimizer.step()

        datum_tup = (self.last_obs.cpu().detach().numpy(), [self.last_action.cpu().detach().numpy()], [reward], next_obs, [terminal])
        # if terminal:
        #     for i in range(100): self.buffer.add(datum_tup)
        # self.buffer.add(datum_tup)
        # if self.iteration>1e4:
        #     self.update_from_buffer(4)
    
    def update_from_buffer(self, sample_size):
        obs, action, reward, next_obs, terminal = self.buffer.sample(sample_size)
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        action = torch.tensor(action, dtype=torch.float32).to(self.device)
        next_obs = torch.tensor(next_obs, dtype=torch.float32).to(self.device).unsqueeze(1).expand(-1, self.actions_cardinal, -1).reshape(-1, max(self.obs_size)).to(self.device)
        avl_actions = torch.tile(torch.tensor(list(range(self.actions_cardinal)), dtype=torch.float32), (int(reward.shape[0]), 1)).to(self.device).unsqueeze(-1).reshape(-1, self.action_cardinal).to(self.device)
        # breakpoint()
        
        v_ = (1 - torch.tensor(terminal, dtype=torch.float32).to(self.device)) * (torch.max(self.q_f(next_obs, avl_actions).detach().reshape((reward.shape[0], self.actions_cardinal, self.action_cardinal)), dim=1))[0]
        target = torch.tensor(reward, dtype=torch.float32).to(self.device) + self.gamma * v_
        loss = self.loss(self.q_f(obs, action), target)
        loss.backward()
        self.optimizer.step()

    def save(self, path:str=None, **kwargs):
        if not os.path.exists(path):
            logging.warning(f'path: {path} not found, creating one')
            exp_dir = os.path.join(path, f'exp_{1:04d}')
            os.makedirs(exp_dir)
        else:
            exp_dir = os.path.join(path, f'exp_{len(os.listdir(path))+1:04d}')
            os.makedirs(exp_dir)
        # print(f'Agent details for the experiment will be saved at {exp_dir}')
        logging.info(f'Agent details for the experiment will be saved at {exp_dir}')
        with open(exp_dir+'/configs.yml', 'w') as configs:
            yaml.dump(self.cfgs, configs, default_flow_style=False)
        torch.save(self.q_f.state_dict(), f"{exp_dir}/q_f.pth")

    def load(self, path:str=None, **kwargs):
        if not os.path.exists(path+'/configs.yml'):
            logging.error(path+'/configs.yml not found, please make sure you have configs of the model to load')
            sys.exit(1)
        if not os.path.exists(path+'/q_f.pth'):
            logging.error(path+'/q_f.pth not found, please make sure you have the parameters of the learned agent')
            sys.exit(1)
        
        cfgs = read_configs(path+'/configs.yml')
        agent = self.__class__(cfgs)
        agent.q_f.load_state_dict(torch.load(path+'/q_f.pth'))
        return agent





class QN(nn.Module):
    def __init__(self, obs_size, action_size, **kwargs):
        super().__init__()
        self.flat1 = nn.Linear(obs_size+action_size, 64)
        self.relu = nn.ReLU()
        self.flat2 = nn.Linear(64, 128)
        self.relu2 = nn.ReLU()
        self.flat3 = nn.Linear(128, 64)
        self.outl = nn.Linear(64, action_size)

    def forward(self, obs, action):
        # breakpoint()
        obs_action = torch.concatenate([obs, action], dim=-1)
        x = self.flat1(obs_action)
        # x = self.relu(x)
        x = self.flat2(x)
        # x = self.relu2(x)
        x = self.flat3(x)
        out = self.outl(x)
        return out


class CQN(nn.Module):
    def __init__(self, input_observation_size, action_size, **kwargs):
        super().__init__()
        # self.actions_cardinality = len(action_size)
        self.input_size = input_observation_size

        # use a pretrained resnet preferably, flatten its output to action_cardinality size
        self.l1 = nn.Conv2d(3, 64, 3)
        self.l2 = nn.Conv2d(64, 128, 3)
        self.l3 = nn.Conv2d(128, 256, 3)
        self.f1 = nn.Flatten(-1, 1000)
        self.f2 = nn.Linear(1000, 127)
        self.f3 = nn.Linear(128, 32)
        self.f4 = nn.Linear(32, action_size)

    def forward(self, obs, action):
        x = self.l1(obs)
        x = self.l2(x)
        x = self.l3(x)
        x = self.f1(x)
        x = self.f2(x)
        ot = self.f3(torch.concatenate(x, action))
        ot = self.f4(ot)
        return ot