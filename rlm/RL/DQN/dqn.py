import importlib
from rlm.Agent.agent import Agent
from rlm.utils import read_configs
rbuffs = importlib.import_module("rlm.Agent.ReplayBuffers.rbuff")

import os
import sys
import yaml
import torch
import logging
import warnings
import numpy as np
import torch.nn as nn


class DQN(Agent):
    def __init__(self, cfgs=None, device=None, **kwargs):
        ''' actions: action space of the environment
        observation_size: what arr size would observations be
        obs_type: ('vec', 'image') are observations normal arrays (in case of joint values etc) or and image
        exploration: ('eps_gr', 'softmax', 'max_ent', 'counts', 'thompson') choose an exploration strategy or provide one'''
        super().__init__()
        # self.actions = actions
        if cfgs is not None:
            self.cfgs = cfgs
            # breakpoint()
            self.actions_cardinal = self.cfgs['env']['actions_shape']
            self.action_cardinal = self.cfgs['env']['action_shape']
            self.obs_size = self.cfgs['env']['obs_size']
            self.obs_type = self.cfgs['env']['obs_type']
            self.gamma = self.cfgs['agent']['discount']
            self.exploration = self.cfgs['exploration']['name']
            self.eps = min(abs(self.cfgs['exploration']['param']['eps']), 1)
            self.buffer = getattr(rbuffs, self.cfgs['ReplayBuffer']['name'])(capacity=self.cfgs['ReplayBuffer']['capacity'], removals=self.cfgs['ReplayBuffer']['remove'])

        else:
            self.actions_cardinal = kwargs['actions_shape']
            self.action_cardinal = kwargs['action_shape']
            self.obs_size = kwargs['obs_size']
            self.obs_type = kwargs['obs_type']
            self.gamma = kwargs['discount']
            self.eps = min(kwargs['epsilon'],1)
            self.exploration = kwargs['exploration']
            self.cfgs = None
            from rlm.Agent.ReplayBuffers import RBuff
            self.buffer = RBuff()
        self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.last_obs, self.last_action = None, None
        self.iteration = 0

        if self.obs_type=='vec':
            self.q_f = QN(max(self.obs_size), self.action_cardinal)
        elif self.obs_type=='image':
            self.q_f = CQN(self.obs_size, self.action_cardinal)
        self.q_f.to(self.device)
        # breakpoint()
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.q_f.parameters(), 0.0001)
    
    def step(self, obs, step, avl_actions=None):
        # print('stepping')
        self.iteration = step
        self.last_obs = torch.tensor(obs, dtype=torch.float32)
        if self.obs_type!='image':
            last_obs = torch.tile(self.last_obs.clone().detach(), (self.actions_cardinal, 1)).to(self.device)
        else:
            last_obs = torch.tensor(self.last_obs.clone().detach(), dtype=torch.float32)
            # print('ac',self.actions_cardinal, ', dim', last_obs.shape)
            last_obs = last_obs.repeat((self.actions_cardinal, 1, 1, 1)).to(self.device)
        if avl_actions==None: #avl_actions = self.actions
            avl_actions = torch.tensor(list(range(self.actions_cardinal)), dtype=torch.float32).unsqueeze(dim=-1)
        avl_actions = avl_actions.to(self.device)
        # breakpoint()  

        # epsilon-greedy strategy
        if self.exploration == 'eps_gr':
            prob = torch.rand(1,).item()
            eps = max(0.05, self.eps*(1-step/1e3))
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
        # print('updating')
        if self.obs_type!='image':
            nxt_obs = torch.tile(torch.tensor(next_obs, dtype=torch.float32), (self.actions_cardinal, 1)).to(self.device)
        else:
            nxt_obs = torch.tensor(next_obs, dtype=torch.float32)
            nxt_obs = nxt_obs.repeat((self.actions_cardinal, 1, 1, 1)).to(self.device)
        if avl_actions==None: # avl_actions = self.actions
            avl_actions = torch.tensor(list(range(self.actions_cardinal)), dtype=torch.float32).unsqueeze(dim=-1)
        avl_actions = avl_actions.to(self.device)
        # if terminal:
        #     v_ = 0
        # else:
        #     v_ = torch.max(self.q_f(nxt_obs, avl_actions))
        # print('evaluating target')
        v_ = (1-terminal) * (torch.max(self.q_f(nxt_obs, avl_actions).detach()))
        target = torch.tensor([reward], dtype=torch.float32).to(self.device) + self.gamma * v_
        # TD-error
        if self.obs_type!='image':
            last_obs = torch.tile(self.last_obs, (self.action_cardinal,1)).to(self.device)
            last_action = self.last_action.unsqueeze(dim=-1).unsqueeze(-1) if self.last_action.shape==torch.Size([]) else self.last_action.unsqueeze(-1)
        else:
            last_obs = self.last_obs.to(self.device)
            last_action = self.last_action.unsqueeze(dim=-1).unsqueeze(-1)
        last_action = last_action.to(self.device)
        # breakpoint()
        # print('evaluating current')
        loss = self.loss(self.q_f(last_obs, last_action), target)
        # print('backing up')
        torch.autograd.set_detect_anomaly(True)
        loss.backward()
        # print('stepping')
        self.optimizer.step()
        # print('stepped')

        # datum_tup = (self.last_obs.cpu().detach().numpy(), [self.last_action.cpu().detach().numpy()], [reward], next_obs, [terminal])
        # if terminal:
        #     for i in range(30): self.buffer.add(datum_tup)
        # self.buffer.add(datum_tup)
        # if self.iteration>1e4 and self.iteration%40==0:
        #     self.update_from_buffer(self.cfgs['ReplayBuffer']['update'])
    
    def update_from_buffer(self, sample_size):
        # print('from buffer')
        obs, action, reward, next_obs, terminal = self.buffer.sample(sample_size)
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        action = torch.tensor(action, dtype=torch.float32).to(self.device)
        if self.obs_type!='image':
            next_obs = torch.tensor(next_obs, dtype=torch.float32).to(self.device).unsqueeze(1).expand(-1, self.actions_cardinal, -1).reshape(-1, max(self.obs_size))
        else:
            next_obs = torch.tensor(next_obs, dtype=torch.float32).to(self.device).unsqueeze(1).expand(-1, self.actions_cardinal, -1, -1, -1)
            next_obs = next_obs.reshape(-1, next_obs.shape[2], next_obs.shape[3], next_obs.shape[4])
        # avl_actions = torch.tile(torch.tensor(list(range(self.actions_cardinal)), dtype=torch.float32), (int(reward.shape[0]), 1)).to(self.device).unsqueeze(-1).reshape(-1, self.action_cardinal).to(self.device)
        avl_actions = torch.tensor(list(range(self.actions_cardinal)), dtype=torch.float32).unsqueeze(dim=-1).to(self.device)
        # breakpoint()
        # print('evaluating target')
        # breakpoint()
        if self.obs_type!='image':
            v_ = (1 - torch.tensor(terminal, dtype=torch.float32).to(self.device)) * (torch.max(self.q_f(next_obs, avl_actions).detach().reshape((reward.shape[0], self.actions_cardinal, self.action_cardinal)), dim=1))[0]
        else:
            avl_actions = avl_actions.repeat(next_obs.shape[0]//self.actions_cardinal, 1)
            v_ = (1 - torch.tensor(terminal, dtype=torch.float32).to(self.device)) * (torch.max(self.q_f(next_obs, avl_actions).detach().reshape((reward.shape[0], self.actions_cardinal, self.action_cardinal)), dim=1))[0]
        target = torch.tensor(reward, dtype=torch.float32).to(self.device) + self.gamma * v_
        # print('evaluating current')
        loss = self.loss(self.q_f(obs, action), target.unsqueeze(0))
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
        # print(f'Agent details for the experiment will be saved at {exp_dir}')
        logging.info(f'Agent details for the experiment will be saved at {exp_dir}')
        with open(exp_dir+'/configs.yml', 'w') as configs:
            yaml.dump(self.cfgs, configs, default_flow_style=False)
        torch.save(self.q_f, f"{exp_dir}/q_f.pth")

    def load(self, path:str=None, **kwargs):
        if not os.path.exists(path+'/configs.yml'):
            logging.error(path+'/configs.yml not found, please make sure you have configs of the model to load')
            sys.exit(1)
        if not os.path.exists(path+'/q_f.pth'):
            logging.error(path+'/q_f.pth not found, please make sure you have the parameters of the learned agent')
            sys.exit(1)
        
        cfgs = read_configs(path+'/configs.yml')
        agent = self.__class__(cfgs, self.device)
        agent.q_f = torch.load(path+'/q_f.pth', weights_only=False)
        return agent





class QN(nn.Module):
    def __init__(self, obs_size, action_size, **kwargs):
        super().__init__()
        self.flat1 = nn.Linear(obs_size+action_size, 16)
        self.relu = nn.ReLU()
        self.flat2 = nn.Linear(16, 128)
        self.relu2 = nn.ReLU()
        self.flat3 = nn.Linear(128, 16)
        self.outl = nn.Linear(16, action_size)

    def forward(self, obs, action):
        # breakpoint()
        obs_action = torch.concatenate([obs, action], dim=-1)
        x = self.flat1(obs_action)
        # x = self.relu(x)
        x = self.flat2(x)
        # x = self.relu2(x)
        x = self.flat3(x)
        x = self.relu(x)
        out = self.outl(x)
        return out


class CQN(nn.Module):
    def __init__(self, input_observation_size, action_size, **kwargs):
        super().__init__()
        # self.actions_cardinality = len(action_size)
        self.input_size = input_observation_size

        # use a pretrained resnet preferably, flatten its output to action_cardinality size
        self.c1 = nn.Conv2d(3, 64, 3)
        self.c2 = nn.Conv2d(64, 128, 3, stride=2)
        self.c3 = nn.Conv2d(128, 256, 4, stride=3)
        self.c4 = nn.Conv2d(256, 512, 3, stride=2)
        self.c5 = nn.Conv2d(512, 1024, 3, stride=2)
        self.f1 = nn.Flatten(start_dim=1)
        # breakpoint()
        # breakpoint()
        # flat_size = self.f1(self.c3(self.c2(self.c1(torch.randn(self.input_size).permute(2,0,1).shape))))
        # flat_size = 35840
        flat_size = self.c5(self.c4(self.c3(self.c2(self.c1(torch.ones(self.input_size).permute(2,0,1)))))).numel()
        self.l1 = nn.Linear(flat_size, 2048)
        self.l2 = nn.Linear(2048, 127)
        self.l3 = nn.Linear(128, 32)
        self.l4 = nn.Linear(32, action_size)

    def forward(self, obs, action):
        # breakpoint()
        
        if len(obs.shape)<4: obs=obs.permute(2,0,1).unsqueeze(0)
        elif len(obs.shape)==4: obs=obs.permute(0,3,1,2)
        else: print('ERROR: check observation input, also make sure its representation is consistent with atari observations: (H*W*C) shape')
        x = self.c1(obs)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        # print('flat shape:', x.shape)
        # breakpoint()
        x = self.f1(x)
        x = self.l1(x)
        x = self.l2(x)
        # breakpoint()
        ot = self.l3(torch.concatenate((x, action), axis=1))
        ot = self.l4(ot)
        return ot