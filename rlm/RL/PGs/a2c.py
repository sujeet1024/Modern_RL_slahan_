
from rlm.Agent import Agent
from rlm.utils import read_configs, to_torch


import os
import sys
import yaml
import torch
import logging
import numpy as np
import torch.nn as nn




class A2C(Agent):
    ''' 
    Implemented for Continuous actions only (gaussian assumption as in a3c paper);
    Discrete case is simpler, for entropy use: sum pi log(pi), parameterizing pi output as action probs
    Advantage -> Q - V ~ r + γ V` - V ~ δ (td)
    here, A = R - V(s)
    With Synchronous Parallel Envs
    '''
    def __init__(self, cfgs=None, device=None, path=None, **kwargs):
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
            self.num_envs = self.cfgs['env']['nums']
            self.beta = self.cfgs['exploration']['param']['entropy_wt']
            # self.buffer = getattr(rbuffs, self.cfgs['ReplayBuffer']['name'])(capacity=self.cfgs['ReplayBuffer']['capacity'], removals=self.cfgs['ReplayBuffer']['remove'])

        else:
            self.actions_cardinal = kwargs['actions_shape']
            self.action_cardinal = kwargs['action_shape']
            self.obs_size = kwargs['obs_size']
            self.obs_type = kwargs['obs_type']
            self.gamma = kwargs['discount']
            self.eps = min(kwargs['epsilon'],1)
            self.exploration = kwargs['exploration']
            self.num_envs = kwargs['num_envs']
            self.beta = kwargs['entropy_wt']
            self.cfgs = None
            # from rlm.Agent.ReplayBuffers import RBuff
            # self.buffer = RBuff()
        self.sav_path = self.get_logs_dir(path) if path is not None else None
        self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.last_obs, self.last_action = None, None
        self.iteration = 0
        self.enc_size = 128 # lstm cells
        self.accumulation = False # to accumulate gradients and then update or a single large update
        self.actions_sampler = ActionsSampler()

        # self.pi = DiscrPi(self.obs_type, self.obs_size, self.action_cardinal, self.actions_cardinal)
        # self.enc = Encoder(self.obs_type, self.obs_size, self.enc_size).to(self.device)
        self.pi = Pi(self.obs_type, self.obs_size, self.action_cardinal, self.enc_size)
        self.vf = VN(self.obs_type, self.obs_size, self.enc_size)
        self.pi.to(self.device)
        self.vf.to(self.device)
        self.step_h, self.step_c = torch.zeros(1, self.num_envs, self.enc_size).to(self.device), torch.zeros(1, self.num_envs, self.enc_size).to(self.device)
        self.update_h, self.update_c = torch.zeros(1, self.num_envs, self.enc_size).to(self.device), torch.zeros(1, self.num_envs, self.enc_size).to(self.device)
        self.updatev_h, self.updatev_c = torch.zeros(1, self.num_envs, self.enc_size).to(self.device), torch.zeros(1, self.num_envs, self.enc_size).to(self.device)

        # self.enc_optimizer = torch.optim.Adam(self.enc.parameters(), 0.0001)
        self.pi_optimizer = torch.optim.Adam(self.pi.parameters(), 0.0001)
        self.vf_optimizer = torch.optim.Adam(self.vf.parameters(), 0.0001)
        # self.pi_optimizer = torch.optim.Adam([*self.pi.parameters(), *self.enc.parameters()], 0.0001)
        # self.vf_optimizer = torch.optim.Adam([*self.vf.parameters(), *self.enc.parameters()], 0.0001)

    def step(self, obs, iteration, dones, avl_actions=None, **kwargs):
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device).unsqueeze(1).unsqueeze(0)
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            mu, sigm2, self.step_h, self.step_c = self.pi(obs, self.step_h, self.step_c)
            self.step_h, self.step_c = (1-dones) * self.step_h, (1-dones) * self.step_c
        action = self.actions_sampler.step(mu, sigm2)
        if self.action_cardinal==1:
            self.last_action = action.detach().cpu().squeeze(-1)
        else:
            self.last_action = action.detach().cpu().squeeze().numpy()
        self.last_obs = obs
        self.iteration+=1
        return self.last_action

    def update(self, traj, avl_actions=None, **kwargs):
        # traj: list of interations [s, a, r, done], lists of individual
        obsss, actionss, rewss, doness, truncatedss = traj
        obsss, actionss, rewss, doness, truncatedss = to_torch(np.stack(obsss), self.device), to_torch(np.stack(actionss), self.device), to_torch(np.stack(rewss), self.device), to_torch(np.stack(doness), self.device), to_torch(np.stack(truncatedss), self.device)

        if len(obsss)<3:
            truncatedss = truncatedss.unsqueeze(-1)
            self.update_h, self.update_c = (1-truncatedss[-1]) * self.update_h.detach(), (1-truncatedss[-1])*self.update_c.detach()
            self.updatev_h, self.updatev_c = (1-truncatedss[-1]) * self.updatev_h.detach(), (1-truncatedss[-1])*self.updatev_c.detach()
            return
            
        with torch.no_grad():
            vfs, _, _ = self.vf(obsss, self.updatev_h, self.updatev_c)

        doness = doness.squeeze()
        Rewss = [(1-doness[-1]) * vfs[:,-1,:].detach().squeeze()] # reverse the order
        removal_indices = [len(rewss)-1] # shouldnt matter if empty
        for i in range(len(rewss)-2, -1, -1):
            Rewss.insert(0, rewss[i]+(1-doness[i])*self.gamma*Rewss[-1])
            if 1 in doness[i]: removal_indices.append(i)
        mask = torch.ones(obsss.shape[0], dtype=torch.bool)
        mask[removal_indices] = False
        obsss = obsss[mask,:,:]
        actionss = actionss[mask,:,:]

        Rewss = torch.stack(Rewss).permute(1,0)
        Rewss = Rewss[:, mask]
        mus, sigm2s, self.update_h, self.update_c = self.pi(obsss, self.update_h, self.update_c)

        log_probs, sigm2s = self.actions_sampler.step(mus, sigm2s, actionss)

        with torch.no_grad():
            vfs, _, _ = self.vf(obsss, self.updatev_h, self.updatev_c)
        A = Rewss - vfs.detach().squeeze()
        # a_pen = torch.relu(mus.abs()-1).pow(2)
        H = (1/2 * torch.log(2*torch.pi*sigm2s)+1).sum(dim=-1)
        loss_pi = - (log_probs.squeeze() * A + self.beta * H).mean() #+ a_pen.mean()
        # self.enc_optimizer.zero_grad()
        self.pi_optimizer.zero_grad()
        self.vf_optimizer.zero_grad()
        loss_pi.backward(retain_graph=True)
        self.pi_optimizer.step()
        vfs, self.updatev_h, self.updatev_c = self.vf(obsss, self.updatev_h, self.updatev_c)
        loss_vf = ((Rewss - vfs.squeeze())**2).mean()
        loss_vf.backward()
        self.vf_optimizer.step()
        # total_loss = loss_pi + loss_vf
        # total_loss.backward()
        # self.enc_optimizer.step()

        truncatedss = truncatedss.unsqueeze(-1)
        self.update_h, self.update_c = (1-truncatedss[-1]) * self.update_h.detach(), (1-truncatedss[-1])*self.update_c.detach()
        self.updatev_h, self.updatev_c = (1-truncatedss[-1]) * self.updatev_h.detach(), (1-truncatedss[-1])*self.updatev_c.detach()
        print(f'\nvf loss: {loss_vf.detach().cpu().item():.4f}', f'\t pi_loss: {loss_pi.detach().cpu().item():.4f}', 
            #   f'\t sigm2: {sigm2s.detach().cpu().mean().item():.4f}', 
              f'\t Adv_mean: {A.cpu().mean().item():.4f}',
              f'\t actss_mean: {actionss.cpu().mean().item():.4f}', f' actss_max: {actionss.cpu().max().item():.4f}',
            #   f'\t H mean: {H.cpu().mean().item():.4f}', 
              f'\t log_probs mean: {log_probs.cpu().mean().item():.4f}',
              f'\t mus mean: {mus.detach().cpu().mean().item():.4f}', f' mus max: {mus.detach().cpu().max().item():.4f}',
              f' mus min: {mus.detach().cpu().min().item():.4f}')

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
        torch.save(self.qf, f"{exp_dir}/qf.pth")

    def get_logs_dir(self, path=None):
        if path is None: return
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
        if not os.path.exists(path+'/qf.pth'):
            logging.error(path+'/qf.pth not found, please make sure you have the parameters of the learned agent')
            sys.exit(1)
        
        cfgs = read_configs(path+'/configs.yml')
        agent = self.__class__(cfgs, self.device) #TODO fix this, why create another instance
        agent.sav_path = self.sav_path
        agent.pi = torch.load(path+'/pi.pth', weights_only=False).to(self.device)
        agent.qf = torch.load(path+'/qf.pth', weights_only=False).to(self.device)
        return agent



class Encoder(nn.Module):
    def __init__(self, obs_type, obs_size, enc_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if obs_type == 'vec':
            lstm_in = 200
            self.pre_enc = VecEnc(obs_size[0], lstm_in)
        elif obs_type == 'image':
            self.pre_enc = PixEnc(obs_size)
            lstm_in = self.pre_enc.lstm_in
        # self.lstm = nn.LSTM(lstm_in, enc_size, batch_first=True)#, bias=False)
        self.l2 = nn.Linear(lstm_in, enc_size)

    def forward(self, obs, h, c):
        x = self.pre_enc(obs)
        if len(x.shape)==2: x = x.unsqueeze(1)
        # out, (h, c) = self.lstm(x, (h, c))
        out = self.l2(x)
        return out, h, c

class Pi(nn.Module):
    def __init__(self, obs_type, obs_size, action_cardinal, enc_size, encoder=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enc = encoder if encoder else Encoder(obs_type, obs_size, enc_size)
        self.l_mu = nn.Linear(enc_size, action_cardinal)
        self.r_mu = nn.LeakyReLU()
        self.l_sgm = nn.Linear(enc_size, action_cardinal)
        self.r_sgm = nn.LeakyReLU()

    def forward(self, obs, h, c):
        enc, h,c = self.enc(obs, h, c)
        mu = self.r_mu(self.l_mu(enc))
        sgm2 = nn.functional.softplus(self.r_sgm(self.l_sgm(enc)))
        return mu, sgm2, h, c

class VN(nn.Module):
    def __init__(self, obs_type, obs_size, enc_size, encoder=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enc = encoder if encoder else Encoder(obs_type, obs_size, enc_size)
        self.l1 = nn.Linear(enc_size, 1)
        self.r1 = nn.LeakyReLU()

    def forward(self, obs, h, c):
        enc, h, c = self.enc(obs, h, c)
        x = self.r1(self.l1(enc))
        return x, h, c

class VecEnc(nn.Module):
    def __init__(self, obs_size, lstm_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.l1 = nn.Linear(obs_size, lstm_size)
        self.r1 = nn.LeakyReLU()
        # self.l2 = nn.Linear(32, lstm_size)
        # self.r2 = nn.LeakyReLU()

    def forward(self, obs):
        if len(obs.shape)==3:obs = obs.permute(1,0,2)
        out = self.r1(self.l1(obs))
        # out = self.r2(self.l2(out))
        return out

class PixEnc(nn.Module):
    def __init__(self, obs_size, lstm_size=None, *args, **kwargs):
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
        flat_size = self.relu3(self.conv3(self.bnorm3(self.relu2(self.conv2(self.bnorm2(self.pool1(self.relu1(self.conv1(torch.zeros(obs_size).permute(2,0,1)))))))))).numel()
        self.lstm_in = flat_size
        # self.bnorm4 = nn.BatchNorm1d(flat_size)
        # self.flat4 = nn.Linear(flat_size, lstm_size)

    def forward(self, obs):
        x = self.pool1(self.relu1(self.conv1(obs)))
        x = self.relu2(self.conv2(self.bnorm2(x)))
        x = self.relu3(self.conv3(self.bnorm3(x)))
        x = self.relu4(self.flat4(self.bnorm4(x)))
        return x        


class ActionsSampler:
    def __init__(self):
        pass

    def step(self, mus:torch.Tensor, sigma2s:torch.Tensor, actions:torch.Tensor=None):
        sigma2s = torch.clamp(sigma2s, min=1e-5, max=0.075)
        dist = torch.distributions.Normal(mus, torch.sqrt(sigma2s))
        if actions is None:
            action = dist.sample()
            return action
        actions = actions.permute(1,0,2)
        log_prob = dist.log_prob(actions)
        return log_prob.sum(dim=-1, keepdim=True), sigma2s