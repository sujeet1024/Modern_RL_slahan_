from rlm.Agent import Agent
from rlm.Env import TransitionModel
from .action_policies.determ import DPolicy, IterPolicy


import numpy as np
from itertools import product


## YOU NEED MODEL


class ValueIter(Agent):
    ''' Model Based (no on/off policy) '''
    def __init__(self, cfgs, device, model:TransitionModel, theta=1e-5, **kwargs):
        super().__init__()
        self.model = model
        self.theta = theta

        if cfgs is None:
            cfgs = {'env': {}, 'agent': {}, 'seed':0, 'exploration':{}, 'max_reward':float('-inf')}
            cfgs['env']['actions_shape'] = kwargs['actions_shape']
            cfgs['env']['action_shape'] = kwargs['action_shape']
            cfgs['env']['action_space'] = kwargs['action_space']
            cfgs['env']['obs_size'] = kwargs['obs_size']
            cfgs['env']['obs_type'] = kwargs['obs_type']
            cfgs['env']['obs_space'] = kwargs['obs_space']
            cfgs['agent']['discount'] = kwargs['discount']
            cfgs['exploration']['param']['eps'] = min(kwargs['epsilon'],1)
            cfgs['exploration']['name'] = kwargs['exploration']
        cfgs['agent']['theta'] = theta

        self.cfgs = cfgs
        self.actions_cardinal = self.cfgs['env']['actions_shape']
        self.action_cardinal = self.cfgs['env']['action_shape']
        self.action_space = self.cfgs['env']['action_space']
        self.obs_size = self.cfgs['env']['obs_size']
        self.obs_type = self.cfgs['env']['obs_type']
        self.obs_space = self.cfgs['env']['obs_space']
        self.gamma = self.cfgs['agent']['discount']
        self.exploration = self.cfgs['exploration']['name']
        self.eps = min(abs(self.cfgs['exploration']['param']['eps']), 1)

        obs_indices = tuple((np.array(self.obs_space['high']) - np.array(self.obs_space['low']) + 1).astype(np.int32).tolist())
        self.o_space = list(product(*(range(n) for n in obs_indices)))
        self.avl_actions = list(range(self.actions_cardinal))
        self.v_f = np.zeros(obs_indices)

        self.policy = IterPolicy(cfgs)
        # policy assumed deterministic and model stochastic

    def step(self, obs, iter, policy:DPolicy=None, **kwargs):
        return int(np.random.choice(self.avl_actions))

    def update(self, r, obs, done, policy:DPolicy=None, **kwargs):
        # sweep, # update policy
        is_delta_big = True
        while is_delta_big:
            delta = 0
            for obs in self.o_space:
                v, v_s = self.v_f[obs], 0
                q_sa = []
                for action in self.avl_actions:
                    rewards, next_obss, probs = self.model.transition(obs, action)
                    v_s = self.eval_vs(rewards, next_obss, probs)
                    q_sa.append(v_s)
                self.v_f[obs] = max(q_sa)
                delta = max(delta, abs(v-self.v_f[obs]))
            if delta<self.theta:
                is_delta_big = False
        for obs in self.o_space:
            q_sa = []
            for action in self.avl_actions:
                rewards, next_obss, probs = self.model.transition(obs, action)
                v_s = self.eval_vs(rewards, next_obss, probs)
                q_sa.append(v_s)
            n_action = int(np.argmax(q_sa))
            self.policy.set_val(obs, n_action)

    def eval_vs(self, rewards, next_obss, probs):
        v_s = 0
        for rsa, s_, psr in zip(rewards, next_obss, probs):
            s_ = tuple(s_.astype(np.int32).tolist())
            v_s += psr * (rsa + self.gamma * float(self.v_f[s_]))
        return v_s

    def save(self, policy, path:str=None, **kwargs):
        ''' save both, policy and corresponding v_f '''
        policy.save(path, **kwargs)
        pass

    def load(self, policy, path:str=None, **kwargs):
        ''' load both, policy and corresponding v_f '''
        policy.load(path, **kwargs)
        pass