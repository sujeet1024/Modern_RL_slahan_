from .base import DPolicy

import numpy as np


class UCB(DPolicy):
    def __init__(self, cfgs=None, **kwargs):
        super().__init__()
        self.step = 0

        if cfgs is not None:
            self.cfgs = cfgs
            self.obs_space = self.cfgs['env']['obs_space']
            self.actions_cardinal = self.cfgs['env']['actions_shape']
        else:
            self.obs_space = kwargs['obs_space']
            self.actions_cardinal = kwargs['actions_shape']
        obs_indices = tuple((np.array(self.obs_space['high']) - np.array(self.obs_space['low']) + 1).astype(np.int32).tolist())
        self.counts_s = np.ones(obs_indices)
        self.counts_a = np.ones((*obs_indices, self.actions_cardinal))

    def step(self, obs, avl_actions, action_potentials, **kwargs):
        last_obs = tuple(obs.astype(np.int32).tolist())
        self.counts_s[*last_obs]+=1
        action = np.argmax(action_potentials + \
                    np.sqrt(np.log(self.counts_s[*last_obs])/self.counts_a[*last_obs]))
        self.counts_a[*last_obs, action]+=1
        return int(action)
    
    def set_val(self, obs, action):
        pass


class Greedy(DPolicy):
    def __init__(self, **kwargs):
        super().__init__()

    def step(self, obs, avl_actions, action_potentials, **kwargs):
        return avl_actions[np.argmax(action_potentials)]
    
    def set_val(self, obs, action):
        pass


class IterPolicy(DPolicy):
    def __init__(self, cfgs=None, **kwargs):
        super().__init__(**kwargs)
        if cfgs is not None:
            self.cfgs = cfgs
            self.obs_space = self.cfgs['env']['obs_space']
            self.actions_cardinal = self.cfgs['env']['actions_shape']
        else:
            self.obs_space = kwargs['obs_space']
            self.actions_cardinal = kwargs['actions_shape']
        obs_indices = tuple((np.array(self.obs_space['high']) - np.array(self.obs_space['low']) + 1).astype(np.int32).tolist())
        self.counts_s = np.ones(obs_indices)

        self.policy = np.random.randint(low=0, high=self.actions_cardinal-1, size=obs_indices)
        self.iterations = 0

    def step(self, obs):
        last_obs = tuple(np.array(obs).astype(np.int32).tolist())
        return int(self.policy[*last_obs])

    def set_val(self, obs, action):
        last_obs = tuple(np.array(obs).astype(np.int32).tolist())
        self.policy[*last_obs] = action
        pass

    def save(self, path, **kwargs):
        raise NotImplementedError

    def load(self, path, **kwargs):
        raise NotImplementedError