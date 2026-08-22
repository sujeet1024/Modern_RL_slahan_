from abc import abstractmethod, ABC
import numpy as np

from rlm.Agent import ActionPolicy


class DPolicy(ActionPolicy):
    ''' Deterministic Policy '''
    def __init__(self):
        super().__init__()

    @abstractmethod
    def step(self, obs=None, avl_actions=None, action_potentials=None, **kwargs):
        pass

    def propose(self, obs=None, avl_actions=None, action_potentials=None, **kwargs):
        action = self.step(obs, avl_actions, action_potentials)
        probs = np.zeros_like(avl_actions)
        probs[action] = 1
        return probs

    @abstractmethod
    def set_val(self, obs, action):
        pass


class SPolicy(ActionPolicy):
    ''' Stochastic Policy '''
    def __init__(self):
        super().__init__()

    @abstractmethod
    def step(self, obs=None, avl_actions=None, action_potentials=None, **kwargs):
        pass

    @abstractmethod
    def propose(self, obs=None, avl_actions=None, action_potentials=None, **kwargs):
        pass