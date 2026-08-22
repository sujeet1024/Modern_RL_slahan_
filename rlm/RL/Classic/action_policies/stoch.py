from .base import SPolicy

import numpy as np




class eGreedy(SPolicy):
    def __init__(self, eps, **kwargs):
        super().__init__()
        self.eps = eps
        self.iter = 0

    def step(self, obs, avl_actions:np.ndarray, action_potentials:np.ndarray, **kwargs):
        self.iter += 1
        prob = np.random.uniform(0.0, 1.)
        if prob < self.eps:
            return np.random.choice(avl_actions)
        else:
            return avl_actions[np.argmax(action_potentials)]

    def propose(self, obs, avl_actions, action_potentials, **kwargs):
        probs = np.ones_like(avl_actions)*self.eps/len(avl_actions)
        probs[np.argmax(action_potentials)] += 1-self.eps
        return probs
    

class Boltzman(SPolicy):
    ''' Boltzman/Softmax/Gibbs/Categorical(distr using gen probs) '''
    def __init__(self, **kwargs):
        super().__init__()
        self.iter = 0

    def step(self, obs, avl_actions:np.ndarray, action_potentials:np.ndarray, **kwargs):
        self.iter += 1
        probs = np.exp(action_potentials)/np.exp(action_potentials).sum()
        return np.random.choice(avl_actions, p=probs)
    
    def propose(self, obs=None, avl_actions=None, action_potentials=None, **kwargs):
        probs = np.exp(action_potentials)/np.exp(action_potentials).sum()
        return probs
    

class URandom(SPolicy):
    ''' Uniform Random '''
    def __init__(self, **kwargs):
        super().__init__()

    def step(self, obs=None, avl_actions=None, action_potentials=None, **kwargs):
        return np.random.choice(avl_actions)
    
    def propose(self, obs=None, avl_actions=None, action_potentials=None, **kwargs):
        return np.ones_like(avl_actions)/len(avl_actions)