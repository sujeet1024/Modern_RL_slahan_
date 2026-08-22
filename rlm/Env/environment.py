
from abc import ABC, abstractmethod





# this class here should inherit from gym env class, or remove Env package and every env should inherit from gym env only
class Environment(ABC):
    def __init__(self, **kwargs):
        self.state = None
        self.actions = []
        pass
    
    @abstractmethod
    def start(self, hyper_params, render:bool):
        ''' Start the environment with a particular setup '''
        reward = 0.0                # No reward is recieved on first observation, action yet to be taken
        terminal = False
        return self.state, reward, terminal

    @abstractmethod
    def step(self, agent_action):
        reward = self.feedback(agent_action)
        self.state, terminal = self.transition(agent_action)
        return self.state, reward, terminal
    
    @abstractmethod
    def reset(self, last:bool):
        ''' Resets the environment after termination of an episode
        Parameters:
        last (bool): If you want to retain environment setting from last episode
        '''
        self.state = None

    @abstractmethod
    def render(self):
        ''' Visualize environment state (including agent) '''
        pass

    @abstractmethod
    def vis_policy(self, policy):
        ''' Visualize policy: snapshot for grid env, gif/video for large env '''
        pass


class TransitionModel(ABC):
    def __init__(self, **kwargs):
        super().__init__()

    @abstractmethod
    def transition(self, obs, action, **kwargs):
        # return list[rewards], list[next_obs], list[probs]
        pass