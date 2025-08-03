
from abc import ABC, abstractmethod





# this class here should inherit from gym env class, or remove Env package and every env should inherit from gym env only
class Environment(ABC):
    def __init__(self):
        self.state = None
        self.actions = []
        pass
    
    @abstractmethod
    def start(self, hyper_params, render:bool):
        ''' Start the environment with a particular setup '''
        reward = 0.0                # No reward is recieved on first observation, action yet to be taken
        terminate = False
        return self.state, reward, terminate

    @abstractmethod
    def step(self, agent_action):
        reward = self.feedback(agent_action)
        self.state = self.transition(agent_action)
        terminate = self.is_terminal()
        return self.state, reward, terminate
    
    @abstractmethod
    def reset(self, last:bool):
        ''' Resets the environment after termination of an episode
        Parameters:
        last (bool): If you want to retain environment setting from last episode
        '''
        self.state = None

    @abstractmethod
    def is_terminal(self):
        ''' Checks in transition state is terminal '''
        self.state
        self.reset()

    def action_space(self):
        return self.actions