from abc import ABC, abstractmethod




class Agent(ABC):
    def __init__(self):
        self.actions = None
        self.vf = None
        self.qf = None
        self.pi = None
        self.env = None
        self.sav_path = None
    
    # @abstractmethod
    # def start(self, **kwargs):
    #     ''' 
    #     In case we want to keep some parameters same, and try out experience from last learning
    #     last_v:bool, last_q:bool, last_pi:bool 
    #     -> this would only make sense if actions still mean the same things, even if they dont, semantics of actions should be parsed by FA
    #     '''
    #     self.env = kwargs['env']

    @abstractmethod
    def step(self, state, actions=None):
        # take an action in the environment state
        pass

    @abstractmethod
    def update(self, rew, next_obs):
        # update policy/value
        pass

    @abstractmethod
    def save(self, **kwargs):
        # save the parameters of the learned agent in a specific directory with 
        # config.yml specifying what env, algorithm, parameters the agent is for
        pass

    @abstractmethod
    def load(self, **kwargs):
        # load the agent as saved using save method
        pass


class ActionPolicy(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def step(self, obs, avl_actions=None, action_potentials=None, **kwargs):
        pass