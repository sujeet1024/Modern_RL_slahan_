from abc import ABC, abstractmethod




class Agent(ABC):
    def __init__(self):
        self.actions = None
        self.v_f = None
        self.q_f = None
        self.pi = None
        self.env = None
    
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
        pass

    @abstractmethod
    def update(self, reward, next_state):
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