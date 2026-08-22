import numpy as np
import pygame as pg
import gymnasium as gym
from gymnasium import spaces

from rlm.Env import Environment




class CityParking(Environment, gym.Env):
    def __init__(self):
        super().__init__()
        self.state = None
        self.actions = []

    def start(self,):
        pass

    def step(self,):
        pass
        return obs, r, done, truncated, info

    def reset(self,):
        pass
        return obs, info