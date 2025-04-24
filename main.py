

import gymnasium as gym
import rlm.RL.DQN as dqn

from rlm.utils import get_cardinals

import time
import logging
from tqdm import tqdm


logging.basicConfig(level=logging.INFO)



import numpy as np
import torch

# np.random.seed(48)
# torch.manual_seed(48)


env = gym.make("MountainCar-v0", render_mode='rgb_array', goal_velocity=.1)

actions_size, action_size, obs_size = get_cardinals(env.action_space, env.observation_space)
agent = dqn(actions_size, action_size, obs_size, 'vec', discount=0.99, epsilon=1, device='cpu')

# breakpoint()
path = '/mnt/c/Users/2jeet/Desktop/Sujz/ml_projects/Modern_RL_slahan_/private_running_repo/Modern_RL_slahan_/data'

iterspepisode = []
episode = 1
episode_start = 0
collected_r = 0
obs, info = env.reset()
with tqdm(range(int(1e7))) as tkdm:
    for i in tkdm:
        tkdm.set_description(f'episode {episode}')
        action = agent.step(obs, i)
        obs, r, done, truncated, info = env.step(action)
        # collected_r += r
        done = (float(obs[0])>=0.5)
        agent.update(r, obs, done)
        if done: #or truncated:
            # termination = {done: 'done', truncated:'truncated'}.get(True)
            # breakpoint()
            # iterspepisode.append(info['episode_frame_number'])
            # iterspepisode.append(i-episode_start)
            if done:
                print(f'episode {episode}: reward collected in this episode was {collected_r} after running for {i+1-episode_start} steps')
                episode_start = i+1
                episode+=1
            # collected_r = 0
            obs, info = env.reset()
    agent.save(path)

# agent = rlm.RL.DQN(configs)