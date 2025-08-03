

import ale_py
import gymnasium as gym
from rlm.RL.DQN.dqn import DQN

from rlm.utils import get_cardinals, read_configs

import time
import logging
from tqdm import tqdm


logging.basicConfig(level=logging.INFO)



import numpy as np
import torch



path = '/mnt/c/Users/2jeet/Desktop/Sujz/ml_projects/Modern_RL_slahan_/private_running_repo/Modern_RL_slahan_/data'
cfgs = read_configs(path+'/configs.yml')

# np.random.seed(cfgs['seed'])
# torch.manual_seed(cfgs['seed'])

rendering = ['rgb_array', 'human']

# env = gym.make("MountainCar-v0", render_mode='rgb_array', goal_velocity=.1)
# env = gym.make('Taxi-v3', render_mode='rgb_array')
env = gym.make(cfgs['env']['name'], render_mode=rendering[cfgs['env']['render']])

# actions_size, action_size, obs_size = get_cardinals(env.action_space, env.observation_space)

cfgs = get_cardinals(env.action_space, env.observation_space, cfgs)

agent = DQN(cfgs, device='cuda')
# breakpoint()
# agent = agent.load(path+'/exp_000008')
# cfgs = agent.cfgs

# breakpoint()

iterspepisode = []
episode = 1
episode_start = 0
max_r = float('-inf') if not 'max_reward' in cfgs else cfgs['max_reward']
# max_r = cfgs['max_reward']
collected_r = 0
best_ep = -max_r
# breakpoint()
obs, info = env.reset()
with tqdm(range(int(7e5))) as tkdm:
    for i in tkdm:
        # breakpoint()
        tkdm.set_description(f'episode {episode}')
        action = agent.step(obs, i)
        obs, r, done, truncated, info = env.step(action)
        collected_r += r
        # done = (float(obs[0])>=0.5)
        agent.update(r, obs, done)
        if done: #or truncated:
            # termination = {done: 'done', truncated:'truncated'}.get(True)
            # breakpoint()
            # iterspepisode.append(info['episode_frame_number'])
            # iterspepisode.append(i-episode_start)
            if done:
                if i+1-episode_start<best_ep:
                    best_ep=i+1-episode_start
                    print(f'episode {episode}: reward collected in this episode was {collected_r} after running for {i+1-episode_start} steps')
                if collected_r>max_r:
                    max_r = collected_r
                episode_start = i+1
                episode+=1
            collected_r = 0
            obs, info = env.reset()
    # breakpoint()
    agent.cfgs['max_reward'] = max_r
    agent.cfgs['episodes']+=episode
    agent.save(path)

# agent = rlm.RL.DQN(configs)