

import ale_py
import gymnasium as gym
from rlm.RL.DVFs import DQN, DDQN
# from rlm.RL.Classic import QLearning, Sarsa, ESarsa, PolicyIter, ValueIter
# from rlm.RL.Classic.action_policies.determ import Greedy, UCB
# from rlm.RL.Classic.action_policies.stoch import Boltzman, eGreedy, URandom
from rlm.RL.PGs import REINFORCE, AcCritic, OffPAC, SAC, A2C
from rlm.RL.DPGs import DDPG, TD3

from rlm.utils import get_cardinals, read_configs, ObsHist, CustomLogger
from rlm import tabular_envs

import time
import logging
from tqdm import tqdm


logging.basicConfig(level=logging.INFO)


import numpy as np
import torch





def train_stepwise(agent, env, logger, obss, bpolicy=None, tpolicy=None):
    iterspepisode = []
    episode = 1
    episode_start = 0
    collected_r = 0
    max_r = float('-inf') if (not 'max_reward' in cfgs) or (not cfgs['max_reward']) else cfgs['max_reward']
    best_ep = max_r
    obs, info = env.reset()
    # obss.intialize(obs)
        
    with tqdm(range(int(2e6))) as tkdm:
        for i in tkdm:
            tkdm.set_description(f'episode {episode}')
            tkdm.set_postfix({'reward': collected_r})
            action = agent.step(obs, i, bpolicy=policy)
            # action = agent.step(obss.get(), i, bpolicy=policy)
            for j in range(cfgs['env']['sticky_actions']):
                obs, r, done, truncated, info = env.step(action)
                # obss.add(obs)
                if done or truncated: break
            agent.update(r, obs, done, tpolicy=policy)
            # agent.update(r, obss.get(), done, tpolicy=policy)
            collected_r += r
            if done or truncated:
                # # termination = {done: 'done', truncated:'truncated'}.get(True)
                # # iterspepisode.append(info['episode_frame_number'])
                # # iterspepisode.append(i-episode_start)
                # if done:
                if i+1-episode_start>best_ep:
                    best_ep=i+1-episode_start
                    logger.append(f'episode {episode}: reward collected in this episode was {collected_r} after running for {i+1-episode_start} steps\n')
                    # print(f'episode {episode}: reward collected in this episode was {collected_r} after running for {i+1-episode_start} steps')
                elif collected_r>=max_r:
                    logger.append(f'episode {episode}: reward collected in this episode was {collected_r} after running for {i+1-episode_start} steps\n')
                if collected_r>=max_r:
                    max_r = collected_r
                    agent.save(path)
                episode_start = i+1
                episode+=1
                collected_r = 0
                obs, info = env.reset()
                # obss.add(obs)
            # if configs.vis_policy:
                # env.unwrapped.vis_policy(agent.policy)
        agent.cfgs['max_reward'] = max_r
        agent.cfgs['episodes']+=episode
        agent.save(path)


def train_episodewise(agent, env, logger, obss, bpolicy=None, tpolicy=None):
    episode = 0
    max_r = float('-inf') if not 'max_reward' in cfgs else cfgs['max_reward']
    best_ep = max_r
    
    with tqdm(range(int(1e4))) as tkdm:
        for ep in tkdm:
            tkdm.set_description(f'episode {ep+1}')
            ep_end = False
            last_obs, info = env.reset()
            traj = []
            t = 0; collected_r = 0
            while not ep_end:
                action = agent.step(last_obs, t)
                obs, r, done, truncated, info = env.step(action)
                traj.append((last_obs, action, r, t))
                collected_r+=r
                t+=1; last_obs = obs
                if done or truncated:
                    episode+=1; ep_end = True
                tkdm.set_postfix({'steps': t})
            agent.update(traj)
            if collected_r>=max_r:
                logger.append(f'episode {episode}: reward collected in this episode was {collected_r} after running for {t} steps\n')
                max_r = collected_r
                agent.save(path)
                agent.cfgs['max_reward'] = max_r
                agent.cfgs['episodes']+=episode
        agent.save(path)


def train_parallel(agent, env_name, logger, obss, num_envs:int=16, bpolicy=None, tpolicy=None, render='rgb_array', update_iter=7):
    max_r = float('-inf') if (not 'max_reward' in cfgs) or (not cfgs['max_reward']) else cfgs['max_reward']
    mean_collected_r = 0
    episode, episode_start = 1, 0
    # def make_env():
    #     nonlocal env_name, render
    #     return gym.make(env_name)
    # envs = gym.vector.AsyncVectorEnv([make_env for _ in range(num_envs)])
    envs = gym.make_vec(env_name, num_envs)
    collected_rs = np.zeros(num_envs)
    dones = [False for i in range(num_envs)]
    obsss, actionss, rewss, doness, truncatedss = [],[],[],[],[] #[False for f in range(num_envs)]
    last_obss, infos = envs.reset()
    with tqdm(range(int(3e6))) as tkdm:
        for iter in tkdm:
            tkdm.set_description(f'episodes {episode}')
            tkdm.set_postfix({'reward': f'{collected_rs.mean():.4f}'})
            actions = agent.step(last_obss, iter, dones)
            obss, rs, dones, truncateds, infos = envs.step(actions)
            obsss.append(last_obss)
            actionss.append(actions)
            rewss.append(rs)
            doness.append(dones)
            truncatedss.append(truncateds)
            last_obss = obss
            if not update_iter:
                if (1 in dones) or (1 in truncateds):
                    agent.update((obsss, actionss, rewss, doness, truncatedss))
                    obsss, actionss, rewss, doness = [],[],[],[]
            elif ((iter+1) % update_iter == 0) or (1 in dones) or (1 in truncateds):
                agent.update((obsss, actionss, rewss, doness, truncatedss))
                obsss, actionss, rewss, doness = [],[],[],[]
            collected_rs += rs
            mean_collected_r = collected_rs.mean()
            if (1 in dones) or (1 in truncateds):
                if mean_collected_r>=max_r:
                    logger.append(f'episode {episode}: reward collected in this episode was {mean_collected_r} after running for {iter+1-episode_start} steps\n')
                    max_r = mean_collected_r
                    agent.save(path)
                collected_rs = np.zeros(num_envs)
                episode += num_envs
                episode_start = iter
        agent.cfgs['max_reward'] = max_r
        agent.cfgs['episodes']+=episode
        agent.save(path)


def test(agent, env, exp_dir):
    agent.load(exp_dir)
    G, iter = 0, 0
    done, truncated = False, False
    obs, info = env.reset()
    # while not done or truncated:
    #     action = agent.step(obs, iter)
    #     obs, r, done, truncated, info = env.step(action)
    #     G += r
    #     iter+=1
    #     if iter>1000: truncated=True
    for i in range(1000):
        action = agent.step(obs, iter)
        obs, r, done, truncated, info = env.step(action)
        G += r
    print("After end of testing an episode the collected reward was:", G)




if __name__=="__main__":

    path = 'data'
    cfgs = read_configs(path+'/configs.yml')

    np.random.seed(cfgs['seed'])
    torch.manual_seed(cfgs['seed'])

    rendering = ['rgb_array', 'human']

    # env = gym.make("MountainCar-v0", render_mode='rgb_array', goal_velocity=.1)
    # env = gym.make('CartPole-v1', render_mode='human')
    env = gym.make(cfgs['env']['name'], render_mode=rendering[cfgs['env']['render']])
    # env = gym.make('Breakout-v0', render_mode=rendering[cfgs['env']['render']])
    # env = gym.make('ClifWalking-v0', render_mode='human')
    # env = gym.make("BipedalWalker-v3", render_mode='human')
    # env = gym.make("InvertedPendulum-v5", render_mode='human')
    # env = gym.make("Pendulum-v1", render_mode='human')
    # env = gym.make("HalfCheetah-v5", render_mode='rgb_array')


    cfgs = get_cardinals(env.action_space, env.observation_space, cfgs)

    # policy = UCB(cfgs)
    # spolicy = eGreedy(0.1)
    policy = None

    # agent = DQN(cfgs, device='cuda')
    # agent = SAC(cfgs, device='cuda', path=path)
    agent = A2C(cfgs, device='cuda', path=path)
    # agent = OffPAC(cfgs, device='cuda')
    # agent = TD3(cfgs, device='cuda')
    obss = ObsHist(cfgs)
    # agent = AcCritic(cfgs, device='cuda')
    # agent = QLearning(cfgs, device='cuda')
    # agent = ESarsa(cfgs, device='cuda')
    # agent = ValueIter(cfgs, device='cuda', model=env.unwrapped.model)
    # agent = agent.load(path+'/exp_000008')
    # cfgs = agent.cfgs


    logger = CustomLogger(agent.sav_path+'/last_log.txt')
    logger.write(f"logs for experiment {cfgs['name']}, with agent {cfgs['agent']['name']} in environment {cfgs['env']['name']}\n\n")
    logging.info(f"running experiment {cfgs['name']}, with agent {cfgs['agent']['name']} in environment {cfgs['env']['name']}")
    obs, info = env.reset()


    # train_stepwise(agent, env, logger, obss, bpolicy=None, tpolicy=None)
    # train_stepwise(agent, env, logger, obs, bpolicy=None, tpolicy=None)
    # train_episodewise(agent, env, logger, obss, bpolicy=None, tpolicy=None)
    train_parallel(agent, cfgs['env']['name'], logger, obss, num_envs=cfgs['env']['nums'], render=rendering[cfgs['env']['render']], update_iter=10)
    # test(agent, env, exp_dir=path+'/exp_000004')