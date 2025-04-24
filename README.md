# Modern_RL_slahan_
(An Overview of Reinforcement Learning Algorithms)

This is an ongoing project,
In this repo, the author presents implementations of popular or important (modern/) Reinforcement Learning algorithms in various gymnasium, Atari2600 environments and a sawyer (grasping) environment, all in simulation.
Along with their implementations the author reports their performances and insights on different algorithms, also discussing the settings of optimal performance amongst the environments experimented.

The author also provides a short background on classical RL (from Sutton and Barto)

This project shall cover,
- Important (foundational) Algorithms
    - Q-learning/ value iters (2D, finite states env)
    - DQN
    - Double DQN
    - REINFORCE
    - Actor Critic (reinforce with baselines) 
    - Soft Actor Critic
    - A2C
    - A3C
    - PPO
    - TRPO
    - Natural Gradients

- OFFLINE-RL (Data from expert online learnt agents for offline RL was used)
    - C-learning
    - Implicit Q-Learning
    - Advantage Weighted Actor Critic (AWAC)
    - COMBO
    - Trajectory Transformer

- Exploration strategies
    - count based (Optimistic)
    - max ent (Information Gain)
    - Thompson sampling (Probability Matching, Posterior Sampling)

- Other methods (Supervised Imitation Learning)
    - Decision Transformer
    - Diffusion Policies
    Although the author doesn't consider these as RL or DeepRL methods, but in light of buzz around those their comparisons are also provided

All the environments used here are directly taken from gymnasium or are built on top of gymnasium