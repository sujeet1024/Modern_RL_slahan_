# Modern_RL_slahan_
(An Overview of Reinforcement Learning Algorithms)

This is an ongoing project,
In this repo, the author (Sujeet) presents implementations of important (modern/) Reinforcement Learning algorithms in various gymnasium, Atari2600 environments
 <!-- and a sawyer (grasping) environment -->
, all in simulation.
<!-- Along with their implementations the author reports their performances and insights on different algorithms, also discussing the settings of optimal performance amongst the environments experimented. -->

<!-- The author also provides a short background on classical RL (from Sutton and Barto) -->

This project shall cover,
- Important (foundational) Algorithms
    Classic (Tabular)
    - Q-learning
    - SARSA
    - Expected SARSA
    - Value Iteration
    - Policy Iteration
    DVFs
    - DQN
    - Double DQN
    - DVN (for AC algorithms - included with individual algorithm implementations)
    PGs
    - REINFORCE
    - Actor Critic (reinforce with vf)
    - Off-policy Actor Critic
    - A2C (Advantage, Parallel)
    - Soft Actor Critic
    DPGs
    - DDPG
    - Twin Delayed DDPG (TD3)
    <!-- NGs
    - Natural Gradients
    - TRPO
    - PPO
    FPGs
    - FPO -->

<!-- - OFFLINE-RL (Data from expert online learnt agents for offline RL was used)
    - C-learning
    - Implicit Q-Learning
    - Advantage Weighted Actor Critic (AWAC)
    - COMBO
    - Trajectory Transformer -->

<!-- - Exploration strategies
    - Count based (Optimistic)
    - Max Ent (Information Gain)
        - VIME
    - Thompson sampling (Probability Matching, Posterior Sampling)
    - DIAYN (info theory)
    - Decoupling Exploration and Policy Optimization: Uncertainty Guided Tree Search for Hard Exploration (Zak Muhammadi) -->

<!-- - Other methods (Supervised Imitation Learning)
    - Decision Transformer
    - Diffusion Policies
    Although the author doesn't consider these as RL or DeepRL methods, but in light of buzz around those their comparisons are also provided -->

All the environments used here are directly taken from gymnasium or are built on top of gymnasium


To use the code base, modify the data/config.yml file or the main.py file, there would likely not be need to change any other files, unless you want to experiment with it.
<!-- (currently only discrete (vector and image) observations are supported with DQN agent, Replay Buffer and only epsilon-greedy exploration). -->

Simply run `python main.py`

First install dependencies with pip using `requirements.txt`, then install the rlm package with `pip install -e .`