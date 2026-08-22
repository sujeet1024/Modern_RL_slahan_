from gymnasium.envs.registration import register



register(
    id="ClifWalking-v0",
    entry_point="rlm.tabular_envs.cliff_walking:CliffWalking"
)