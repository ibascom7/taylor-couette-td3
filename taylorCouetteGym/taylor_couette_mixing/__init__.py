from gymnasium.envs.registration import register

register(
    id="taylor_couette_mixing/GridWorld-v0",
    entry_point="taylor_couette_mixing.envs:GridWorldEnv",
)
