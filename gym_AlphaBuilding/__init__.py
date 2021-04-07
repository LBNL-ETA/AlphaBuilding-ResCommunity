from gym.envs.registration import register

register(
    id='AlphaRes-v0',
    entry_point='gym_AlphaBuilding.envs:AlphaResEnv',
)