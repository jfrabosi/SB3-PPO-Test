from gymnasium.envs.registration import register

register(
    id='SerialStewartPlatform/SerialStewartPlatform-v0',
    entry_point='envs:SerialStewartPlatform',
    max_episode_steps=5000
)