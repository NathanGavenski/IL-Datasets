"""Register Atari imitation teachers."""
from stable_baselines3 import PPO, DQN

atari = {
    'spaceinvaders': {
        'name': 'SpaceInvadersNoFrameskip-v4',
        'repo_id': "sb3/ppo-SpaceInvadersNoFrameskip-v4",
        'filename': "ppo-SpaceInvadersNoFrameskip-v4.zip",
        'threshold': 680.,
        'algo': PPO
    },
    'breakout': {
        'name': 'BreakoutNoFrameskip-v4',
        'repo_id': "sb3/ppo-BreakoutNoFrameskip-v4",
        'filename': "ppo-BreakoutNoFrameskip-v4.zip",
        'threshold': 390.,
        'algo': PPO
    },

}
