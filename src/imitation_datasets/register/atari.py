"""Register Atari imitation teachers."""
from stable_baselines3 import PPO
from sb3_contrib import QRDQN

atari = {
    'asteroids': {
        'name': 'AsteroidsNoFrameskip-v4',
        'repo_id': "sb3/ppo-AsteroidsNoFrameskip-v4",
        'filename': "ppo-AsteroidsNoFrameskip-v4.zip",
        'threshold': 2439.,
        'algo': PPO
    },
    'spaceinvaders': {
        'name': 'SpaceInvadersNoFrameskip-v4',
        'repo_id': 'meln1k/qrdqn-SpaceInvadersNoFrameskip-v4',
        'filename': 'qrdqn-SpaceInvadersNoFrameskip-v4.zip',
        'threshold': 2581.,
        'algo': QRDQN
    },
    'breakout': {
        'name': 'BreakoutNoFrameskip-v4',
        'repo_id': 'sb3/ppo-BreakoutNoFrameskip-v4',
        'filename': 'ppo-BreakoutNoFrameskip-v4.zip',
        'threshold': 398.,
        'algo': PPO
    },
    'qbert': {
        'name': 'QbertNoFrameskip-v4',
        'repo_id': 'Corianas/ppo-QbertNoFrameskip-v4_4',
        'filename': 'ppo-QbertNoFrameskip-v4.zip',
        'threshold': 19340.,
        'algo': PPO
    },
    'beamrider': {
        'name': 'BeamriderNoFrameskip-v4',
        'repo_id': 'sb3/qrdqn-BeamRiderNoFrameskip-v4',
        'filename': 'qrdqn-BeamRiderNoFrameskip-v4.zip',
        'threshold': 15785.,
        'algo': QRDQN
    },
    'pong': {
        'name': 'PongNoFrameskip-v4',
        'repo_id': 'sb3/ppo-PongNoFrameskip-v4',
        'filename': 'ppo-PongNoFrameskip-v4.zip',
        'threshold': 21.,
        'algo': PPO
    },
    'enduro': {
        'name': 'EnduroNoFrameskip-v4',
        'repo_id': 'sb3/qrdqn-EnduroNoFrameskip-v4',
        'filename': 'qrdqn-EnduroNoFrameskip-v4.zip',
        'threshold': 2827.7,
        'algo': QRDQN
    },
    'roadrunner': {
        'name': 'RoadRunnerNoFrameskip-v4',
        'repo_id': 'sb3/ppo-RoadRunnerNoFrameskip-v4',
        'filename': 'ppo-RoadRunnerNoFrameskip-v4.zip',
        'threshold': 970.,
        'algo': PPO
    },
    'seaquest': {
        'name': 'qrdqn-SeaquestNoFrameskip-v4',
        'repo_id': 'sb3/qrdqn-SeaquestNoFrameskip-v4',
        'filename': 'qrdqn-SeaquestNoFrameskip-v4.zip',
        'threshold': 2562.,
        'algo': QRDQN
    }
}
