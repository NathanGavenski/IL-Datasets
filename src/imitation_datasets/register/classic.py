"""Register classic control imitation teachers."""
from stable_baselines3 import PPO, DQN

classic = {
    'cartpole': {
        'name': 'CartPole-v1',
        'repo_id': "sb3/ppo-CartPole-v1",
        'filename': "ppo-CartPole-v1.zip",
        'threshold': 500.,
        'algo': PPO
    },
    "mountaincar": {
        'name': 'MountainCar-v0',
        'repo_id': "sb3/dqn-MountainCar-v0",
        'filename': "dqn-MountainCar-v0.zip",
        'threshold': -110.,
        'algo': DQN
    },
    "acrobot": {
        'name': 'Acrobot-v1',
        'repo_id': "sb3/dqn-Acrobot-v1",
        'filename': "dqn-Acrobot-v1.zip",
        'threshold': -75.,
        'algo': DQN
    }
}
