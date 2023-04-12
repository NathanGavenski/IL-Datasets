from stable_baselines3 import PPO

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
        'repo_id': "sb3/ppo-MountainCar-v0",
        'filename': "ppo-MountainCar-v0.zip",
        'threshold': -108.,
        'algo': PPO
    }
}