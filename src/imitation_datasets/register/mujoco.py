"""Register of MuJoCo imitation learning teachers."""
from stable_baselines3 import TD3, SAC, PPO

mujoco = {
    'ant': {
        'name': 'Ant-v4',
        'repo_id': "sb3/td3-Ant-v3",
        'filename': "td3-Ant-v3.zip",
        'threshold': 5822.,
        'algo': TD3
    },
    'ant-1': {
        'name': 'Ant-v3',
        'repo_id': 'sb3/td3-Ant-v3',
        'filename': 'td3-Ant-v3.zip',
        'threshold': 5181,
        'algo': TD3
    },
    'walker': {
        'name': 'Walker2d-v3',
        'repo_id': 'sb3/td3-Walker2d-v3',
        'filename': 'td3-Walker2d-v3.zip',
        'threshold': 4703.,
        'algo': TD3
    },
    'swimmer': {
        'name': 'Swimmer-v3',
        'repo_id': 'sb3/td3-Swimmer-v3',
        'filename': 'td3-Swimmer-v3.zip',
        'threshold': 359.,
        'algo': TD3
    },
    'cheetah': {
        'name': 'HalfCheetah-v4',
        'repo_id': 'sb3/td3-HalfCheetah-v3',
        'filename': 'td3-HalfCheetah-v3.zip',
        'threshold': 9709.,
        'algo': TD3
    },
    'hopper': {
        'name': 'Hopper-v4',
        'repo_id': 'sb3/td3-Hopper-v3',
        'filename': 'td3-Hopper-v3.zip',
        'threshold': 3500.,
        'algo': TD3
    },
    'humanoid': {
        'name': 'Humanoid-v3',
        'repo_id': 'sb3/sac-Humanoid-v3',
        'filename': 'sac-Humanoid-v3.zip',
        'threshold': 6251.,
        'algo': SAC
    },
    'invertedpendulum': {
        'name': 'InvertedPendulum-v2',
        'repo_id': 'qgallouedec/ppo-InvertedPendulum-v2-288745441',
        'filename': 'ppo-InvertedPendulum-v2.zip',
        'threshold': 1000.,
        'algo': PPO
    },
}
