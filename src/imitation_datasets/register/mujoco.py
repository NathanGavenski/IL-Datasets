"""Register of MuJoCo imitation learning teachers."""
from stable_baselines3 import TD3, SAC
from sb3_contrib import TQC

mujoco = {
    'ant': {
        'name': 'Ant-v3',
        'repo_id': "sb3/td3-Ant-v3",
        'filename': "td3-Ant-v3.zip",
        'threshold': 5822.,
        'algo': TD3
    },
    'ant-1': {
        'name': 'Ant-v3',
        'repo_id': 'sb3/sac-Ant-v3',
        'filename': 'sac-Ant-v3.zip',
        'threshold': 5181,
        'algo': SAC
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
        'name': 'HalfCheetah-v3',
        'repo_id': 'sb3/tqc-HalfCheetah-v3',
        'filename': 'tqc-HalfCheetah-v3.zip',
        'threshold': 9709.,
        'algo': TD3
    },
    'hopper': {
        'name': 'Hopper-v3',
        'repo_id': 'sb3/tqc-Hopper-v3',
        'filename': 'tqc-Hopper-v3.zip',
        'threshold': 3752.,
        'algo': TQC
    },
    'humanoid': {
        'name': 'Humanoid-v3',
        'repo_id': 'sb3/tqc-Humanoid-v3',
        'filename': 'tqc-Humanoid-v3.zip',
        'threshold': 6866.,
        'algo': TQC
    },
}
