import os
import gym
import numpy
from imitation_datasets import get_args
from imitation_datasets import Controller
from imitation_datasets import Policy, Context

async def enjoy(expert: Policy, path, context: Context) -> bool:
    done = False
    expert.load()
    
    env = gym.make(expert.get_environment())
    
    states, actions = [], []
    acc_reward, state = 0, env.reset()
    while not done:
        action, _ = expert.predict(state)
        state, reward, done, _ = env.step(action)
        acc_reward += reward
        states.append(state)
        actions.append(action)
        env.render()
    
    episode = {
        'states': numpy.array(states),
        'actions': numpy.array(actions)
    }
    numpy.savez(f'{path}{context.index}', **episode)
    context.add_log(f'Accumulated reward {acc_reward}')
    return True
        
def collate(path, data) -> bool:
    episodes_starts = []
    states, actions = [], []

    for f in data:
        episode = numpy.load(f'{path}{f}')
        states.append(episode['states'])
        actions.append(episode['actions'])
        episode_starts = numpy.zeros(episode['actions'].shape)
        episode_starts[0] = 1
        episodes_starts.append(episode_starts)

    states = numpy.array(states)
    states = states.reshape((-1, states.shape[-1]))
    actions = numpy.array(actions).reshape(-1)
    episodes_starts = numpy.array(episodes_starts).reshape(-1)

    episode = {
        'states': states,
        'actions': actions,
        'episode_starts': episodes_starts
    }
    numpy.savez(f'{path}cartpole', **episode)

    for f in data:
        os.remove(f'{path}{f}')

    return True

if __name__ == '__main__':
    args = get_args()

    controller = Controller(enjoy, collate, args.episodes, args.threads)
    controller.start(args)
