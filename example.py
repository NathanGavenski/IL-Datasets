"""Example for imitation_datasets usage."""
from imitation_datasets import get_args
from imitation_datasets import Controller
from imitation_datasets import Policy, Experts
from imitation_datasets import baseline_enjoy, baseline_collate


if __name__ == '__main__':
    from stable_baselines3 import DQN

    args = get_args()

    # Example on how to register a new policy
    Experts.register(
        "mountaincar",
        Policy(
            name="MountainCar-v0",
            repo_id="sb3/dqn-MountainCar-v0",
            filename="dqn-MountainCar-v0.zip",
            threshold=-110.,
            algo=DQN,
        )
    )

    # How to start the creation of a dataset
    controller = Controller(baseline_enjoy, baseline_collate, args.episodes, args.threads)
    controller.start(args)
