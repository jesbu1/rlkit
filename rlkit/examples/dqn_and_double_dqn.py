"""
Run DQN on CartPole-v0.
"""

import gym
import gym_minigrid
import sys
import random
from torch import nn as nn

from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy
from rlkit.policies.argmax import ArgmaxDiscretePolicy
from rlkit.torch.dqn.dqn import DQNTrainer
from rlkit.torch.dqn.double_dqn import DoubleDQNTrainer
from rlkit.torch.networks import Mlp
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.launchers.launcher_util import setup_logger, set_seed
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm


def experiment(variant):
    #expl_env = gym.make('CartPole-v0').env
    #eval_env = gym.make('CartPole-v0').env
    skills = variant['env_kwargs']['skills']
    train = variant['env_kwargs']['train']
    expl_env = gym.make('MiniGrid-FourRoomsSkills-v0', train=train, skills=skills)
    eval_env = gym.make('MiniGrid-FourRoomsSkills-v0', train=train, skills=skills)
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.n
    hidden_size = variant['hidden_size']
    qf = Mlp(
        hidden_sizes=[hidden_size, hidden_size],
        input_size=obs_dim,
        output_size=action_dim,
    )
    target_qf = Mlp(
        hidden_sizes=[hidden_size, hidden_size],
        input_size=obs_dim,
        output_size=action_dim,
    )
    qf_criterion = nn.MSELoss()
    eval_policy = ArgmaxDiscretePolicy(qf)
    expl_policy = PolicyWrappedWithExplorationStrategy(
        EpsilonGreedy(expl_env.action_space, prob_random_action=variant['epsilon']),
        eval_policy,
    )
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        expl_policy,
    )
    trainer = DQNTrainer(
        qf=qf,
        target_qf=target_qf,
        qf_criterion=qf_criterion,
        **variant['trainer_kwargs']
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    set_seed(variant['seed'])
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    skills = [[1, 2, 2, 2, 2, 1, 2, 2], [2, 2, 2, 2, 2], [2, 2, 2, 0, 2, 2], [1, 2, 2, 2, 2, 2], [2, 2, 1, 2, 2], [2, 0, 2, 2, 2], [2, 0, 2, 2, 2, 2], [2, 2, 2, 2, 0], [2, 2, 2, 2, 2, 2], [0, 2, 2, 2, 2], [0, 2, 2, 0, 2], [2, 2, 2, 1, 2], [2, 2, 2, 2, 2, 1], [2, 2, 2, 2], [1, 2, 2, 2, 2], [2, 2, 1, 2, 0], [2, 2, 2, 2, 2, 2, 2, 2], [1, 2, 2, 2, 2, 1, 2], [2, 2, 2, 2, 2, 2, 2, 1], [2, 2, 2, 1, 2, 2, 2, 2], [2, 2, 2, 2, 2, 1, 2, 2], [1, 2, 1, 2, 2, 2, 2, 2], [1, 2, 2, 2, 2, 1], [2, 2, 0, 2, 2], [1, 2, 2, 1, 2, 2, 0, 2], [0, 2, 2, 2, 0, 2, 1, 2], [2, 2, 2, 2, 1, 2], [0, 2, 2, 2], [1, 2, 0, 2, 2, 2], [2, 2, 2, 1, 2, 2], [2, 1, 2, 2, 2], [0, 2, 2, 0, 2, 2, 2, 2], [0, 2, 2, 2, 2, 2, 2, 0], [2, 1, 2, 2], [2, 2, 2, 1, 2, 0, 2], [2, 2, 2, 2, 2, 0, 2, 2], [1, 2, 1, 2], [2, 2, 0, 2, 2, 2], [2, 2, 1, 2], [1, 2, 1, 2, 2, 2], [1, 2, 2, 2, 1, 2], [0, 2, 2, 2, 2, 2], [1, 2, 2, 2, 1], [2, 2, 2, 0, 2, 2, 1, 2], [1, 1, 2, 2, 2], [2, 2, 2, 0, 2], [0, 2, 2, 2, 1, 2, 2], [2, 2, 2, 1], [2, 0, 2, 2], [2, 2, 0, 2, 2, 2, 2, 2], [1, 2, 2, 2, 2, 2, 1, 2], [2, 2, 2, 2, 0, 2], [2, 0, 2, 2, 1, 2, 2, 2], [2, 2, 2, 2, 1], [0, 2, 2, 2, 0, 2, 2, 2], [2, 1, 2, 2, 2, 2, 2, 2], [2, 2, 1, 2, 2, 2], [0, 2, 2, 2, 0, 2], [0, 0, 2, 2], [2, 0, 2, 2, 2, 2, 2, 2], [1, 2, 1, 2, 2], [1, 2, 2, 2], [2, 1, 2, 2, 2, 2, 0, 2], [0, 2, 0, 2, 2, 2, 2], [1, 2, 2, 1, 2, 0], [2, 1, 2, 2, 2, 2], [2, 0, 2, 2, 1], [2, 2, 2, 1, 2, 2, 2], [2, 2, 2, 2, 0, 2, 2, 2], [2, 2, 2, 0, 2, 2, 2, 2], [2, 2, 2, 2, 2, 0], [2, 2, 0, 2, 2, 2, 2], [1, 1, 2, 0, 2, 2, 2, 0], [2, 2, 2, 2, 2, 2, 1, 2], [2, 2, 0, 2], [2, 0, 2, 1, 2, 2], [2, 2, 2, 2, 2, 2, 0, 2], [0, 2, 1, 2, 2, 2], [0, 2, 2, 0, 2, 2], [2, 1, 2, 2, 2, 0], [0, 2, 2, 2, 2, 0], [0, 2, 2, 0], [1, 2, 2, 1, 2], [1, 2, 2, 2, 1, 2, 2, 2], [2, 2, 2, 2, 2, 0, 2, 1], [2, 2, 2, 2, 1, 2, 2], [0, 2, 0, 2], [1, 2, 2, 0, 2], [2, 2, 2, 0], [2, 2, 2, 0, 2, 2, 2], [0, 2, 0, 2, 2, 2], [2, 2, 2, 2, 1, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2], [0, 2, 2, 2, 2, 0, 2, 2], [1, 2, 2, 1], [1, 2, 2, 1, 2, 2, 2, 1], [1, 1, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 0], [1, 2, 2, 1, 2, 2, 2, 2], [0, 0, 2, 2, 2], [2, 2, 1, 2, 2, 0, 2], [2, 1, 2, 2, 0, 2], [1, 2, 2, 1, 2, 2], [0, 2, 0, 2, 2], [0, 2, 0, 2, 1, 2], [1, 2, 2, 2, 2, 2, 2, 1], [2, 2, 2, 1, 2, 0], [1, 2, 2, 2, 2, 2, 2], [1, 1, 2, 2, 0], [2, 2, 2, 2, 2, 0, 2], [1, 2, 1, 2, 0], [1, 1, 2, 2, 2, 2], [1, 2, 2, 2, 2, 2, 2, 2], [2, 1, 2, 0, 2], [0, 2, 2, 2, 0], [2, 2, 2, 2, 0, 2, 2], [0, 2, 2, 2, 2, 2, 1, 2], [2, 2, 1, 2, 2, 2, 2, 0], [1, 2, 2, 1, 2, 0, 2, 2], [1, 2, 1, 2, 2, 2, 2, 0], [2, 2, 0, 2, 2, 1, 2, 2], [2, 2, 2, 1, 2, 0, 2, 2], [0, 0, 2, 2, 2, 1, 2, 2], [1, 2, 2, 0, 2, 2, 2, 2], [1, 2, 2, 2, 2, 0], [2, 2, 2, 0, 2, 1, 2, 2], [2, 2, 2, 2, 2, 1, 2], [2, 0, 2, 2, 1, 2, 2], [2, 2, 2, 2, 1, 2, 2, 0], [2, 2, 1, 2, 2, 2, 0, 2], [2, 0, 2, 2, 2, 2, 1, 2], [1, 2, 2, 2, 1, 2, 2, 0], [2, 1, 2, 2, 2, 2, 1, 2], [2, 2, 2, 0, 2, 1], [2, 1, 2, 2, 2, 2, 2, 1], [0, 2, 0, 2, 2, 2, 2, 2], [1, 2, 0, 2, 2, 0, 2], [2, 1, 2, 0, 2, 2], [0, 2, 1, 2, 2], [2, 2, 1, 2, 2, 2, 2, 2], [0, 2, 2, 1, 2], [0, 2, 2, 2, 2, 2, 2, 2], [2, 2, 1, 2, 0, 2], [2, 2, 0, 2, 1], [0, 2, 2, 2, 2, 2, 0, 2], [2, 0, 2, 2, 2, 2, 2, 0], [0, 2, 1, 2], [0, 0, 2, 2, 2, 2, 2, 2], [2, 1, 2, 2, 2, 2, 2], [2, 2, 0, 2, 2, 2, 2, 0], [2, 0, 2, 2, 2, 1, 2, 2], [0, 2, 2, 2, 0, 2, 2], [2, 2, 2, 2, 2, 1, 2, 0], [1, 2, 2, 2, 0, 2, 2], [2, 2, 2, 1, 2, 2, 0, 2], [0, 2, 2, 2, 1], [2, 2, 0, 2, 2, 2, 1, 2], [0, 0, 2, 2, 2, 2], [1, 2, 0, 2, 2, 1, 2, 2], [1, 2, 2, 2, 0], [2, 0, 2, 1, 2], [2, 2, 1, 2, 0, 2, 2, 1], [1, 2, 0, 2], [0, 2, 2, 2, 2, 2, 2], [2, 0, 2, 2, 0], [2, 1, 2, 2, 2, 2, 2, 0], [1, 1, 2, 0, 2, 2], [1, 2, 2, 1, 2, 2, 2], [1, 2, 2, 2, 2, 0, 2, 2], [1, 2, 2, 0, 2, 2], [1, 2, 2, 2, 0, 2], [1, 1, 2, 0, 2], [0, 2, 2, 1, 2, 2], [0, 2, 2, 2, 1, 2], [2, 0, 2, 2, 2, 2, 2], [2, 0, 2, 2, 0, 2], [0, 2, 2, 1, 2, 2, 2, 1], [2, 2, 0, 2, 2, 0, 2], [2, 2, 0, 2, 2, 1], [0, 2, 2, 0, 2, 2, 2, 1], [1, 2, 2, 2, 2, 0, 2], [2, 1, 2, 0], [2, 0, 2, 1, 2, 2, 2, 2], [2, 0, 2, 2, 1, 2], [2, 2, 1, 2, 2, 2, 2], [2, 2, 2, 2, 0, 2, 2, 1], [2, 1, 2, 2, 2, 0, 2, 2], [1, 2, 2, 2, 1, 2, 2], [2, 1, 2, 0, 2, 2, 2, 2], [0, 2, 2, 2, 1, 2, 2, 2], [2, 0, 2, 1], [2, 2, 1, 2, 2, 0], [0, 2, 2, 2, 2, 1], [2, 2, 1, 2, 0, 2, 2, 2], [1, 2, 2, 2, 0, 2, 2, 2], [2, 2, 0, 2, 2, 0], [2, 1, 2, 2, 1, 2, 2], [2, 1, 2, 2, 0, 2, 2, 2], [1, 2, 0, 2, 2]]
    variant = dict(
        algorithm="DQN",
        version="normal",
        replay_buffer_size=int(1E6),
        seed=random.randint(0, 100000),
        epsilon=float(sys.argv[1]),
        hidden_size=32,
        algorithm_kwargs=dict(
            num_epochs=1000,
            num_eval_steps_per_epoch=5000,
            num_trains_per_train_loop=500,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=100,
            batch_size=256,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            learning_rate=3E-4,
        ),
        env_kwargs=dict(
            #skills=[[2,2,2], [1,1,1], [0,0,0], [1], [2], [0], [1,2,0], [0,1,2], [0,1,1,1,1,1], [0,2,2,2,2,2], [2,2,2,2], [2,2,2,1,1,2]],
            skills=skills[:200],
            train=True,
        )
    )
    setup_logger('dqn-MinigridFourRoomsSkills-200-skills', variant=variant)
    # ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant)
