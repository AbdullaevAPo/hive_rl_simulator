import matplotlib

matplotlib.use('TkAgg')

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
from ray import tune
import gymnasium as gym
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch.nn as nn
from ray.tune import register_env

from hive_rl_simulator.agent import PlayerUNet, state_to_tensor
from hive_rl_simulator.game import MAX_PIECES, HiveGame, BOARD_SIZE
from hive_rl_simulator.gym_wrapper import GymEnvAdapter, GymEnvSelfPlayAdapter, Player, make_env_args
from ray.tune.logger import pretty_print

# ray.init(local_mode=True)
ray.init(
  num_cpus=16,
  num_gpus=1,
)


class PlayerUNetRLLibWrapper(TorchModelV2, nn.Module):
    def __init__(self, obs_space, act_space, num_outputs, *args, player_net: PlayerUNet, **kwargs):
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
        nn.Module.__init__(self)
        self.player_net = player_net

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        _state = state_to_tensor(
            obs["enemy_table"],
            obs["animal_type_table"],
            obs["animal_idx_table"],
            obs["animal_types"],
            obs["action_mask"],
        )
        model_out, value_out = self.player_net(*_state)
        self._value_out = value_out
        return model_out, state

    def value_function(self):
        return self._value_out.flatten()


def env_builder(
        env_config
):
    enemy_policy: PlayerUNet = env_config["enemy_policy"]
    num_ants: int = env_config.get("num_ants") or 3
    num_spiders: int = env_config.get("num_spiders") or 3
    num_grasshoppers = env_config.get("num_grasshoppers") or 3
    max_episode_steps = env_config.get("max_episode_steps") or 24
    return GymEnvSelfPlayAdapter(**make_env_args(
        enemy_policy=enemy_policy,
        num_ants=num_ants,
        num_spiders=num_spiders,
        num_grasshoppers=num_grasshoppers,
        max_episode_steps=max_episode_steps
    ))


enemy_policy = PlayerUNet(board_info_channels=3, max_piece_nums=MAX_PIECES, use_action_mask=True,
                          board_size=BOARD_SIZE)
our_policy = PlayerUNet(board_info_channels=3, max_piece_nums=MAX_PIECES, use_action_mask=True,
                        board_size=BOARD_SIZE)

env = env_builder({"enemy_policy": enemy_policy})
player = Player(our_policy, obs_space=env.observation_space, action_space=env.action_space)
wrapper = PlayerUNetRLLibWrapper(env.observation_space, env.action_space, MAX_PIECES, player_net=our_policy,
                                 model_config={}, name="")

gym.envs.register(
     id='self-play-hive',
     entry_point=GymEnvSelfPlayAdapter,
)
vector_env = gym.vector.AsyncVectorEnv([
    lambda: gym.make("self-play-hive", **make_env_args(enemy_policy)),
    lambda: gym.make("self-play-hive", **make_env_args(enemy_policy))
])
vector_env.reset()

for i in range(2):
    obs, _ = env.reset()
    for i in range(10):
        print(f"STEP {i}")
        action = player.get_step(obs)
        wrapper({"obs": env._get_obs(1)}, [], None)
        env.step(action)
        obs = env._get_obs(player_idx=1)


register_env("my_env", env_builder)

ppo_config = (
    PPOConfig()
    # .rl_module(_enable_rl_module_api=False)
    .environment(
        env="my_env",
        env_config={
            "enemy_policy": enemy_policy,
        },
        observation_space=env.observation_space,
        action_space=env.action_space,
        action_mask_key="action_mask"
    )
    .env_runners(rollout_fragment_length='auto', num_env_runners=15, num_envs_per_env_runner=1)
    .training(
        train_batch_size=128,
        lr=2e-5,
        # gamma=0.99,
        # lambda_=0.9,
        # use_gae=True,
        # clip_param=0.4,
        # grad_clip=None,
        # entropy_coeff=0.1,
        # vf_loss_coeff=0.25,
        # sgd_minibatch_size=64,
        # num_sgd_iter=10,
        model={
            "custom_model": PlayerUNetRLLibWrapper,
            "custom_model_config": {'player_net': our_policy},
            # "vf_share_layers": True,
        }
    )
    .fault_tolerance(
        ignore_env_runner_failures=True,
        recreate_failed_env_runners=True,
    )
    .debugging(log_level="DEBUG")
    .framework(framework="torch")
    .resources(num_gpus=1)
)

algo = ppo_config.build()

for i in range(100):
    result = algo.train()
    print(pretty_print(result))

    if i % 5 == 0:
        checkpoint_dir = algo.save().checkpoint.path
        print(f"Checkpoint saved in directory {checkpoint_dir}")

#
# tune.run(
#     "PPO",
#     name="PPO",
#     stop={"timesteps_total": 100},
#     checkpoint_freq=10,
#     storage_path="~/ray_results/",
#     config=ppo_config.to_dict()
# )
