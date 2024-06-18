import random

import gymnasium as gym
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

from hive_rl_simulator.agent import PlayerUNet
from hive_rl_simulator.game import MAX_PIECES, HiveGame
from hive_rl_simulator.gym_wrapper import GymEnvAdapter




# class PlayerUNetSB3Wrapper(BaseFeaturesExtractor):
#
#     def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
#
#         super(PlayerUNetSB3Wrapper, self).__init__(observation_space, features_dim)
#         self.net = PlayerUNet(board_info_channels=3, max_piece_nums=MAX_PIECES)
#
#         # We assume CxHxW images (channels first)
#         # Re-ordering will be done by pre-preprocessing or wrapper
#         n_input_channels = observation_space.shape[0]
#         self.cnn = nn.Sequential(
#             nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
#             nn.ReLU(),
#             nn.Flatten(),
#         )
#
#         # Compute shape by doing one forward pass
#         with th.no_grad():
#             n_flatten = self.cnn(
#                 th.as_tensor(observation_space.sample()[None]).float()
#             ).shape[1]
#
#         self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
#
#     def forward(self, observations: th.Tensor) -> th.Tensor:
#         return self.linear(self.cnn(observations))

if __name__ == "__main__":
    num_cpu = 4  # Number of processes to use
    # Create the vectorized environment
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Reinitialize agent every seed
    net = PlayerUNet(board_info_channels=3, max_piece_nums=MAX_PIECES)
    env = GymEnvAdapter(
        game=HiveGame.from_setup(
            num_ants=np.random.randint(4)+2,
            num_spiders=np.random.randint(4)+2,
            num_grasshoppers=np.random.randint(4)+2
        ),
        render_mode="human"
    )

    model = PPO('MultiInputPolicy', env, verbose=1, device="cuda")
    model.learn(100000, progress_bar=True)

    print("LEARNING FINISHED")
    model.save("ppo.path")
    del model

    model = PPO.load("ppo.path")

    obs = env.reset()
    # obs = FlattenObservation(obs)

    for i in range(1000):
        action, _state = model.predict(obs[0], deterministic=True)
        # action, _state = model.predict(FlattenObservation(obs), deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        # env.render()
        if reward < 0:
            print("fail")
        if terminated:
            obs = env.reset()

    # vec_env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you.
    # You can choose between `DummyVecEnv` (usually faster) and `SubprocVecEnv`
    # env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)

    # policy_kwargs = dict(
    #     features_extractor_class=CustomCNN,
    #     features_extractor_kwargs=dict(features_dim=128),
    # )
    # model = PPO(, env, verbose=1)
    # model.learn(total_timesteps=25_000)
    #
    # obs = env.reset()
    # for _ in range(1000):
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = env.step(action)
    #     env.render()
