import os
import random
import warnings
from functools import partial
from typing import Callable, Dict, Tuple, Union, Optional, List

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector.utils import spaces

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork
from torch import Tensor
from torchrl.modules import MaskedCategorical

from hive_rl_simulator.agent import PlayerUNet, PlayerUNetBackBone
from hive_rl_simulator.game import MAX_PIECES, HiveGame, BOARD_SIZE
from hive_rl_simulator.gym_wrapper import GymEnvAdapter, Player, GymEnvSelfPlayAdapter
import torch.nn as nn
import wandb
from wandb.integration.sb3 import WandbCallback


class PlayerUNetSB3FeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, player_net: PlayerUNet):
        feature_dim = 64
        super().__init__(observation_space, features_dim=feature_dim)
        self.player_net = player_net

    def forward(self, observations: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.player_net.forward(**observations)


def env_builder(
        enemy_policy: PlayerUNet,
        num_ants: int = 4,
        num_spiders: int = 4,
        num_grasshoppers=4,
        max_episode_steps=48,
        render_mode="rgb_array"
):
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    env = GymEnvAdapter(
        game=HiveGame.from_setup(num_ants=num_ants, num_spiders=num_spiders, num_grasshoppers=num_grasshoppers),
        render_mode="rgb_array"
    )
    enemy_player = Player(enemy_policy, obs_space=env.observation_space, action_space=env.action_space)
    return GymEnvSelfPlayAdapter(
        enemy_player,
        render_mode=render_mode,
        game=HiveGame.from_setup(num_ants=num_ants, num_spiders=num_spiders, num_grasshoppers=num_grasshoppers),
        max_episode_steps=max_episode_steps
    )


class PlayerUNetSb3Wrapper(nn.Module):
    def __init__(self, observation: gym.spaces.Space, policy: PlayerUNet, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.features_dim = 64
        self.policy = policy

    def forward(self, obs):
        return self.policy(**obs)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            lr_schedule: Callable[[float], float],
            policy: PlayerUNet,
            *args,
            **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        kwargs["share_features_extractor"] = True
        kwargs["features_extractor_class"] = PlayerUNetSb3Wrapper
        kwargs["features_extractor_kwargs"] = {"policy": policy}
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

    def forward(self, obs: PyTorchObs, deterministic: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
        action_logits, values = self.extract_features(obs, self.features_extractor)

        # distribution = self.action_dist.proba_distribution(action_logits=action_logits)
        distribution = MaskedCategorical(logits=action_logits, mask=torch.flatten(obs["action_mask"], 1).bool())
        actions = distribution.sample()
        # for i, action in enumerate(actions):
        #     if obs["action_mask"].flatten(1)[i].numpy()[action] != 1:
        #         raise ValueError("fuuuuck")
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob

    def get_distribution(self, obs: PyTorchObs) -> Distribution:
        action_logits, value = self.extract_features(obs, self.features_extractor)
        return self.action_dist.proba_distribution(action_logits=action_logits)

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> Tensor:
        return self.get_distribution(observation).get_actions(deterministic=deterministic)

    def evaluate_actions(self, obs: PyTorchObs, actions: Tensor) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        action_logits, values = self.extract_features(obs, self.features_extractor)

        distribution = self.action_dist.proba_distribution(action_logits=action_logits)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def predict_values(self, obs: PyTorchObs) -> Tensor:
        _, values = self.extract_features(obs, self.pi_features_extractor)
        return values


class CustomDQNPolicy(DQNPolicy):
    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            lr_schedule: Callable[[float], float],
            policy: PlayerUNet,
            *args,
            **kwargs,
    ):
        kwargs["features_extractor_class"] = PlayerUNetSb3Wrapper
        kwargs["features_extractor_kwargs"] = {"policy": policy}
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

    def make_q_net(self) -> QNetwork:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor({
            "observation_space": self.observation_space,
            "action_space": self.action_space,
        }, features_extractor=None)
        del net_args["features_dim"]
        return CustomQNetwork(**net_args).to(self.device)


class CustomQNetwork(BasePolicy):
    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Discrete,
            features_extractor: PlayerUNet,
            normalize_images: bool = True,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )
        self.q_net = features_extractor

    def forward(self, obs: PyTorchObs) -> torch.Tensor:
        return self.extract_features(obs, self.features_extractor)

    def _predict(self, observation: PyTorchObs, deterministic: bool = True) -> torch.Tensor:
        q_values = self(observation)
        # Greedy action
        action = q_values.argmax(dim=1).reshape(-1)
        return action


class CustomDQN(DQN):
    def predict(
            self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        add_no_action_to_mask = lambda mask: np.array([0] + list(mask), dtype=np.int8)
        if not deterministic and np.random.rand() < self.exploration_rate:
            if self.policy.is_vectorized_observation(observation):
                if isinstance(observation, dict):
                    n_batch = observation[next(iter(observation.keys()))].shape[0]
                else:
                    n_batch = observation.shape[0]
                action = np.array([
                    self.action_space.sample(add_no_action_to_mask(observation["action_mask"][i].flatten()))
                    for i in range(n_batch)
                ])
            else:
                action = np.array(self.action_space.sample(add_no_action_to_mask(observation["action_mask"].flatten())))
        else:
            action, state = self.policy.predict(observation, state, episode_start, deterministic)
        return action, state


class PlayerUnetDQNWRapper(nn.Module):
    def __init__(self, player: PlayerUNet, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.player = player

    def forward(self, *args, **kwargs) -> torch.Tensor:
        q_values = self.player(*args, **kwargs)[0]
        return q_values


if __name__ == "__main__":
    run_num = 6
    for i in range(10):
        print(f"RUN {i} GLOBAL ITER")
        # Create the vectorized environment
        seed = 42
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # run = wandb.init(
        #     project="hive_rl_simulator",
        #     sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        #     monitor_gym=True,  # auto-upload the videos of agents playing the game
        #     save_code=True,  # optional
        # )

        if i < 1:
            our_policy = PlayerUNet(
                board_info_channels=3,
                max_piece_nums=MAX_PIECES,
                use_action_mask=True,
                board_size=BOARD_SIZE
            )
            enemy_policy = PlayerUNet()
        else:
            our_policy = CustomDQN.load(f"dqn_{run_num}_{i - 1}.path").policy.q_net.features_extractor.policy.player
            enemy_policy = CustomDQN.load(f"dqn_{run_num}_{i - 1}.path").policy.q_net.features_extractor.policy.player

        # Reinitialize agent every seed
        max_episode_steps = 32
        n_envs = 8
        env = make_vec_env(
            partial(
                env_builder,
                enemy_policy=enemy_policy,
                max_episode_steps=max_episode_steps
            ),
            n_envs=n_envs,
            # vec_env_cls=DummyVecEnv,
            vec_env_cls=SubprocVecEnv,
            vec_env_kwargs={"start_method": "spawn"}
        )

        model = CustomDQN(
            partial(CustomDQNPolicy, policy=PlayerUnetDQNWRapper(our_policy)),
            env=env,
            device="cpu",
            verbose=1,
            exploration_fraction=0.2,
            exploration_final_eps=0.07,
            target_update_interval=5000,
            learning_starts=0,
            buffer_size=10000,
            batch_size=128,
            learning_rate=2e-5,
            tensorboard_log="./dqn_hive_self_play/",
            seed=102
        )
        if i >= 1:
            model.load_replay_buffer(f"dqn_{run_num}_{i - 1}_replay_buffer.path")

        model.learn(
            total_timesteps=max_episode_steps * n_envs * 32,
            progress_bar=True,
            tb_log_name="1_iter",
            # callback=WandbCallback(
            #     gradient_save_freq=100,
            #     model_save_path=f"models/{run.id}",
            #     verbose=2,
            # ),
        )
        model.save_replay_buffer(f"dqn_{run_num}_{i}_replay_buffer.path")

        print("LEARNING FINISHED")
        # run.finish()
        model.save(f"dqn_{run_num}_{i}.path")
        del model

        # model = PPO.load(f"/home/vladlenopoleno/ppo_4_{i}.path")
        #
        # obs = env.reset()
    # obs = FlattenObservation(obs)

    # for i in range(1000):
    #     action, _state = model.predict(obs, deterministic=True)
    #     # action, _state = model.predict(FlattenObservation(obs), deterministic=True)
    #     observations, rewards, dones, infos = env.step(action)
    #     # env.render()
    #     if rewards < 0:
    #         print("fail")
    #     if terminated:
    #         obs = env.reset()

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
